#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <exception>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

namespace {
using namespace AgcDriver::GuestMemory;
constexpr std::size_t Unit = 65536;
constexpr std::size_t ChunkGuestBytes = 256ull << 20u;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Range {
public:
    explicit Range(std::size_t bytes) : bytes(bytes) {
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, ChunkGuestBytes));
        Require(data != nullptr, "cannot reserve page-state fixture");
    }

    ~Range() {
        {
            GuestAllocations::Mutation mutation;
            for (const auto offset : committed) mutation.Remove(data + offset);
        }
        GuestArena::GuestArenaReset_nid_postfix(data, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, bytes);
    }

    std::byte* Commit(std::size_t offset) {
        auto* at = data + offset;
        GuestArena::GuestArenaCommit_nid_postfix(at, Unit, PAGE_READWRITE, Unit);
        std::memset(at, 0x5b, Unit);
        GuestAllocations::Mutation mutation;
        mutation.Add(at, Unit, true, true, true);
        committed.push_back(offset);
        return at;
    }

    std::byte* Data() const { return data; }

private:
    std::byte* data = nullptr;
    std::size_t bytes;
    std::vector<std::size_t> committed;
};

std::size_t PrivateBytes() {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    Require(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)), "cannot query page-state fixture private commit");
    return counters.PrivateUsage;
}

class NativeInaccessiblePage {
public:
    NativeInaccessiblePage(std::byte* address, std::size_t bytes, bool decommit) : address(address), bytes(bytes), decommit(decommit) {
        if (decommit) Require(VirtualFree(address, bytes, MEM_DECOMMIT) != 0, "cannot natively decommit the cached page");
        else Require(VirtualProtect(address, bytes, PAGE_NOACCESS, &previous) != 0, "cannot natively revoke the cached page");
    }

    ~NativeInaccessiblePage() {
        DWORD ignored = 0;
        const bool restored = decommit ? VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE) == address : VirtualProtect(address, bytes, previous, &ignored) != 0;
        if (!restored) std::terminate();
    }

private:
    std::byte* address;
    std::size_t bytes;
    bool decommit;
    DWORD previous = 0;
};

void CheckCommittedRead() {
    Range range(3 * Unit);
    auto* block = range.Commit(0);
    range.Commit(2 * Unit);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::vector<std::byte> copy(Unit, std::byte{0xcc});
    ReadCommitted(address, copy);
    Require(std::all_of(copy.begin(), copy.end(), [](std::byte value) { return value == std::byte{0x5b}; }), "a complete committed copy changed its source bytes");
    std::fill(copy.begin(), copy.end(), std::byte{0xcc});
    ReadCommittedUnsynced(address, copy);
    Require(std::all_of(copy.begin(), copy.end(), [](std::byte value) { return value == std::byte{0x5b}; }), "an unsynced complete committed copy changed its source bytes");
    copy.resize(3 * Unit);
    for (const bool unsynced : {false, true}) {
        std::fill(copy.begin(), copy.end(), std::byte{0xcc});
        if (unsynced) ReadCommittedUnsynced(address, copy);
        else ReadCommitted(address, copy);
        for (std::size_t index = 0; index < copy.size(); ++index) {
            const auto expected = index >= Unit && index < 2 * Unit ? std::byte{0} : std::byte{0x5b};
            Require(copy[index] == expected, "a sparse committed copy changed data or a previously known hole");
        }
    }
    std::puts("exact and sparse committed reads passed");
}

void CheckStaleCommittedRead(bool sparse, bool unsynced, bool decommit) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const auto page = static_cast<std::size_t>(info.dwPageSize);
    Require(page * 3 <= Unit, "the fixture needs an interior native page");
    const auto bytes = sparse ? 3 * Unit : Unit;
    Range range(bytes);
    auto* block = range.Commit(0);
    if (sparse) range.Commit(2 * Unit);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto described = DescribeCommitted(address, bytes);
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> expected = sparse ?
        std::vector<std::pair<std::uint64_t, std::uint64_t>>{{address, address + Unit}, {address + 2 * Unit, address + 3 * Unit}} :
        std::vector<std::pair<std::uint64_t, std::uint64_t>>{{address, address + Unit}};
    Require(described.ranges == expected && described.whole == !sparse, "the initial committed description is wrong");
    const auto forgotten = ForgetSerial();
    NativeInaccessiblePage change(block + page, page, decommit);
    Require(ForgetSerial() == forgotten && Accessible(block, Unit), "the native change did not leave a stale readable proof");
    Require(Accessible(block, bytes) == !sparse, "the fixture did not select the intended committed-copy branch");
    std::vector<std::byte> copy(bytes, std::byte{0xcc});
    std::string failure;
    try {
        if (unsynced) ReadCommittedUnsynced(address, copy);
        else ReadCommitted(address, copy);
    } catch (const std::runtime_error& error) {
        failure = error.what();
    }
    Require(failure.find("committed guest read") != std::string::npos, "a stale committed proof did not fail with the original safe-copy error");
    std::printf("stale %s %s read refused: %s\n", sparse ? "sparse" : "whole", decommit ? "decommitted" : "protected", failure.c_str());
}

void CheckMappedMetadata() {
    using AgcDriver::Graphics::DccKeys;
    using AgcDriver::Graphics::ReadDccKeys;
    Range range(3 * Unit);
    auto* block = range.Commit(0);
    range.Commit(2 * Unit);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    std::vector<std::byte> copy(Unit, std::byte{0xcc});
    Require(CopyMapped(address, copy) == Compare::Equal, "a complete exact copy was rejected");
    Require(std::all_of(copy.begin(), copy.end(), [](std::byte value) { return value == std::byte{0x5b}; }), "a complete exact copy changed data");
    copy.resize(3 * Unit);
    Require(CopyMapped(address, copy) == Compare::Unmapped, "an exact copy accepted a known hole");
    std::memset(block, 0, Unit);
    Require(ReadDccKeys(address, Unit * 256) == DccKeys::Clear0000, "readable zero metadata lost its clear classification");
    std::memset(block, 0xff, Unit);
    Require(ReadDccKeys(address, Unit * 256) == DccKeys::Uncompressed, "readable expanded metadata lost its classification");
    block[Unit - 1] = std::byte{0};
    Require(ReadDccKeys(address, Unit * 256) == DccKeys::Mixed, "mixed metadata was accepted as uniform");
    Require(ReadDccKeys(address, 3 * Unit * 256) == DccKeys::Unreadable, "a metadata hole became a clear proof");
    std::puts("exact copies and readable, mixed and sparse DCC metadata passed");
}

void CheckStaleMappedCopy(bool decommit) {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const auto page = static_cast<std::size_t>(info.dwPageSize);
    Require(page * 2 <= Unit, "the exact-copy fixture needs two native pages");
    Range range(Unit);
    auto* block = range.Commit(0);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    Require(Accessible(block, Unit), "the exact-copy fixture was not readable");
    NativeInaccessiblePage change(block + page, page, decommit);
    Require(Accessible(block, Unit), "the exact-copy fixture did not retain its stale readable proof");
    std::vector<std::byte> copy(Unit, std::byte{0});
    Require(CopyMapped(address, copy) == Compare::Unmapped, "a failed or partial native exact copy was accepted");
    std::puts("stale exact copy returned Unmapped");
}

void CheckStaleDccRead(bool decommit, bool partial) {
    using AgcDriver::Graphics::DccKeys;
    using AgcDriver::Graphics::ReadDccKeys;
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const auto page = static_cast<std::size_t>(info.dwPageSize);
    Require(page * 2 <= Unit, "the DCC fixture needs two native pages");
    Range range(Unit);
    auto* block = range.Commit(0);
    std::memset(block, 0, Unit);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto bytes = page * 2;
    Require(AgcDriver::Graphics::DccKeyBytes(bytes * 256) == bytes, "the DCC fixture key count changed");
    Require(ReadDccKeys(address, bytes * 256) == DccKeys::Clear0000, "the initial zero metadata is not a clear");
    BumpCollectEpoch();
    Require(CollectWrites(address, bytes) != 0, "the DCC fixture did not establish a reusable generation");
    NativeInaccessiblePage change(block + (partial ? page : 0), page, decommit);
    Require(Accessible(block, bytes), "the DCC fixture did not retain its stale readable proof");
    Require(CollectWrites(address, bytes) != 0, "the native change did not preserve the cached generation");
    Require(ReadDccKeys(address, bytes * 256) == DccKeys::Unreadable, "failed or partial metadata became a zero clear proof");
    AgcDriver::Graphics::GuestTextureResource resource{};
    resource.baseAddress = address;
    resource.width = 16;
    resource.height = 16;
    resource.mipCount = 1;
    resource.format = 22;
    resource.dimension = AgcDriver::Graphics::TextureDimension::k2D;
    resource.dccAddress = address;
    AgcDriver::Graphics::DccKeyProof proof{};
    Require(AgcDriver::Graphics::ProvedClearKeys(resource, bytes * 256, proof) != DccKeys::Clear0000, "missing metadata became a surface clear");
    Require(proof.generation == 0, "a failed metadata snapshot became a reusable proof");
    std::puts("stale DCC metadata returned Unreadable without a clear proof");
}

void CheckMetadata() {
    Range range(3 * ChunkGuestBytes);
    auto* data = range.Commit(0);
    const auto before = PrivateBytes();
    Require(Accessible(data, Unit, true), "committed fixture is inaccessible");
    const auto after = PrivateBytes();
    const auto added = after > before ? after - before : 0;
    std::printf("page-permission initialization: %zu bytes of private commit\n", added);
    Require(added < (16ull << 20u), "page permissions committed a dense arena table");
    const auto queried = PrivateBytes();
    Require(!Accessible(range.Data() + 2 * ChunkGuestBytes, Unit), "reserved fixture tail is accessible");
    Require(PrivateBytes() < queried + (4ull << 20u), "querying reserved pages allocated untouched metadata chunks");
}

void CheckBoundaries() {
    constexpr auto span = 3 * ChunkGuestBytes;
    Range range(span);
    const std::array<std::size_t, 4> offsets{0, ChunkGuestBytes - Unit, ChunkGuestBytes, 2 * ChunkGuestBytes + Unit};
    for (const auto offset : offsets) range.Commit(offset);
    auto* block = range.Data();
    std::atomic<bool> begin{false};
    std::atomic<unsigned> errors{0};
    std::vector<std::thread> readers;
    for (unsigned reader = 0; reader < 8; ++reader) {
        readers.emplace_back([&] {
            while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
            for (unsigned round = 0; round < 500; ++round) {
                if (!Accessible(block + offsets.back(), Unit, true)) errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    begin.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();
    Require(errors.load() == 0, "concurrent page-state publication changed access");
    for (const auto offset : offsets) {
        std::array<std::byte, 16> copy{};
        Read(reinterpret_cast<std::uint64_t>(block + offset), copy);
        Require(std::all_of(copy.begin(), copy.end(), [](std::byte value) { return value == std::byte{0x5b}; }), "page cache changed guest bytes");
    }
    Require(Accessible(block + ChunkGuestBytes - Unit, 2 * Unit, true), "adjacent chunks lost their boundary pages");
    const auto described = DescribeCommitted(reinterpret_cast<std::uint64_t>(block), span, true);
    std::uint64_t total = 0;
    for (const auto& [first, last] : described.ranges) total += last - first;
    Require(!described.whole && total == offsets.size() * Unit, "reserved holes were reported committed");
    auto* changed = block + offsets[1];
    const auto protect = [&](bool writable) {
        GuestAllocations::Mutation mutation;
        mutation.Protect(changed, Unit, true, writable, true, [&] {
            const DWORD protection = writable ? PAGE_READWRITE : PAGE_READONLY;
            DWORD previous = 0;
            Require(VirtualProtect(changed, Unit, protection, &previous) != 0, "cannot protect page-state fixture");
            GuestArena::GuestArenaSetProtection_nid_postfix(reinterpret_cast<std::uintptr_t>(changed), Unit, protection);
        });
    };
    protect(false);
    Require(Accessible(changed, Unit) && !Accessible(changed, Unit, true), "protection invalidation retained writable state");
    protect(true);
    Require(Accessible(changed, Unit, true), "restored writable state was not discovered");
    GuestArena::GuestArenaReset_nid_postfix(block, Unit);
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(reinterpret_cast<std::uintptr_t>(block), Unit);
    Require(!Accessible(block, Unit), "decommitted pages remained readable");
    GuestArena::GuestArenaCommit_nid_postfix(block, Unit, PAGE_READWRITE, Unit);
    Require(Accessible(block, Unit, true), "access observer did not invalidate inaccessible state after commit");
    std::puts("page-state boundaries, publication, holes and invalidation passed");
}
}

int main(int argc, char** argv) {
    try {
        Require(GuestArena::GuestArenaAvailable_nid_postfix(), "page-state fixture needs the guest arena");
        if (argc == 2 && std::string_view(argv[1]) == "--metadata") CheckMetadata();
        else if (argc == 2 && std::string_view(argv[1]) == "--committed-read") CheckCommittedRead();
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-committed-whole") CheckStaleCommittedRead(false, false, false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-committed-sparse") CheckStaleCommittedRead(true, false, false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-committed-unsynced") CheckStaleCommittedRead(false, true, false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-committed-decommit") CheckStaleCommittedRead(true, true, true);
        else if (argc == 2 && std::string_view(argv[1]) == "--mapped-metadata") CheckMappedMetadata();
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-mapped-protect") CheckStaleMappedCopy(false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-mapped-decommit") CheckStaleMappedCopy(true);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-dcc-protect") CheckStaleDccRead(false, false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-dcc-decommit") CheckStaleDccRead(true, false);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-dcc-partial-protect") CheckStaleDccRead(false, true);
        else if (argc == 2 && std::string_view(argv[1]) == "--stale-dcc-partial-decommit") CheckStaleDccRead(true, true);
        else CheckBoundaries();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
