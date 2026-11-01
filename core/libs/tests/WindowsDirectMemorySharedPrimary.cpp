#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/DirectMemory/WindowsSection.hpp"
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t PageBytes = 0x4000;
constexpr std::size_t Bytes = 2u << 20u;
constexpr std::size_t SmallPageBytes = 4096;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class ArenaRegion {
public:
    ArenaRegion() : base(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Bytes)) {
        Require(base != nullptr, "could not allocate guest arena test range");
    }
    ArenaRegion(const ArenaRegion&) = delete;
    ArenaRegion& operator=(const ArenaRegion&) = delete;
    ~ArenaRegion() {
        if (base == nullptr) return;
        try { GuestArena::GuestArenaReset_nid_postfix(base, Bytes); } catch (...) {}
        GuestArena::GuestArenaRelease_nid_postfix(base, Bytes);
    }
    void* base;
};

class PagefileSection {
public:
    PagefileSection() : handle(CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(Bytes), nullptr)) {
        Require(handle != nullptr, "could not create the default-control section");
    }
    PagefileSection(const PagefileSection&) = delete;
    PagefileSection& operator=(const PagefileSection&) = delete;
    ~PagefileSection() { if (handle != nullptr) CloseHandle(handle); }
    HANDLE handle;
};

class FileSection {
public:
    FileSection() : resource(DirectMemoryWindows::CreateFileBackedDirectMemorySection(Bytes)) {
        Require(resource.section != nullptr, "could not create the file-backed direct-memory section");
    }
    FileSection(const FileSection&) = delete;
    FileSection& operator=(const FileSection&) = delete;
    ~FileSection() { DirectMemoryWindows::CloseSectionResource(resource); }
    DirectMemoryWindows::SectionResource resource;
};

std::vector<void*> Collect(std::uintptr_t address, std::size_t bytes, bool fresh) {
    std::vector<void*> pages(bytes / SmallPageBytes);
    auto count = pages.size();
    const bool ok = fresh
        ? GuestArena::GuestArenaCollectFreshWrites_nid_postfix(address, bytes, pages.data(), &count, true)
        : GuestArena::GuestArenaCollectWrites_nid_postfix(address, bytes, pages.data(), &count, true);
    Require(ok, fresh ? "fresh collection refused shared direct-memory view" : "dirty collection refused shared direct-memory view");
    pages.resize(count);
    return pages;
}

bool ContainsPage(const std::vector<void*>& pages, std::uintptr_t address) {
    const auto expected = address & ~std::uintptr_t{SmallPageBytes - 1};
    return std::any_of(pages.begin(), pages.end(), [expected](const void* page) {
        return reinterpret_cast<std::uintptr_t>(page) == expected;
    });
}

MEMORY_BASIC_INFORMATION Query(const void* address) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(address, &memory, sizeof(memory)) == sizeof(memory), "VirtualQuery failed for direct-memory view");
    return memory;
}

void RequireMapped(const void* address, const char* message) {
    const auto memory = Query(address);
    Require(memory.State == MEM_COMMIT && memory.Type == MEM_MAPPED, message);
}

void CheckDefaultPrivateControl() {
    const auto* mode = std::getenv("APS5_DIRECT_MEMORY_SHARED");
    Require(mode == nullptr || std::strcmp(mode, "0") == 0, "default control must run in a separate process with shared mode disabled");
    PagefileSection section;
    {
        ArenaRegion region;
        GuestArena::GuestArenaMapDirect_nid_postfix(region.base, Bytes, section.handle, 0, PAGE_READWRITE);
        const auto memory = Query(region.base);
        Require(memory.State == MEM_COMMIT && memory.Type == MEM_PRIVATE, "shared-primary mode is not opt-in or default direct mapping stopped being resident private");
        auto* bytes = static_cast<volatile std::uint8_t*>(region.base);
        bytes[0] = 0x63;
        Require(bytes[0] == 0x63, "default resident control stopped accepting writes");
        GuestArena::GuestArenaReset_nid_postfix(region.base, Bytes);
        GuestArena::GuestArenaForgetSection_nid_postfix(section.handle);
    }
}

void CheckDirectSharedPrimary() {
    Require(GuestArena::GuestArenaAvailable_nid_postfix(), "guest arena is unavailable");
    FileSection backing;
    const auto section = backing.resource.section;
    {
        ArenaRegion first;
        auto* a = static_cast<volatile std::uint8_t*>(first.base);
        GuestArena::GuestArenaMapDirect_nid_postfix(first.base, Bytes, section, 0, PAGE_READWRITE);
        RequireMapped(first.base, "GuestArenaMapDirect did not create a MEM_MAPPED primary view");
        Require(a[0] == 0, "fresh shared direct memory was not zero initialized");

        a[0x80] = 0x5a;
        GuestArena::GuestArenaReset_nid_postfix(first.base, Bytes);
        GuestArena::GuestArenaMapDirect_nid_postfix(first.base, Bytes, section, 0, PAGE_READWRITE);
        RequireMapped(first.base, "full reset/remap did not restore a mapped primary view");
        Require(a[0x80] == 0x5a, "full reset/remap lost file-backed bytes");

        ArenaRegion second;
        auto* b = static_cast<volatile std::uint8_t*>(second.base);
        GuestArena::GuestArenaMapDirect_nid_postfix(second.base, Bytes, section, 0, PAGE_READWRITE);
        RequireMapped(second.base, "the aliased GuestArenaMapDirect view is not MEM_MAPPED");
        Require(a[0] == 0 && b[0] == 0, "fresh shared direct memory did not read as zero");

        const auto fresh = Collect(reinterpret_cast<std::uintptr_t>(first.base), Bytes, true);
        Require(!fresh.empty(), "new direct-memory pages were not reported as fresh");
        Require(Collect(reinterpret_cast<std::uintptr_t>(first.base), Bytes, true).empty(), "cleared fresh direct-memory pages were reported again");

        a[0] = 0x31;
        Require(b[0] == 0x31, "CPU writes were not coherent across shared direct-memory aliases");
        const auto cpuWrites = Collect(reinterpret_cast<std::uintptr_t>(first.base), Bytes, false);
        Require(ContainsPage(cpuWrites, reinterpret_cast<std::uintptr_t>(first.base)), "a guest CPU store did not reach write tracking through the Windows fault path");
        Require(Collect(reinterpret_cast<std::uintptr_t>(first.base), Bytes, false).empty(), "cleared guest CPU write was reported again");

        const auto hostAddress = reinterpret_cast<std::uintptr_t>(second.base) + 64;
        {
            GuestArena::HostWrite hostWrite(reinterpret_cast<void*>(hostAddress), sizeof(std::uint32_t));
            Require(hostWrite.Open(), "GuestArena HostWrite refused a shared direct-memory view");
            const std::uint32_t marker = 0x9a3bc4d5;
            std::memcpy(reinterpret_cast<void*>(hostAddress), &marker, sizeof(marker));
        }
        const std::uint32_t aliasMarker = static_cast<std::uint32_t>(a[64]) | (static_cast<std::uint32_t>(a[65]) << 8) | (static_cast<std::uint32_t>(a[66]) << 16) | (static_cast<std::uint32_t>(a[67]) << 24);
        Require(aliasMarker == 0x9a3bc4d5, "HostWrite bytes were not coherent through the first alias");
        const auto hostWrites = Collect(reinterpret_cast<std::uintptr_t>(second.base), Bytes, false);
        Require(ContainsPage(hostWrites, hostAddress), "GuestArena HostWrite completion was not reported as dirty");

        constexpr std::size_t protectedOffset = 2 * PageBytes;
        constexpr std::size_t resetOffset = PageBytes;
        b[resetOffset + 17] = 0x76;
        DWORD previous = 0;
        Require(VirtualProtect(static_cast<std::uint8_t*>(first.base) + protectedOffset, PageBytes, PAGE_READONLY, &previous), "could not set a mixed direct-view protection");
        GuestArena::GuestArenaSetProtection_nid_postfix(reinterpret_cast<std::uintptr_t>(first.base) + protectedOffset, PageBytes, PAGE_READONLY);
        Require(Query(static_cast<std::uint8_t*>(first.base) + protectedOffset).Protect == PAGE_READONLY, "read-only protection was not present before partial reset");

        GuestArena::GuestArenaReset_nid_postfix(static_cast<std::uint8_t*>(first.base) + resetOffset, PageBytes);
        GuestArena::GuestArenaMapDirect_nid_postfix(static_cast<std::uint8_t*>(first.base) + resetOffset, PageBytes, section, resetOffset, PAGE_READWRITE);
        RequireMapped(static_cast<std::uint8_t*>(first.base) + resetOffset, "partial direct remap did not restore a mapped view");
        Require(a[resetOffset + 17] == 0x76 && b[resetOffset + 17] == 0x76, "partial reset/remap lost shared backing data");
        Require(Query(static_cast<std::uint8_t*>(first.base) + protectedOffset).Protect == PAGE_READONLY, "partial reset/remap changed a neighboring protection run");
        std::uint32_t trackedProtection = 0;
        Require(GuestArena::GuestArenaProtection_nid_postfix(reinterpret_cast<std::uintptr_t>(first.base) + protectedOffset, &trackedProtection) && trackedProtection == PAGE_READONLY,
            "partial reset/remap changed neighboring protection metadata");
        Require(ContainsPage(Collect(reinterpret_cast<std::uintptr_t>(first.base) + resetOffset, PageBytes, true), reinterpret_cast<std::uintptr_t>(first.base) + resetOffset),
            "partial remap did not mark its pages fresh");

        GuestArena::GuestArenaReset_nid_postfix(first.base, Bytes);
        GuestArena::GuestArenaReset_nid_postfix(second.base, Bytes);
        GuestArena::GuestArenaForgetSection_nid_postfix(section);
    }
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--default-control") == 0) {
            CheckDefaultPrivateControl();
            std::puts("GuestArenaMapDirect default resident-private control passed");
        } else if (argc == 2 && std::strcmp(argv[1], "--shared-primary") == 0) {
            const auto* mode = std::getenv("APS5_DIRECT_MEMORY_SHARED");
            Require(mode != nullptr && std::strcmp(mode, "1") == 0, "shared-primary test must run in a separate process with opt-in enabled");
            CheckDirectSharedPrimary();
            std::puts("GuestArenaMapDirect shared-primary mapping, alias coherence, tracking, partial reset and protection passed");
        } else {
            throw std::runtime_error("expected --default-control or --shared-primary");
        }
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "%s\n", failure.what());
        return 1;
    }
}


