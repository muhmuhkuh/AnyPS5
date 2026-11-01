#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <cstdlib>
#else
#include <cstdlib>
#endif

namespace {
constexpr std::size_t Piece = 64 * 1024;
struct WaitProbe {
    std::uintptr_t expectedOwnerBase = 0;
    std::size_t expectedOwnerBytes = 0;
    std::uintptr_t neighborBase = 0;
    std::size_t neighborBytes = 0;
    std::shared_ptr<const GuestAllocations::Range> owner;
    std::shared_ptr<const GuestAllocations::Range> neighborOwner;
    std::vector<std::pair<std::uintptr_t, std::size_t>> calls;
};
WaitProbe* activeProbe = nullptr;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool releaseOwnerForPinnedRange(std::uintptr_t address, std::size_t bytes) {
    auto& probe = *activeProbe;
    if (probe.calls.empty()) probe.calls.emplace_back(address, bytes);
    const auto end = address + bytes;
    check(address >= probe.neighborBase + probe.neighborBytes || end <= probe.neighborBase,
          "pin waiter was incorrectly called for the independent neighboring Range");
    const auto servedPrefixEnd = probe.expectedOwnerBase + Piece / 2;
    if (address <= probe.expectedOwnerBase && end >= servedPrefixEnd) probe.owner.reset();
    return probe.owner == nullptr;
}

std::shared_ptr<const GuestAllocations::Range> findRange(std::uintptr_t address) {
    auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
    const auto found = std::find_if(lease.begin(), lease.end(), [=](const auto& range) { return range->address == address; });
    check(found != lease.end(), "fixture could not find its registered Range");
    return *found;
}

void run() {
    std::array<std::byte, 3 * Piece> memory{};
    auto* const base = memory.data();
    const auto baseAddress = reinterpret_cast<std::uintptr_t>(base);
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(base, 2 * Piece, true, true, true);
        mutation.Add(base + 2 * Piece, Piece, true, true, true);
    }

    WaitProbe probe;
    probe.neighborBase = baseAddress + 2 * Piece;
    probe.neighborBytes = Piece;
    probe.neighborOwner = findRange(probe.neighborBase);
    activeProbe = &probe;
    GuestAllocations::GuestAllocationsSetPinWaiter_nid_postfix(&releaseOwnerForPinnedRange);
    std::exception_ptr failure;
    bool protectedTail = false;
    try {
        probe.expectedOwnerBase = baseAddress;
        probe.expectedOwnerBytes = 2 * Piece;
        probe.owner = findRange(probe.expectedOwnerBase);
        {
            GuestAllocations::Mutation mutation;
            mutation.Protect(base + Piece, Piece, true, false, true, [] {});
        }
        {
            const auto prefix = *findRange(baseAddress);
            const auto tail = *findRange(baseAddress + Piece);
            check(prefix.address == baseAddress && prefix.bytes == Piece && prefix.readable && prefix.writable && prefix.gpu,
                  "partial Protect changed the prefix bounds or protection");
            check(tail.address == baseAddress + Piece && tail.bytes == Piece && tail.readable && !tail.writable && tail.gpu,
                  "partial Protect did not apply the requested tail protection");
        }
        protectedTail = true;
        check(probe.calls.size() == 1 && probe.calls.back() == std::pair{probe.expectedOwnerBase, probe.expectedOwnerBytes},
              "partial Protect did not wait with full pinned-Range bounds");
        probe.owner.reset();

        probe.calls.clear();
        probe.expectedOwnerBase = baseAddress;
        probe.expectedOwnerBytes = Piece;
        probe.owner = findRange(probe.expectedOwnerBase);
        bool unmapped = false;
        {
            GuestAllocations::Mutation mutation;
            mutation.Unmap(base + Piece / 2, Piece / 2, [&](const void* piece, std::size_t bytes, const void* allocation, bool last) {
                check(piece == base + Piece / 2 && bytes == Piece / 2 && allocation == base && !last,
                      "partial Unmap applied a range other than the requested subrange");
                unmapped = true;
            });
        }
        check(unmapped, "partial Unmap did not proceed after the source owner was released");
        {
            GuestAllocations::Mutation mutation;
            check(mutation.Covers(base, Piece / 2) && !mutation.Covers(base + Piece / 2, Piece / 2),
                  "partial Unmap did not remove only the requested prefix subrange");
            check(mutation.Covers(base + Piece, Piece), "partial Unmap changed the protected tail");
        }
        check(probe.calls.size() == 1 && probe.calls.back() == std::pair{probe.expectedOwnerBase, probe.expectedOwnerBytes},
              "partial Unmap did not wait with full pinned-Range bounds");
        check(probe.neighborOwner != nullptr, "the disjoint neighboring Range was released or replaced");
    } catch (...) {
        failure = std::current_exception();
    }
    GuestAllocations::GuestAllocationsSetPinWaiter_nid_postfix(nullptr);
    activeProbe = nullptr;
    probe.owner.reset();
    probe.neighborOwner.reset();
    {
        GuestAllocations::Mutation cleanup;
        if (cleanup.Overlaps(base, 2 * Piece)) cleanup.Remove(base);
        if (cleanup.Covers(base + 2 * Piece, Piece)) cleanup.Remove(base + 2 * Piece);
    }
    if (failure) std::rethrow_exception(failure);
    check(protectedTail, "partial Protect did not complete");
}
}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIN_WAIT_MS", "200");
#else
    setenv("APS5_PIN_WAIT_MS", "200", 1);
#endif
    try {
        run();
        std::puts("partial Protect/Unmap pin-wait bounds passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
