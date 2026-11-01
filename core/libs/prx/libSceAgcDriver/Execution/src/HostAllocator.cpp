#ifdef _WIN32

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <intrin.h>
#include <malloc.h>
#include <new>
#include <windows.h>

thread_local bool ReportLargeHostReleases_nid_no_patch = false;

namespace {


struct LargeOwner {
    void* pointer = nullptr;
    void* caller = nullptr;
    std::size_t requested = 0;
    std::size_t usable = 0;
};
constexpr std::size_t OwnerSlots = 65536;
LargeOwner largeOwners[OwnerSlots];
SRWLOCK ownerLock = SRWLOCK_INIT;
std::uint64_t ownerMisses = 0;

void TrackLarge(void* pointer, std::size_t bytes, void* caller) noexcept {
    if (pointer == nullptr || bytes < 65536) return;
    const auto usable = _msize(pointer);
    AcquireSRWLockExclusive(&ownerLock);
    const auto start = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) % OwnerSlots;
    bool stored = false;
    for (std::size_t probe = 0; probe < OwnerSlots; ++probe) {
        auto& entry = largeOwners[(start + probe) % OwnerSlots];
        if (entry.pointer == nullptr || entry.pointer == reinterpret_cast<void*>(1)) {
            entry = {pointer, caller, bytes, usable};
            stored = true;
            break;
        }
    }
    if (!stored) ++ownerMisses;
    ReleaseSRWLockExclusive(&ownerLock);
}

void ForgetLarge(void* pointer) noexcept {
    if (pointer == nullptr) return;
    LargeOwner released;
    AcquireSRWLockExclusive(&ownerLock);
    const auto start = (reinterpret_cast<std::uintptr_t>(pointer) >> 4) % OwnerSlots;
    for (std::size_t probe = 0; probe < OwnerSlots; ++probe) {
        auto& entry = largeOwners[(start + probe) % OwnerSlots];
        if (entry.pointer == nullptr) break;
        if (entry.pointer == pointer) {
            released = entry;
            entry.pointer = reinterpret_cast<void*>(1);
            break;
        }
    }
    ReleaseSRWLockExclusive(&ownerLock);
    if (ReportLargeHostReleases_nid_no_patch && released.pointer != nullptr) std::fprintf(stderr, "[host-reclaim-free] pointer=%p caller=%p requested=%zu usable=%zu\n", released.pointer, released.caller, released.requested, released.usable);
}

constexpr std::size_t Granule = 16;
constexpr std::size_t LargestCached = 1024;
constexpr std::size_t Classes = LargestCached / Granule;

constexpr std::uint16_t Depth(std::size_t sizeClass) {
    return sizeClass < 256 / Granule ? 64 : 16;
}

struct Cache {
    void* heads[Classes];
    std::uint16_t counts[Classes];
};

constexpr std::uint16_t KeptLists = 8;

struct Depot {
    SRWLOCK lock = SRWLOCK_INIT;
    void* lists = nullptr;
    std::uint16_t count = 0;
};

Depot depots[Classes];

bool KeepList(std::size_t sizeClass, void* list) noexcept {
    auto& depot = depots[sizeClass];
    AcquireSRWLockExclusive(&depot.lock);
    const bool kept = depot.count < KeptLists;
    if (kept) {
        static_cast<void**>(list)[1] = depot.lists;
        depot.lists = list;
        ++depot.count;
    }
    ReleaseSRWLockExclusive(&depot.lock);
    return kept;
}

void* TakeList(std::size_t sizeClass) noexcept {
    auto& depot = depots[sizeClass];
    AcquireSRWLockExclusive(&depot.lock);
    void* list = depot.lists;
    if (list != nullptr) {
        depot.lists = static_cast<void**>(list)[1];
        --depot.count;
    }
    ReleaseSRWLockExclusive(&depot.lock);
    return list;
}

Cache* const Closed = reinterpret_cast<Cache*>(std::uintptr_t{1});

constexpr unsigned long TlsSlotsOffset = 0x1480;
constexpr DWORD DirectSlots = 64;

constexpr DWORD Unset = TLS_OUT_OF_INDEXES;
constexpr DWORD Failed = TLS_OUT_OF_INDEXES - 1;
std::atomic<DWORD> tlsSlot{Unset};
std::atomic<DWORD> flsSlot{FLS_OUT_OF_INDEXES};

void Drain(Cache* cache) noexcept {
    for (std::size_t sizeClass = 0; sizeClass < Classes; ++sizeClass) {
        while (void* block = cache->heads[sizeClass]) {
            cache->heads[sizeClass] = *static_cast<void**>(block);
            std::free(block);
        }
        cache->counts[sizeClass] = 0;
    }
}

void WINAPI ThreadEnded(void* data) {
    auto* cache = static_cast<Cache*>(data);
    if (cache == nullptr || cache == Closed) return;
    const auto slot = tlsSlot.load(std::memory_order_acquire);
    if (slot < Failed) TlsSetValue(slot, Closed);
    Drain(cache);
    std::free(cache);
}

DWORD Slot() noexcept {
    auto slot = tlsSlot.load(std::memory_order_acquire);
    if (slot != Unset) return slot;
    const DWORD error = GetLastError();
    const DWORD fls = FlsAlloc(ThreadEnded);
    DWORD tls = fls == FLS_OUT_OF_INDEXES ? TLS_OUT_OF_INDEXES : TlsAlloc();
    if (tls == TLS_OUT_OF_INDEXES || tls >= Failed) {
        if (tls != TLS_OUT_OF_INDEXES) TlsFree(tls);
        if (fls != FLS_OUT_OF_INDEXES) FlsFree(fls);
        tls = Failed;
    }
    DWORD expected = Unset;
    if (tlsSlot.compare_exchange_strong(expected, tls, std::memory_order_acq_rel)) {
        if (tls != Failed) flsSlot.store(fls, std::memory_order_release);
        SetLastError(error);
        return tls;
    }
    if (tls != Failed) {
        TlsFree(tls);
        FlsFree(fls);
    }
    SetLastError(error);
    return expected;
}

Cache* SlotValue(DWORD slot) noexcept {
    if (slot < DirectSlots) return reinterpret_cast<Cache*>(__readgsqword(TlsSlotsOffset + slot * sizeof(void*)));
    const DWORD error = GetLastError();
    auto* value = static_cast<Cache*>(TlsGetValue(slot));
    SetLastError(error);
    return value;
}

Cache* CreateCache(DWORD slot) noexcept {
    const auto fls = flsSlot.load(std::memory_order_acquire);
    if (fls == FLS_OUT_OF_INDEXES) return nullptr;
    auto* cache = static_cast<Cache*>(std::calloc(1, sizeof(Cache)));
    if (cache == nullptr) return nullptr;
    const DWORD error = GetLastError();
    if (!FlsSetValue(fls, cache) || !TlsSetValue(slot, cache)) {
        FlsSetValue(fls, nullptr);
        std::free(cache);
        cache = nullptr;
    }
    SetLastError(error);
    return cache;
}

Cache* CurrentCache() noexcept {
    const auto slot = Slot();
    if (slot == Failed) return nullptr;
    auto* cache = SlotValue(slot);
    if (cache == Closed) return nullptr;
    return cache != nullptr ? cache : CreateCache(slot);
}

void* TryAllocate(std::size_t bytes) noexcept {
    if (bytes == 0) bytes = 1;
    if (bytes <= LargestCached) {
        const auto sizeClass = (bytes - 1) / Granule;
        if (auto* cache = CurrentCache()) {
            if (cache->heads[sizeClass] == nullptr) {
                if (void* list = TakeList(sizeClass)) {
                    cache->heads[sizeClass] = list;
                    cache->counts[sizeClass] = Depth(sizeClass);
                }
            }
            if (void* block = cache->heads[sizeClass]) {
                cache->heads[sizeClass] = *static_cast<void**>(block);
                --cache->counts[sizeClass];
                return block;
            }
        }
        return std::malloc((sizeClass + 1) * Granule);
    }
    return std::malloc(bytes);
}

void* Allocate(std::size_t bytes, void* caller) {
    for (;;) {
        if (void* block = TryAllocate(bytes)) {
            TrackLarge(block, bytes, caller);
            return block;
        }
        const auto handler = std::get_new_handler();
        if (handler == nullptr) throw std::bad_alloc();
        handler();
    }
}

void Release(void* block, std::size_t bytes = 0) noexcept {
    if (block == nullptr) return;
    if (_msize(block) >= 65536) ForgetLarge(block);
    if (auto* cache = CurrentCache()) {
        const auto usable = bytes != 0 && bytes % Granule == 0 ? bytes : _msize(block);
        if (usable >= Granule && usable < LargestCached + Granule) {
            const auto sizeClass = usable / Granule - 1;
            if (cache->counts[sizeClass] == Depth(sizeClass) && KeepList(sizeClass, cache->heads[sizeClass])) {
                cache->heads[sizeClass] = nullptr;
                cache->counts[sizeClass] = 0;
            }
            if (cache->counts[sizeClass] < Depth(sizeClass)) {
                *static_cast<void**>(block) = cache->heads[sizeClass];
                cache->heads[sizeClass] = block;
                ++cache->counts[sizeClass];
                return;
            }
        }
    }
    std::free(block);
}

}


void ReportLargeHostOwners_nid_no_patch(std::uint64_t serial) {
    struct Group { void* caller = nullptr; std::uint64_t count = 0; std::uint64_t requested = 0; std::uint64_t usable = 0; };
    Group groups[256]{};
    std::uint64_t count = 0, requested = 0, usable = 0, ungrouped = 0;
    AcquireSRWLockShared(&ownerLock);
    for (const auto& entry : largeOwners) {
        if (entry.pointer == nullptr || entry.pointer == reinterpret_cast<void*>(1)) continue;
        ++count;
        requested += entry.requested;
        usable += entry.usable;
        bool found = false;
        for (auto& group : groups) {
            if (group.caller != nullptr && group.caller != entry.caller) continue;
            group.caller = entry.caller;
            ++group.count;
            group.requested += entry.requested;
            group.usable += entry.usable;
            found = true;
            break;
        }
        if (!found) ++ungrouped;
    }
    const auto misses = ownerMisses;
    ReleaseSRWLockShared(&ownerLock);
    std::fprintf(stderr, "[large-host-owners] build=%llu count=%llu requested=%llu usable=%llu misses=%llu ungrouped=%llu threshold=65536\n", static_cast<unsigned long long>(serial), static_cast<unsigned long long>(count), static_cast<unsigned long long>(requested), static_cast<unsigned long long>(usable), static_cast<unsigned long long>(misses), static_cast<unsigned long long>(ungrouped));
    for (const auto& group : groups) {
        if (group.count == 0) continue;
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(group.caller, &info, sizeof(info));
        std::fprintf(stderr, "[large-host-owner] build=%llu caller=%p module=%p rva=%llu count=%llu requested=%llu usable=%llu\n", static_cast<unsigned long long>(serial), group.caller, info.AllocationBase, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(group.caller) - reinterpret_cast<std::uintptr_t>(info.AllocationBase)), static_cast<unsigned long long>(group.count), static_cast<unsigned long long>(group.requested), static_cast<unsigned long long>(group.usable));
    }
}

void* operator new(std::size_t bytes) { return Allocate(bytes, __builtin_return_address(0)); }
void* operator new[](std::size_t bytes) { return Allocate(bytes, __builtin_return_address(0)); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* block) noexcept { Release(block); }
void operator delete[](void* block) noexcept { Release(block); }
void operator delete(void* block, std::size_t bytes) noexcept { Release(block, bytes); }
void operator delete[](void* block, std::size_t bytes) noexcept { Release(block, bytes); }
void operator delete(void* block, const std::nothrow_t&) noexcept { Release(block); }
void operator delete[](void* block, const std::nothrow_t&) noexcept { Release(block); }

#endif

