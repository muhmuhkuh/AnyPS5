#include "prx/libc/include/WindowsMappings.hpp"
#if __has_include("prx/libkernel/DirectMemory/WindowsSection.hpp")
#include "prx/libkernel/DirectMemory/WindowsSection.hpp"
#define ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY 1
#else
#define ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY 0
namespace DirectMemoryWindows {
struct SectionResource {
    HANDLE section = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE;
};
}
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if __has_include(<psapi.h>)
#include <psapi.h>
#define ANYPS5_HAS_PSAPI_TYPES 1
#else
#define ANYPS5_HAS_PSAPI_TYPES 0
#endif

namespace {

constexpr std::size_t PageBytes = 0x4000;
constexpr std::size_t Bytes = 2u << 20u;
constexpr std::size_t HalfBytes = Bytes / 2;

void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

DirectMemoryWindows::SectionResource PagefileSection(std::size_t bytes) {
    const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    if (section == nullptr) throw std::runtime_error("could not create the pagefile-section control");
    return {section, INVALID_HANDLE_VALUE};
}

DirectMemoryWindows::SectionResource CreateSection(std::size_t bytes, bool fileBacked) {
#if ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY
    if (fileBacked) return DirectMemoryWindows::CreateFileBackedDirectMemorySection(bytes);
#else
    static_cast<void>(fileBacked);
#endif
    return PagefileSection(bytes);
}

void CloseSection(DirectMemoryWindows::SectionResource resource) noexcept {
#if ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY
    DirectMemoryWindows::CloseSectionResource(resource);
#else
    if (resource.section != nullptr) CloseHandle(resource.section);
    if (resource.file != INVALID_HANDLE_VALUE) CloseHandle(resource.file);
#endif
}

std::wstring PathForFile(HANDLE file) {
    std::wstring path(32768, L'\0');
    const auto length = GetFinalPathNameByHandleW(file, path.data(), static_cast<DWORD>(path.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    Require(length != 0 && length < path.size(), "could not get the backing-file path");
    path.resize(length);
    return path;
}

bool FileExists(const std::wstring& path) {
    SetLastError(ERROR_SUCCESS);
    const auto attributes = GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES) return true;
    const auto error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return false;
    throw std::runtime_error("could not query the backing-file path");
}

void WaitForFileRemoval(const std::wstring& path) {
    for (unsigned attempt = 0; attempt < 200; ++attempt) {
        if (!FileExists(path)) return;
        Sleep(5);
    }
    Require(false, "the temporary backing file remained after its last mapping closed");
}

struct Reservation {
    GuestArena::WindowsMappings* mappings = nullptr;
    void* base = nullptr;
    std::size_t bytes = 0;
    bool reset = false;

    Reservation(GuestArena::WindowsMappings& owner, std::size_t size) : mappings(&owner), bytes(size) {
        base = owner.Reserve(nullptr, size);
        Require(base != nullptr, "could not reserve guest placeholders");
    }

    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    void Reset() {
        if (base == nullptr || reset) return;
        mappings->Reset(base, bytes);
        reset = true;
    }

    ~Reservation() {
        if (base == nullptr) return;
        if (!reset) {
            try { mappings->Reset(base, bytes); } catch (...) {}
        }
        VirtualFree(base, bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
        VirtualFree(base, 0, MEM_RELEASE);
    }
};

#if ANYPS5_HAS_PSAPI_TYPES
bool QueryResident(const void* address, bool& available) {
    available = false;
    auto module = GetModuleHandleW(L"psapi.dll");
    const bool loaded = module == nullptr;
    if (loaded) module = LoadLibraryW(L"psapi.dll");
    if (module == nullptr) return false;
    using QueryWorkingSetExFn = BOOL (WINAPI*)(HANDLE, PVOID, DWORD);
    const auto query = reinterpret_cast<QueryWorkingSetExFn>(GetProcAddress(module, "QueryWorkingSetEx"));
    if (query == nullptr) {
        if (loaded) FreeLibrary(module);
        return false;
    }
    PSAPI_WORKING_SET_EX_INFORMATION information{};
    information.VirtualAddress = const_cast<void*>(address);
    available = query(GetCurrentProcess(), &information, sizeof(information)) != FALSE;
    const bool resident = available && information.VirtualAttributes.Valid != 0;
    if (loaded) FreeLibrary(module);
    return resident;
}
#endif

void CheckDemandBehavior(const void* address) {
#if ANYPS5_HAS_PSAPI_TYPES
    bool available = false;
    const bool resident = QueryResident(address, available);
    if (available) Require(!resident, "mapping a fresh backing eagerly faulted its untouched middle page into this process");
#else
    static_cast<void>(address);
#endif
}

std::vector<void*> Collect(GuestArena::WindowsMappings& mappings, std::uintptr_t address, std::size_t bytes,
    GuestArena::WindowsMappings::Collection collection) {
    std::vector<void*> pages(bytes / 4096);
    auto count = pages.size();
    Require(mappings.Collect(address, bytes, pages.data(), &count, true, collection), "write-watch collection was refused");
    pages.resize(count);
    return pages;
}

bool ContainsPage(const std::vector<void*>& pages, std::uintptr_t address) {
    const auto expected = address & ~std::uintptr_t{4095};
    return std::any_of(pages.begin(), pages.end(), [expected](const void* page) {
        return reinterpret_cast<std::uintptr_t>(page) == expected;
    });
}

std::size_t AllocationCount(std::uintptr_t address, std::size_t bytes) {
    std::vector<std::uintptr_t> bases;
    const auto end = address + bytes;
    for (auto cursor = address; cursor < end;) {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(reinterpret_cast<void*>(cursor), &memory, sizeof(memory)) == sizeof(memory), "could not query a mapped run");
        const auto allocation = reinterpret_cast<std::uintptr_t>(memory.AllocationBase);
        if (bases.empty() || bases.back() != allocation) bases.push_back(allocation);
        const auto regionEnd = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
        Require(regionEnd > cursor, "mapped run query did not advance");
        cursor = std::min(end, regionEnd);
    }
    return bases.size();
}

void CheckFactoryValidation() {
#if ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY
    bool zeroRejected = false;
    try { static_cast<void>(DirectMemoryWindows::CreateFileBackedDirectMemorySection(0)); }
    catch (const std::invalid_argument&) { zeroRejected = true; }
    Require(zeroRejected, "the file-backed section factory accepted zero capacity");

    bool overflowRejected = false;
    try { static_cast<void>(DirectMemoryWindows::CreateFileBackedDirectMemorySection(std::numeric_limits<std::uint64_t>::max())); }
    catch (const std::invalid_argument&) { overflowRejected = true; }
    Require(overflowRejected, "the file-backed section factory accepted an overflowing size");
#endif
}

void CheckResidentDemotionAndFileLifetime(bool fileBacked) {
    auto& mappings = GuestArena::WindowsMappings::Get();
    auto resource = CreateSection(Bytes, fileBacked);
    Require(resource.section != nullptr, "the test section is invalid");
    const auto sectionKey = resource.section;
    const auto path = resource.file != INVALID_HANDLE_VALUE ? PathForFile(resource.file) : std::wstring{};
    Reservation first(mappings, Bytes);
    Reservation second(mappings, Bytes);
    auto* a = static_cast<std::uint8_t*>(first.base);
    auto* b = static_cast<std::uint8_t*>(second.base);
    mappings.Map(a, Bytes, resource.section, 0, PAGE_READWRITE, true, true);
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(a, &memory, sizeof(memory)) == sizeof(memory) && memory.Type == MEM_PRIVATE, "the first direct-memory view is not resident private memory");
    CheckDemandBehavior(a + Bytes / 2);
    Require(a[0] == 0 && a[Bytes / 2] == 0 && a[Bytes - PageBytes] == 0, "fresh direct memory did not read as zero");

    const auto base = reinterpret_cast<std::uintptr_t>(a);
    const auto residentAllocations = AllocationCount(base, Bytes);
    static_cast<void>(Collect(mappings, base, Bytes, GuestArena::WindowsMappings::Collection::Fresh));
    constexpr std::size_t middle = Bytes / 2;
    constexpr std::size_t tail = Bytes - PageBytes;
    a[0] = 0x31;
    a[middle] = 0x52;
    a[tail] = 0x73;
    const auto written = Collect(mappings, base, Bytes, GuestArena::WindowsMappings::Collection::All);
    Require(written.size() == 3 && ContainsPage(written, base) && ContainsPage(written, base + middle) && ContainsPage(written, base + tail),
        "resident direct-memory writes were not reported at the touched pages");
    Require(Collect(mappings, base, Bytes, GuestArena::WindowsMappings::Collection::All).empty(),
        "cleared resident direct-memory writes were reported again");
    DWORD mixedPrevious = 0;
    Require(VirtualProtect(a + PageBytes, PageBytes, PAGE_READONLY, &mixedPrevious), "could not set a mixed resident protection run");
    mappings.SetProtection(base + PageBytes, PageBytes, PAGE_READONLY);

    mappings.Map(b, Bytes, resource.section, 0, PAGE_READWRITE, true, true);
    std::uintptr_t residentBegin = 0;
    std::uintptr_t residentEnd = 0;
    Require(!mappings.ResidentSpan(base, Bytes, &residentBegin, &residentEnd), "mapping a second direct-memory view did not demote the resident run");
    MEMORY_BASIC_INFORMATION firstView{};
    MEMORY_BASIC_INFORMATION secondView{};
    Require(VirtualQuery(a, &firstView, sizeof(firstView)) == sizeof(firstView) && firstView.Type == MEM_MAPPED,
        "the demoted direct-memory view is not mapped to its backing section");
    Require(VirtualQuery(b, &secondView, sizeof(secondView)) == sizeof(secondView) && secondView.Type == MEM_MAPPED,
        "the second direct-memory view is not mapped to its backing section");
    const auto demotedAllocations = AllocationCount(base, Bytes);
    std::printf("resident demotion allocation count: %zu before, %zu after\n", residentAllocations, demotedAllocations);
    Require(demotedAllocations == residentAllocations,
        "resident demotion split one contiguous backing run into multiple mapped allocations");
    MEMORY_BASIC_INFORMATION mixedView{};
    Require(VirtualQuery(a + PageBytes, &mixedView, sizeof(mixedView)) == sizeof(mixedView) && mixedView.Protect == PAGE_READONLY,
        "demoting a coalesced run did not restore its per-view protection boundary");
    Require(VirtualProtect(a + PageBytes, PageBytes, PAGE_READWRITE, &mixedPrevious), "could not restore the mixed protection run");
    mappings.SetProtection(base + PageBytes, PageBytes, PAGE_READWRITE);
    Require(b[0] == 0x31 && b[middle] == 0x52 && b[tail] == 0x73,
        "resident-to-shared demotion did not copy all marker bytes");
    b[middle + 32] = 0x94;
    Require(a[middle + 32] == 0x94, "writes did not stay coherent between aliases");
    mappings.Reset(a + middle, PageBytes);
    Require(a[0] == 0x31 && a[middle + PageBytes] == b[middle + PageBytes],
        "partial reset of a coalesced demoted run changed neighboring mapped data");
    mappings.Map(a + middle, PageBytes, resource.section, middle, PAGE_READWRITE, true, false);
    Require(a[middle] == 0x52 && b[middle] == 0x52,
        "partial remap after splitting a coalesced demoted run lost backing data or alias coherence");

    DWORD previous = 0;
    Require(VirtualProtect(a, 4096, PAGE_READONLY, &previous), "could not protect the first alias read-only");
    mappings.SetProtection(base, 4096, PAGE_READONLY);
    std::uint32_t trackedProtection = 0;
    MEMORY_BASIC_INFORMATION protectedView{};
    Require(mappings.Protection(base, &trackedProtection) && trackedProtection == PAGE_READONLY &&
        VirtualQuery(a, &protectedView, sizeof(protectedView)) == sizeof(protectedView) && protectedView.Protect == PAGE_READONLY,
        "read-only protection was not retained by the mapping metadata and view");
    Require(VirtualProtect(a, 4096, PAGE_READWRITE, &previous), "could not restore writable protection");
    mappings.SetProtection(base, 4096, PAGE_READWRITE);

    if (resource.file != INVALID_HANDLE_VALUE) {
        Require(FileExists(path), "the temporary backing file disappeared while mappings were live");
        CloseSection(resource);
        Require(a[0] == 0x31 && b[middle + 32] == 0x94, "closing caller handles invalidated mapped alias data");
        first.Reset();
        static_cast<void>(mappings.ForgetSection(sectionKey));
        Require(b[0] == 0x31 && b[middle] == 0x52 && b[tail] == 0x73, "closing the first alias invalidated the remaining alias data");
        b[64] = 0xa6;
        Require(b[64] == 0xa6, "the remaining alias stopped accepting writes after its caller handles closed");
        second.Reset();
        static_cast<void>(mappings.ForgetSection(sectionKey));
        WaitForFileRemoval(path);
    } else {
        first.Reset();
        second.Reset();
        static_cast<void>(mappings.ForgetSection(sectionKey));
        CloseSection(resource);
    }
}

void CheckPartialRemap(bool fileBacked) {
    auto& mappings = GuestArena::WindowsMappings::Get();
    auto firstSection = CreateSection(Bytes, fileBacked);
    auto secondSection = CreateSection(Bytes, fileBacked);
    Reservation destination(mappings, Bytes);
    auto* base = static_cast<std::uint8_t*>(destination.base);
    mappings.Map(base, HalfBytes, firstSection.section, 0, PAGE_READWRITE, false, false, PAGE_READWRITE);
    mappings.Map(base + HalfBytes, HalfBytes, secondSection.section, 0, PAGE_READWRITE, false, false, PAGE_READWRITE);
    CheckDemandBehavior(base + HalfBytes / 2);
    Require(base[0] == 0 && base[HalfBytes / 2] == 0 && base[HalfBytes] == 0 && base[Bytes - PageBytes] == 0,
        "new file-backed direct memory did not read as zero");
    base[HalfBytes - PageBytes] = 0x21;
    base[HalfBytes] = 0x42;
    Require(base[HalfBytes - PageBytes] == 0x21 && base[HalfBytes] == 0x42, "partial direct-memory mappings did not preserve their independent data");

    mappings.Reset(base + HalfBytes, HalfBytes);
    mappings.Map(base + HalfBytes, HalfBytes, secondSection.section, HalfBytes, PAGE_READWRITE, false, false, PAGE_READWRITE);
    Require(base[HalfBytes - PageBytes] == 0x21 && base[HalfBytes] == 0,
        "partial unmap/remap changed the neighboring run or retained stale bytes");
    destination.Reset();
    static_cast<void>(mappings.ForgetSection(firstSection.section));
    static_cast<void>(mappings.ForgetSection(secondSection.section));
    CloseSection(firstSection);
    CloseSection(secondSection);
}

void Run(bool fileBacked) {
    CheckFactoryValidation();
    CheckResidentDemotionAndFileLifetime(fileBacked);
    CheckPartialRemap(fileBacked);
#if !ANYPS5_HAS_FILE_BACKED_DIRECT_MEMORY
    if (fileBacked) Require(false, "file-backed direct-memory factory is not present in this source tree");
#endif
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected --pagefile-control or --file-backed");
        const std::string mode = argv[1];
        if (mode == "--pagefile-control") Run(false);
        else if (mode == "--file-backed") Run(true);
        else throw std::runtime_error("unknown test mode");
        std::puts("direct-memory backing, aliasing, tracking and cleanup passed");
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "%s\n", failure.what());
        return 1;
    }
}
