#pragma once
#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <stdexcept>

namespace DirectMemoryWindows {

struct SectionResource {
    HANDLE section = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE;
};


inline SectionResource CreateFileBackedDirectMemorySection(std::uint64_t size) {
    if (size == 0 || size > static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()))
        throw std::invalid_argument("invalid direct memory file backing size");
    std::wstring directory(32768, L'\0');
    const auto length = GetTempPathW(static_cast<DWORD>(directory.size()), directory.data());
    if (length == 0) {
        const auto error = GetLastError();
        throw std::system_error(static_cast<int>(error), std::system_category(), "get direct memory temporary directory");
    }
    if (length >= directory.size())
        throw std::system_error(ERROR_BUFFER_OVERFLOW, std::system_category(), "get direct memory temporary directory");
    directory.resize(length);
    if (directory.back() != L'\\' && directory.back() != L'/') directory.push_back(L'\\');
    static std::atomic<std::uint64_t> sequence{0};
    HANDLE file = INVALID_HANDLE_VALUE;
    for (std::uint32_t attempt = 0; attempt < 128; ++attempt) {
        const auto suffix = std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()) + L"-" +
            std::to_wstring(sequence.fetch_add(1, std::memory_order_relaxed));
        const auto path = directory + L"AnyPS5-direct-memory-" + suffix + L".tmp";
        file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE | GENERIC_EXECUTE | DELETE, 0, nullptr, CREATE_NEW,
            FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (file != INVALID_HANDLE_VALUE) break;
        const auto error = GetLastError();
        if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS)
            throw std::system_error(static_cast<int>(error), std::system_category(), "create direct memory temporary file");
    }
    if (file == INVALID_HANDLE_VALUE)
        throw std::system_error(ERROR_FILE_EXISTS, std::system_category(), "create unique direct memory temporary file");

    auto fail = [&](DWORD error, const char* operation) -> void {
        CloseHandle(file);
        throw std::system_error(static_cast<int>(error), std::system_category(), operation);
    };
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(size);
    if (!SetFileInformationByHandle(file, FileAllocationInfo, &allocation, sizeof(allocation))) fail(GetLastError(), "reserve direct memory file space");
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    if (!SetFilePointerEx(file, end, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) fail(GetLastError(), "size direct memory file");
    FILE_ZERO_DATA_INFORMATION zeroRange{};
    zeroRange.FileOffset.QuadPart = 0;
    zeroRange.BeyondFinalZero.QuadPart = static_cast<LONGLONG>(size);
    DWORD returned = 0;
    if (!DeviceIoControl(file, FSCTL_SET_ZERO_DATA, &zeroRange, sizeof(zeroRange), nullptr, 0, &returned, nullptr))
        fail(GetLastError(), "zero direct memory file contents");
    const auto section = CreateFileMappingW(file, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE,
        static_cast<DWORD>(size >> 32u), static_cast<DWORD>(size), nullptr);
    if (section == nullptr) fail(GetLastError(), "create file-backed direct memory section");
    return {section, file};
}

inline void CloseSectionResource(SectionResource resource) noexcept {
    if (resource.section != nullptr) CloseHandle(resource.section);
    if (resource.file != INVALID_HANDLE_VALUE) CloseHandle(resource.file);
}


}
#endif
