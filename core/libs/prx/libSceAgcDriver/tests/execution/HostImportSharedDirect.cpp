#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "prx/libkernel/DirectMemory/WindowsSection.hpp"
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifdef _WIN32

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}

namespace {

namespace Graphics = AgcDriver::Graphics;
using Graphics::Require;

constexpr std::size_t Bytes = 2u << 20u;
constexpr std::size_t ProbeBytes = 4096;
constexpr std::size_t Alignment = 2u << 20u;

class FileSection {
public:
    FileSection() : resource(DirectMemoryWindows::CreateFileBackedDirectMemorySection(Bytes)) {}
    FileSection(const FileSection&) = delete;
    FileSection& operator=(const FileSection&) = delete;
    ~FileSection() { DirectMemoryWindows::CloseSectionResource(resource); }
    DirectMemoryWindows::SectionResource resource;
};

class DirectMapping {
public:
    DirectMapping(HANDLE section, std::uint64_t offset) : address(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Alignment)), section(section), offset(offset) {
        Require(address != nullptr, "cannot allocate guest direct-memory range");
        const auto guestAddress = reinterpret_cast<std::uint64_t>(address);
        Require(AgcDriver::GuestMemory::WriteWatched(), "shared direct-memory probe requires write tracking");
        AgcDriver::GuestMemory::Unwatch(guestAddress, Bytes);
        Require(!AgcDriver::GuestMemory::Watched(guestAddress, Bytes), "the probe failed to exclude the mapping from tracking");
        GuestArena::GuestArenaMapDirect_nid_postfix(address, Bytes, section, offset, PAGE_READWRITE);
        Require(AgcDriver::GuestMemory::Watched(guestAddress, Bytes), "a new direct mapping did not restore excluded tracking coverage");
        Require(AgcDriver::GuestMemory::CollectWritesUncached(guestAddress, Bytes) != 0, "a new direct mapping did not restore write generation tracking");
    }
    DirectMapping(const DirectMapping&) = delete;
    DirectMapping& operator=(const DirectMapping&) = delete;
    ~DirectMapping() {
        if (address == nullptr) return;
        try { GuestArena::GuestArenaReset_nid_postfix(address, Bytes); } catch (...) {}
        GuestArena::GuestArenaRelease_nid_postfix(address, Bytes);
    }
    void* address;
    HANDLE section;
    std::uint64_t offset;
};

class RegisteredAliases {
public:
    RegisteredAliases(void* first, void* second) : first(first), second(second) {
        GuestAllocations::Mutation mutation;
        mutation.Add(first, Bytes, true, true, true);
        mutation.Add(second, Bytes, true, true, true);
    }
    RegisteredAliases(const RegisteredAliases&) = delete;
    RegisteredAliases& operator=(const RegisteredAliases&) = delete;
    ~RegisteredAliases() {
        GuestAllocations::Mutation mutation;
        mutation.Remove(second);
        mutation.Remove(first);
    }
    void* first;
    void* second;
};

void RequireMapped(const void* address, const char* message) {
    MEMORY_BASIC_INFORMATION memory{};
    Require(VirtualQuery(address, &memory, sizeof(memory)) == sizeof(memory), "VirtualQuery failed for shared direct mapping");
    Require(memory.State == MEM_COMMIT && memory.Type == MEM_MAPPED, message);
}

bool Fill(AgcDriver::VulkanDevice& device, std::uint64_t address) {
    const std::array<std::uint32_t, 4> pattern{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const bool filled = device.FillBuffer(address, ProbeBytes, pattern);
    device.WaitIdle();
    return filled;
}

bool CheckSharedDirectImport() {
    const auto* mode = std::getenv("APS5_DIRECT_MEMORY_SHARED");
    Require(mode != nullptr && std::strcmp(mode, "1") == 0, "GPU probe must run in a separate process with shared mode enabled");
    FileSection backing;
    {
        DirectMapping first(backing.resource.section, 0);
        RequireMapped(first.address, "the first real GuestArenaMapDirect view is not MEM_MAPPED");
        DirectMapping second(backing.resource.section, 0);
        RequireMapped(second.address, "the second real GuestArenaMapDirect alias is not MEM_MAPPED");
        RegisteredAliases registered(first.address, second.address);
        auto device = OpenVulkanTestDevice();
        if (!device) return false;
        const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (context.hostImportAlignment == 0) return false;
        const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(first.address));
        Require(Fill(*device, address), "the device could not fill the file-backed shared import");
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Require(Graphics::HostImportCovers(context, address, ProbeBytes), "GPU fill did not retain the imported primary range");
        }
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first.address, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(second.address, ProbeBytes);
        const std::array<std::uint32_t, 4> expected{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
        for (std::size_t offset = 0; offset < ProbeBytes; offset += sizeof(std::uint32_t)) {
            std::uint32_t firstValue = 0;
            std::uint32_t aliasValue = 0;
            std::memcpy(&firstValue, static_cast<const std::uint8_t*>(first.address) + offset, sizeof(firstValue));
            std::memcpy(&aliasValue, static_cast<const std::uint8_t*>(second.address) + offset, sizeof(aliasValue));
            const auto value = expected[(offset / sizeof(std::uint32_t)) % expected.size()];
            Require(firstValue == value, "CPU readback from GPU-imported direct memory differed from the fill");
            Require(aliasValue == value, "the second direct-memory alias did not observe GPU writes");
        }
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Graphics::ClearHostImports(device->Device());
        }
    }
    GuestArena::GuestArenaForgetSection_nid_postfix(backing.resource.section);
    return true;
}

}

int main() {
    try {
        if (!CheckSharedDirectImport()) return VulkanTestSkipped;
        std::puts("file-backed GuestArenaMapDirect GPU import and alias readback passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

#else

int main() {
    return VulkanTestSkipped;
}

#endif
