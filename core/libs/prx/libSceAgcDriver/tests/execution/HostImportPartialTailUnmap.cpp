#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
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
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}

namespace {
namespace Graphics = AgcDriver::Graphics;
using Graphics::Require;
constexpr std::size_t Window = 64u << 20u;
constexpr std::size_t Bytes = 2u * Window;
constexpr std::size_t ProbeBytes = 4096;
constexpr std::size_t Alignment = 2u << 20u;

class FileSection {
public:
    explicit FileSection(std::uint64_t bytes) : resource(DirectMemoryWindows::CreateFileBackedDirectMemorySection(bytes)) {}
    FileSection(const FileSection&) = delete;
    FileSection& operator=(const FileSection&) = delete;
    ~FileSection() { DirectMemoryWindows::CloseSectionResource(resource); }
    DirectMemoryWindows::SectionResource resource;
};

class DirectMapping {
public:
    DirectMapping(HANDLE section, std::uint64_t offset)
        : address(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Alignment)), section(section), offset(offset) {
        Require(address != nullptr, "cannot allocate guest direct-memory range");
        const auto guestAddress = reinterpret_cast<std::uint64_t>(address);
        Require(AgcDriver::GuestMemory::WriteWatched(), "bounded import test requires write tracking");
        AgcDriver::GuestMemory::Unwatch(guestAddress, Bytes);
        Require(!AgcDriver::GuestMemory::Watched(guestAddress, Bytes), "mapping was not excluded from tracking before MapDirect");
        GuestArena::GuestArenaMapDirect_nid_postfix(address, Bytes, section, offset, PAGE_READWRITE);
        Require(AgcDriver::GuestMemory::Watched(guestAddress, Bytes), "MapDirect did not restore write tracking");
        Require(AgcDriver::GuestMemory::CollectWritesUncached(guestAddress, Bytes) != 0, "MapDirect did not restore write generations");
        MEMORY_BASIC_INFORMATION info{};
        Require(VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT && info.Type == MEM_MAPPED,
                "the primary is not a real MEM_MAPPED GuestArenaMapDirect view");
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

class RegisteredPrimary {
public:
    explicit RegisteredPrimary(void* primary) : primary(primary) {
        GuestAllocations::Mutation mutation;
        mutation.Add(primary, Bytes, true, true, true);
    }
    RegisteredPrimary(const RegisteredPrimary&) = delete;
    RegisteredPrimary& operator=(const RegisteredPrimary&) = delete;
    ~RegisteredPrimary() {
        GuestAllocations::Mutation mutation;
        mutation.Remove(primary);
    }
    void* primary;
};

class ClearHostImportsOnExit {
public:
    explicit ClearHostImportsOnExit(VkDevice device) : device(device) {}
    ClearHostImportsOnExit(const ClearHostImportsOnExit&) = delete;
    ClearHostImportsOnExit& operator=(const ClearHostImportsOnExit&) = delete;
    ~ClearHostImportsOnExit() noexcept {
        try {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            Graphics::ClearHostImports(device);
        } catch (...) {}
    }
private:
    VkDevice device;
};
class RestoreImportPolicy {
public:
    RestoreImportPolicy() {
        Graphics::SetHostImportRangeWindow(Window);
        Graphics::SetHostImportMaxBytes(0);
    }
    ~RestoreImportPolicy() {
        Graphics::SetHostImportBudget(0);
        Graphics::SetHostImportRangeWindow(64ull << 20u);
        Graphics::SetHostImportMaxBytes(0);
    }
};

void HostFill(void* address, std::uint32_t value) {
    GuestArena::HostWrite write(address, ProbeBytes);
    Require(write.Open(), "GuestArena HostWrite did not open for a mapped range");
    auto* words = static_cast<std::uint32_t*>(address);
    std::fill_n(words, ProbeBytes / sizeof(*words), value);
}

void RequireWords(const void* address, std::uint32_t value, const char* message) {
    const auto* words = static_cast<const std::uint32_t*>(address);
    for (std::size_t i = 0; i < ProbeBytes / sizeof(*words); ++i) Require(words[i] == value, message);
}

void FillGpu(AgcDriver::VulkanDevice& device, AgcDriver::Graphics::Recorder& recorder,
             const VkDescriptorBufferInfo& importedBoundary, const VkDescriptorBufferInfo& mirroredBoundary,
             std::uint32_t importedValue, std::uint32_t mirroredValue) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto fill = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto commands = recorder.Commands();
    RecordMemoryBarrier(context, commands,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT);
    fill(commands, importedBoundary.buffer, importedBoundary.offset, ProbeBytes, importedValue);
    fill(commands, mirroredBoundary.buffer, mirroredBoundary.offset, ProbeBytes, mirroredValue);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT);
    recorder.Sync();
}

void RequireReadback(AgcDriver::VulkanDevice& device, AgcDriver::Graphics::Recorder& recorder,
                     const VkDescriptorBufferInfo& prefix, const VkDescriptorBufferInfo& tail,
                     std::uint32_t prefixValue, std::uint32_t tailValue) {
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    AgcDriver::Graphics::Buffer readback(context, 2 * ProbeBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    const auto commands = recorder.Commands();
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferCopy first{prefix.offset, 0, ProbeBytes};
    VkBufferCopy second{tail.offset, ProbeBytes, ProbeBytes};
    copy(commands, prefix.buffer, readback.Handle(), 1, &first);
    copy(commands, tail.buffer, readback.Handle(), 1, &second);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    recorder.Sync();
    readback.Invalidate();
    const auto bytes = readback.Bytes();
    RequireWords(bytes.data(), prefixValue, "GPU readback did not observe the current imported-prefix host bytes");
    RequireWords(bytes.data() + ProbeBytes, tailValue, "GPU readback did not observe the current mirrored-tail host bytes");
}

bool CheckBoundedSharedDirectImport() {
    const auto* shared = std::getenv("APS5_DIRECT_MEMORY_SHARED");
    const auto* cap = std::getenv("APS5_HOST_IMPORT_MIB");
    Require(shared != nullptr && std::strcmp(shared, "1") == 0, "test requires a fresh process with APS5_DIRECT_MEMORY_SHARED=1");
    Require(cap != nullptr && std::strcmp(cap, "64") == 0, "test requires a fresh process with APS5_HOST_IMPORT_MIB=64");

    auto device = OpenVulkanTestDevice();
    if (!device) return false;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    if (context.hostImportAlignment == 0 || Window % context.hostImportAlignment != 0) return false;
    Require(Graphics::HostImportLiveBytes() == 0, "bounded direct-import test must start with no imported bytes");
    RestoreImportPolicy restore;

    FileSection backing(Bytes);
    {
        DirectMapping primary(backing.resource.section, 0);
        DirectMapping alias(backing.resource.section, 0);
        RegisteredPrimary registered(primary.address);
        ClearHostImportsOnExit clearImportsOnExit(device->Device());
        auto* first = static_cast<std::uint8_t*>(primary.address);
        auto* second = static_cast<std::uint8_t*>(alias.address);
        const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(first));
        const auto prefixOffset = Window - ProbeBytes;
        const auto tailOffset = Window;
        constexpr std::uint32_t GpuPrefix = 0x13579bdfu;
        constexpr std::uint32_t GpuTail = 0x2468ace0u;
        constexpr std::uint32_t CpuPrefix = 0x55aa33ccu;
        constexpr std::uint32_t CpuTail = 0xa55ac33cu;
        HostFill(first + prefixOffset, 0x11111111u);
        HostFill(first + tailOffset, 0x22222222u);

        AgcDriver::Graphics::Recorder recorder(context);
        recorder.Activate();
        {
            AgcDriver::Graphics::GuestBufferMemory memory(context);
            memory.AcquireRegistered();
            memory.AddWritable(address + prefixOffset, ProbeBytes);
            memory.AddWritable(address + tailOffset, ProbeBytes);
            {
                std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
                memory.Upload(true);
            }
            Require(Graphics::HostImportLiveBytes() == Window, "the 64 MiB cap did not import exactly the first window");
            Require(Graphics::HostImportCovers(context, address + prefixOffset, ProbeBytes), "the imported prefix does not reach the first boundary probe");
            Require(!Graphics::HostImportCovers(context, address + tailOffset, ProbeBytes), "the tail unexpectedly imported past the cap");
            std::uint32_t prefixAdjustment = 0;
            std::uint32_t tailAdjustment = 0;
            const auto prefix = memory.Descriptor(address + prefixOffset, ProbeBytes, prefixAdjustment);
            const auto tail = memory.Descriptor(address + tailOffset, ProbeBytes, tailAdjustment);
            Require(prefix.buffer != VK_NULL_HANDLE && tail.buffer != VK_NULL_HANDLE, "a boundary descriptor has no Vulkan buffer");
            Require(prefixAdjustment == 0 && tailAdjustment == 0, "the aligned boundary descriptors unexpectedly need address adjustment");
            FillGpu(*device, recorder, prefix, tail, GpuPrefix, GpuTail);
            memory.WriteBack();
        }
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + prefixOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(second + prefixOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + tailOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(second + tailOffset, ProbeBytes);
        RequireWords(first + prefixOffset, GpuPrefix, "GPU writes to the imported prefix did not reach primary guest memory");
        RequireWords(second + prefixOffset, GpuPrefix, "GPU writes to the imported prefix did not reach the shared file alias");
        RequireWords(first + tailOffset, GpuTail, "GPU writes to the mirrored tail were not copied back to primary guest memory");
        RequireWords(second + tailOffset, GpuTail, "GPU writes to the mirrored tail were not copied back to the shared file alias");

        HostFill(second + prefixOffset, CpuPrefix);
        HostFill(second + tailOffset, CpuTail);
        {
            AgcDriver::Graphics::GuestBufferMemory refreshed(context);
            refreshed.AcquireRegistered();
            refreshed.AddReadable(address + prefixOffset, ProbeBytes);
            refreshed.AddReadable(address + tailOffset, ProbeBytes);
            {
                std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
                refreshed.Upload(true);
            }
            Require(Graphics::HostImportLiveBytes() == Window, "refresh exceeded or dropped the bounded imported prefix");
            Require(Graphics::HostImportCovers(context, address + prefixOffset, ProbeBytes), "refresh lost the imported prefix");
            Require(!Graphics::HostImportCovers(context, address + tailOffset, ProbeBytes), "refresh imported tail bytes beyond the cap");
            std::uint32_t prefixAdjustment = 0;
            std::uint32_t tailAdjustment = 0;
            const auto prefix = refreshed.Descriptor(address + prefixOffset, ProbeBytes, prefixAdjustment);
            const auto tail = refreshed.Descriptor(address + tailOffset, ProbeBytes, tailAdjustment);
            Require(prefixAdjustment == 0 && tailAdjustment == 0, "the refreshed boundary descriptors unexpectedly need address adjustment");
            RequireReadback(*device, recorder, prefix, tail, CpuPrefix, CpuTail);
        }
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + prefixOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + tailOffset, ProbeBytes);
        RequireWords(first + prefixOffset, CpuPrefix, "HostWrite to the alias did not update the primary imported bytes");
        RequireWords(first + tailOffset, CpuTail, "HostWrite to the alias did not update the primary tail bytes");
        recorder.Sync();
        device->WaitIdle();
        Require(Graphics::HostImportLiveBytes() == Window, "the bounded prefix import disappeared before tail mutation");
        {
            GuestAllocations::Mutation mutation;
            bool unmapped = false;
            mutation.Unmap(first + Window, Window, [&](const void* piece, std::size_t bytes, const void* allocation, bool last) {
                Require(piece == first + Window && bytes == Window && allocation == first && !last,
                        "tail-only unmap did not preserve the original allocation's prefix");
                GuestArena::GuestArenaReset_nid_postfix(const_cast<void*>(piece), bytes);
                unmapped = true;
            });
            Require(unmapped, "the tail-only unmap callback was not reached");
            Require(mutation.Covers(first, Window) && !mutation.Covers(first + Window, Window),
                    "tail-only unmap did not leave exactly the prefix registered");
        }
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + prefixOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(second + prefixOffset, ProbeBytes);
        RequireWords(first + prefixOffset, CpuPrefix, "tail unmap damaged bytes in the retained prefix");
        RequireWords(second + prefixOffset, CpuPrefix, "tail unmap damaged the shared alias of the retained prefix");
        constexpr std::uint32_t AfterTailUnmap = 0x6ac31f84u;
        HostFill(second + prefixOffset, AfterTailUnmap);
        {
            AgcDriver::Graphics::GuestBufferMemory rebuilt(context);
            rebuilt.AcquireRegistered();
            rebuilt.AddWritable(address + prefixOffset, ProbeBytes);
            {
                std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
                rebuilt.Upload(true);
            }
            Require(Graphics::HostImportLiveBytes() == Window, "rebuilding after tail unmap did not retain exactly the 64 MiB prefix import");
            Require(Graphics::HostImportCovers(context, address + prefixOffset, ProbeBytes), "the retained prefix was not importable after tail unmap");
            std::uint32_t adjustment = 0;
            const auto prefix = rebuilt.Descriptor(address + prefixOffset, ProbeBytes, adjustment);
            Require(prefix.buffer != VK_NULL_HANDLE && adjustment == 0, "the retained prefix descriptor is invalid after tail unmap");
            FillGpu(*device, recorder, prefix, prefix, GpuPrefix, GpuPrefix);
            rebuilt.WriteBack();
        }
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(first + prefixOffset, ProbeBytes);
        GuestArena::GuestArenaPrepareHostRead_nid_no_patch(second + prefixOffset, ProbeBytes);
        RequireWords(first + prefixOffset, GpuPrefix, "GPU write to the retained prefix failed after tail unmap");
        RequireWords(second + prefixOffset, GpuPrefix, "retained prefix GPU write did not remain coherent with its shared alias");
        recorder.Sync();
        device->WaitIdle();
    }
    GuestArena::GuestArenaForgetSection_nid_postfix(backing.resource.section);
    Require(Graphics::HostImportLiveBytes() == 0, "shared direct imports remained live after the test allocation was released");
    return true;
}

}

int main() {
    try {
        if (!CheckBoundedSharedDirectImport()) return VulkanTestSkipped;
        std::puts("bounded shared direct host import and copied-tail boundary passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

#else

int main() { return VulkanTestSkipped; }

#endif
