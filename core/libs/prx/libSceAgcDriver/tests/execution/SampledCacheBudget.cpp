#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t SurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t Surfaces = 12;
constexpr std::size_t BlockBytes = SurfaceBytes * Surfaces;
constexpr std::size_t RegistrationGapBytes = 65536;
constexpr std::size_t AllocationBytes = BlockBytes + RegistrationGapBytes;
constexpr std::uint64_t Budget = 1ull << 20u;
constexpr std::uint64_t Ceiling = Budget + Budget / 4u;

std::optional<std::size_t> selectDevice(std::span<const VkPhysicalDeviceProperties> devices, std::string_view request) {
    const auto rankDeviceType = [](VkPhysicalDeviceType type) {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
            default: return 0;
        }
    };
    std::optional<std::size_t> selected;
    int selectedRank = -1;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        const auto& candidate = devices[index];
        if (candidate.apiVersion < VK_API_VERSION_1_1) continue;
        const std::string_view name(candidate.deviceName);
        if (!request.empty() && std::search(name.begin(), name.end(), request.begin(), request.end(), [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) == std::tolower(static_cast<unsigned char>(right));
        }) == name.end()) continue;
        const int rank = request.empty() ? rankDeviceType(candidate.deviceType) : 4;
        if (rank <= selectedRank) continue;
        selected = index;
        selectedRank = rank;
    }
    return selected;
}

void deviceSelectionTests() {
    const auto properties = [](const char* name, VkPhysicalDeviceType type) {
        VkPhysicalDeviceProperties result{};
        result.apiVersion = VK_API_VERSION_1_1;
        result.deviceType = type;
        std::memcpy(result.deviceName, name, std::strlen(name) + 1u);
        return result;
    };
    std::array devices{
        properties("AMD Radeon Graphics", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        properties("NVIDIA GeForce RTX 5080", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
        properties("Virtual GPU", VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU),
        properties("CPU", VK_PHYSICAL_DEVICE_TYPE_CPU),
    };
    Require(!selectDevice({}, {}).has_value(), "an empty device list selected a device");
    Require(selectDevice(devices, {}) == 1u, "default selection did not prefer the discrete GPU");
    Require(selectDevice(devices, "AMD") == 0u, "the AMD request did not select the integrated GPU");
    Require(selectDevice(devices, "NVIDIA") == 1u, "the NVIDIA request did not select the discrete GPU");
    Require(selectDevice(devices, "aMd rAdEoN") == 0u, "a requested integrated GPU lost to an unrelated discrete GPU");
    Require(selectDevice(devices, "rtx 5080") == 1u, "a partial GPU name did not match without regard to case");
    Require(selectDevice(devices, "Virtual") == 2u, "a requested virtual GPU was not selected");
    Require(selectDevice(devices, "cpu") == 3u, "a requested CPU device was not selected");
    Require(!selectDevice(devices, "missing").has_value(), "an unmatched request selected another GPU");
    Require(!selectDevice(devices, "NVIDIA GeForce RTX 5080 with a longer name").has_value(), "a request longer than the device name matched");
    Require(selectDevice(std::span(devices).subspan(2), {}) == 0u, "default selection did not prefer the virtual GPU to the CPU");
    devices[1].apiVersion = VK_API_VERSION_1_0;
    Require(selectDevice(devices, {}) == 0u, "default selection did not prefer the integrated GPU to the virtual GPU");
    Require(!selectDevice(devices, "NVIDIA").has_value(), "a requested device without Vulkan 1.1 was selected");
    devices[1] = devices[0];
    Require(selectDevice(devices, {}) == 0u && selectDevice(devices, "AMD") == 0u, "equally ranked matches did not retain enumeration order");
    for (auto& device : devices) device.apiVersion = VK_API_VERSION_1_0;
    Require(!selectDevice(devices, {}).has_value(), "a list without Vulkan 1.1 selected a device");
    std::puts("sampled-cache device selection CPU tests passed");
}

class Device {
public:
    explicit Device(bool enableStorageImageMultisample = true) {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Require(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Require(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            const auto physicalProperties = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
            std::vector<VkPhysicalDeviceProperties> deviceProperties(devices.size());
            for (std::size_t index = 0; index < devices.size(); ++index) physicalProperties(devices[index], &deviceProperties[index]);
            const char* requestedName = std::getenv("ANYPS5_GPU");
            const auto selectedIndex = selectDevice(deviceProperties, requestedName != nullptr ? requestedName : "");
            Require(selectedIndex.has_value(), requestedName != nullptr && *requestedName != '\0' ? std::string("no Vulkan 1.1 device matching ANYPS5_GPU=\"") + requestedName + "\"" : "no Vulkan 1.1 device");
            context.physical = devices[*selectedIndex];
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            const auto hasExtension = [&](const char* name) {
                for (const auto& extension : available) {
                    if (std::strcmp(extension.extensionName, name) == 0) return true;
                }
                return false;
            };
            auto bytes = AgcDriver::QueryBdaByteFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            auto address = AgcDriver::QueryBdaFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            std::uint32_t family = 0;
            while (family < count && (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) ++family;
            Require(family < count, "no Vulkan compute queue");
            const float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkPhysicalDeviceFeatures supported{};
            function<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")(context.physical, &supported);
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            enabled.textureCompressionBC = supported.textureCompressionBC;
            context.textureCompressionBC = enabled.textureCompressionBC == VK_TRUE;
            enabled.shaderStorageImageMultisample = enableStorageImageMultisample && supported.shaderStorageImageMultisample;
            context.storageImageMultisample = enabled.shaderStorageImageMultisample == VK_TRUE;
            address.pNext = &bytes;
            std::vector<const char*> extensionsEnabled{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME};
            if (hasExtension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                extensionsEnabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hostProperties};
                function<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(context.physical, &properties);
                context.hostImportAlignment = hostProperties.minImportedHostPointerAlignment;
            }
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &address};
            device.queueCreateInfoCount = 1;
            device.pQueueCreateInfos = &queue;
            device.enabledExtensionCount = static_cast<std::uint32_t>(extensionsEnabled.size());
            device.ppEnabledExtensionNames = extensionsEnabled.data();
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            VkPhysicalDeviceProperties properties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
            std::printf("sampled-cache fixture Vulkan device: %s vendor=0x%04x device=0x%04x\n", properties.deviceName, properties.vendorID, properties.deviceID);
            context.limits = properties.limits;
            context.bufferDeviceAddress = true;
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            if (hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) context.memoryProperties2 = function<PFN_vkGetPhysicalDeviceMemoryProperties2>("vkGetPhysicalDeviceMemoryProperties2");
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }
    const Context& GetContext() const { return context; }

    void VerifySelection() const {
        VkPhysicalDeviceProperties properties{};
        function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
        const char* request = std::getenv("ANYPS5_GPU");
        Require(selectDevice(std::span(&properties, 1), request != nullptr ? request : "").has_value(), "the sampled-cache fixture physical device does not match ANYPS5_GPU");
        std::puts("sampled-cache fixture physical device selection verified");
    }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Require(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
        context.bufferPool.reset();
        if (context.device != VK_NULL_HANDLE) function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
};

class Block {
public:
    explicit Block(const Context& context) : context(context) {
#ifdef _WIN32
        memory = VirtualAlloc(nullptr, AllocationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        memory = std::aligned_alloc(65536, AllocationBytes);
#endif
        Require(memory != nullptr, "cannot allocate the texture block");
        std::memset(memory, 0x5a, BlockBytes);
        GuestAllocations::Mutation().Add(memory, BlockBytes, true, true, true);
    }

    ~Block() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        HostImportFor(context, Surface(0), BlockBytes);
#ifdef _WIN32
        VirtualFree(memory, 0, MEM_RELEASE);
#else
        std::free(memory);
#endif
    }

    std::uint64_t Surface(std::size_t surface) const { return reinterpret_cast<std::uint64_t>(memory) + surface * SurfaceBytes; }

private:
    const Context& context;
    void* memory = nullptr;
};

class WatchedSurface {
public:
    explicit WatchedSurface(const Context& context, std::size_t bytes = SurfaceBytes) : context(context), span(bytes) {
#ifdef _WIN32
        memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536));
        if (memory != nullptr) GuestArena::GuestArenaCommit_nid_postfix(memory, bytes, PAGE_READWRITE, bytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
            memory = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(memory, bytes);
        }
#endif
        Require(memory != nullptr, "cannot allocate the watched surface");
        Require(AgcDriver::GuestMemory::Watched(Address(), bytes), "the watched surface is not write-watched");
        std::memset(memory, 0x5a, bytes);
        GuestAllocations::Mutation().Add(memory, bytes, true, true, true);
        AgcDriver::GuestMemory::CollectWritesUncached(Address(), bytes);
    }

    ~WatchedSurface() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(memory, span);
        GuestArena::GuestArenaRelease_nid_postfix(memory, span);
#else
        munmap(memory, span);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, span);
#endif
    }

    WatchedSurface(const WatchedSurface&) = delete;
    WatchedSurface& operator=(const WatchedSurface&) = delete;

    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(memory); }

private:
    const Context& context;
    std::size_t span = 0;
    std::uint8_t* memory = nullptr;
};

std::array<std::uint32_t, 8> Descriptor(std::uint64_t address, std::uint32_t side = Side, std::uint32_t format = Format32UInt) {
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | (((side - 1u) & 3u) << 30u),
        ((side - 1u) >> 2u) | ((side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::shared_ptr<Texture> Sampled(const Context& context, std::uint64_t address, std::uint32_t side = Side, std::uint32_t format = Format32UInt) {
    const auto words = Descriptor(address, side, format);
    return CachedSampledTexture(context, words);
}

GuestTextureResource Resource(std::uint64_t address, std::uint32_t side = Side, std::uint32_t format = Format32UInt) {
    const auto words = Descriptor(address, side, format);
    return DecodeTextureResource(words);
}

std::shared_ptr<ShaderResources> ImageResources(const Context& context, const Block& block, std::size_t first, std::uint32_t count, bool sampled) {
    ShaderRecompiler::DescriptorBinding binding{};
    binding.kind = sampled ? ShaderRecompiler::DescriptorKind::SampledImage : ShaderRecompiler::DescriptorKind::StorageImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = (sampled ? ShaderRecompiler::RuntimeAbi::FirstImageBinding : ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding) + PlaceholderDimensionsPerClass + 2u;
    binding.count = count;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageWritten.assign(count, false);
    binding.imageSamplers.assign(count, 0u);
    for (std::uint32_t element = 0; element < count; ++element) {
        const auto words = Descriptor(block.Surface(first + element));
        binding.guestDescriptor.insert(binding.guestDescriptor.end(), words.begin(), words.end());
    }
    ShaderRecompiler::RecompileResult program;
    program.bindings.push_back(std::move(binding));
    const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    return std::make_shared<ShaderResources>(context, shader);
}

void recorderEvictionTests(const Context& context, const Block& block) {
    auto& cache = SharedResourceCache();
    cache.Clear();
    ClearCachedTextures(context.device);
    auto sampled = ImageResources(context, block, 0, 1, true);
    const auto images = sampled->StorageImages();
    Require(images.size() == 1, "the eviction fixture has no sampled storage image");
    const std::weak_ptr<ShaderResources> resources = sampled;
    const std::weak_ptr<Texture> view = sampled->SampledTextures().front();
    const std::weak_ptr<StorageTexture> image = sampled->SampledTextures().front()->SharedStorageSource();
    cache.Insert({1}, sampled);
    Require(cache.Find({1}) == sampled, "the recorder fixture's descriptor set was not cached");
    sampled.reset();
    Buffer readback(context, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    {
        Recorder recorder(context);
        recorder.Activate();
        recorder.Commands();
        for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
        Require(cache.Find({1}) == nullptr, "storage eviction left the recorded descriptor set cached");
        Require(!resources.expired() && !view.expired() && !image.expired(), "eviction destroyed the recorder's descriptor set, view or image");
        Require(!image.lock()->Cached(), "the recorder fixture did not evict the source image");
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {1, 1, 1};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, images.front().first, VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Sync();
        std::uint32_t pixel = 0;
        std::memcpy(&pixel, readback.Bytes().data(), sizeof(pixel));
        Require(pixel == 0x5a5a5a5a, "eviction changed the recorder's storage image pixel");
    }
    Require(resources.expired() && view.expired() && image.expired(), "an evicted image outlived its completed recorder");
    std::puts("storage eviction recorder lifetime and pixel readback passed");
}

#ifdef _WIN32
void CheckSnapshotWords(std::span<const std::byte> bytes, std::uint32_t expected, const char* label) {
    Require(bytes.size() % 4 == 0, "word oracle has nonword extent");
    for (std::size_t offset = 0; offset < bytes.size(); offset += 4) {
        std::uint32_t actual;
        std::memcpy(&actual, bytes.data() + offset, 4);
        if (actual != expected) {
            char detail[180];
            std::snprintf(detail, sizeof(detail), "%s word=%zu expected=0x%08x actual=0x%08x", label, offset / 4, expected, actual);
            throw std::runtime_error(detail);
        }
    }
}

class PendingSnapshotBlock {
public:
    static constexpr std::size_t GuardBytes = 65536;
    static constexpr std::uint32_t Initial = 0x15263748u, Guard = 0x6d29c853u;
    PendingSnapshotBlock(const Context& context, std::size_t surfaceBytes) : context(context), surfaceBytes(surfaceBytes), bytes(2 * surfaceBytes + 3 * GuardBytes) {
        base = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, GuardBytes));
        Require(base != nullptr, "pending snapshot: guest allocation failed");
        GuestArena::GuestArenaCommit_nid_postfix(base, bytes, PAGE_READWRITE, GuardBytes);
        GuestAllocations::Mutation().Add(base, bytes - 256, true, true, true);
        std::fill_n(reinterpret_cast<std::uint32_t*>(base), bytes / 4, Guard);
        for (const auto address : {Sample(), Disjoint()})
            std::fill_n(reinterpret_cast<std::uint32_t*>(address), surfaceBytes / 4, Initial);
    }
    ~PendingSnapshotBlock() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(base);
        GuestArena::GuestArenaReset_nid_postfix(base, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(base, bytes);
    }
    std::uint64_t Sample() const { return reinterpret_cast<std::uint64_t>(base) + GuardBytes; }
    std::uint64_t Disjoint() const { return Sample() + surfaceBytes + GuardBytes; }
    void Check(std::uint32_t sample, std::uint32_t disjoint) const {
        CheckSnapshotWords({reinterpret_cast<const std::byte*>(Sample()), surfaceBytes}, sample, "sampled guest backing");
        CheckSnapshotWords({reinterpret_cast<const std::byte*>(Disjoint()), surfaceBytes}, disjoint, "disjoint guest backing");
        for (const auto offset : {std::size_t{0}, GuardBytes + surfaceBytes, 2 * GuardBytes + 2 * surfaceBytes})
            CheckSnapshotWords({base + offset, GuardBytes}, Guard, "pending snapshot guard");
    }
private:
    Context context;
    std::size_t surfaceBytes;
    std::size_t bytes;
    std::byte* base = nullptr;
};

void pendingSnapshotTests(const Context& context, bool disjoint, std::uint32_t side) {
    namespace GM = AgcDriver::GuestMemory;
    for (const char* name : {"APS5_NO_TEXTURE_CACHE", "APS5_GPU_NO_WRITEBACK"})
        Require(std::getenv(name) == nullptr, std::string("pending snapshot fixture requires unset ") + name);
    const auto surfaceBytes = std::size_t{side} * side * 4u;
    PendingSnapshotBlock memory(context, surfaceBytes);
    Recorder recorder(context);
    recorder.Activate();
    const auto address = memory.Sample();
    const auto words = Descriptor(address, side);
    Require(DescribeSurface(DecodeTextureResource(words)).guestBytes == surfaceBytes, "pending snapshot: unexpected descriptor extent");
    GM::BumpCollectEpoch();
    Require(GM::WriteWatched() && GM::CollectWrites(address, surfaceBytes) != 0, "pending snapshot: full watched proof unavailable");
    Require(GM::Watched(address, surfaceBytes), "pending snapshot: snapshot surface is not fully watched");
    const auto topology = GM::CommittedRanges(address, surfaceBytes);
    Require(topology.size() == 1 && topology[0] == std::pair{address, address + surfaceBytes}, "pending snapshot: snapshot surface is not fully committed");
    auto previous = CachedSampledTexture(context, words);
    Require(previous != nullptr && !previous->ViewsStorageImage(), "pending snapshot: lookup did not create an owned snapshot");
    recorder.Sync();
    Require(CachedSampledTexture(context, words) == previous, "pending snapshot: initial unchanged lookup did not reuse");
    Require(GM::Watched(address, surfaceBytes), "pending snapshot: snapshot setup removed write watching");
    memory.Check(PendingSnapshotBlock::Initial, PendingSnapshotBlock::Initial);
    std::uint32_t sampleExpected = PendingSnapshotBlock::Initial, disjointExpected = PendingSnapshotBlock::Initial;
    for (const bool submitted : {false, true}) {
        const auto destination = disjoint ? memory.Disjoint() : address;
        const std::uint32_t value = submitted ? 0x392b17a5u : 0x725c183eu;
        const auto copied = std::make_shared<Buffer>(context, surfaceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::fill(copied->Bytes().begin(), copied->Bytes().end(), std::byte{0xa6});
        const auto landed = std::make_shared<bool>(false);
        const auto epoch = GM::ThreadCollectEpoch();
        const auto ownership = std::array<std::uint64_t, 5>{epoch, GM::UnwatchSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), DepthSurfacesSerial(), GM::ForgetSerial()};
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, copied->Handle(), 0, surfaceBytes, value);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.OnComplete([copied, landed, destination, value] {
            copied->Invalidate();
            CheckSnapshotWords(copied->Bytes(), value, "GPU copied-buffer output");
            GM::StoreOwnBytes(destination, copied->Bytes());
            *landed = true;
        });
        recorder.Keep(copied);
        recorder.NotePendingWrite(destination, surfaceBytes);
        if (submitted) recorder.Submit();
        Require(!*landed && Recorder::SnapshotWriteOverlaps(destination, surfaceBytes), "pending snapshot: copied writer was not pending");
        Require(Recorder::SnapshotWriteOverlaps(address, surfaceBytes) != disjoint, "pending snapshot: overlap geometry differs");
        const auto next = CachedSampledTexture(context, words);
        Require(std::array<std::uint64_t, 5>{GM::ThreadCollectEpoch(), GM::UnwatchSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), DepthSurfacesSerial(), GM::ForgetSerial()} == ownership,
                "pending snapshot: lookup changed ownership proof state");
        if (disjoint) {
            Require(next == previous && !*landed && Recorder::SnapshotWriteOverlaps(destination, surfaceBytes),
                    "pending snapshot: disjoint lookup waited or replaced its unchanged snapshot");
            memory.Check(sampleExpected, disjointExpected);
            recorder.Sync();
            disjointExpected = value;
        } else {
            Require(*landed && !Recorder::SnapshotWriteOverlaps(destination, surfaceBytes),
                    "pending snapshot: overlapping lookup returned before copied write publication");
            Require(next != previous && !next->ViewsStorageImage(), "pending snapshot: overlapping lookup reused stale snapshot");
            sampleExpected = value;
            recorder.Sync();
        }
        Require(*landed && !Recorder::SnapshotWriteOverlaps(destination, surfaceBytes), "pending snapshot: explicit completion did not publish");
        memory.Check(sampleExpected, disjointExpected);
        const auto completed = CachedSampledTexture(context, words);
        if (!disjoint && surfaceBytes > (256u << 10u)) {
            Require(completed != nullptr && !completed->ViewsStorageImage(), "pending snapshot: completed large snapshot was not owned");
        } else {
            Require(completed == next, "pending snapshot: completed snapshot was not reused");
        }
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            Require(CachedSampledTexture(context, words) == completed, "pending snapshot: completed snapshot did not stabilize after one refresh");
            Require(!Recorder::SnapshotWriteOverlaps(destination, surfaceBytes), "pending snapshot: stable lookup left a pending writer");
            CheckSnapshotWords(copied->Bytes(), value, "stable GPU copied-buffer output");
            memory.Check(sampleExpected, disjointExpected);
        }
        previous = completed;
        std::printf("snapshot_pending_write_case=PASS disjoint=%u submitted=%u surface_bytes=%zu full_gpu_and_guest_words=%zu guards=%zu\n",
                    static_cast<unsigned>(disjoint), static_cast<unsigned>(submitted), surfaceBytes, surfaceBytes / 4, 3 * PendingSnapshotBlock::GuardBytes / 4);
    }
    recorder.Sync();
    const std::weak_ptr<Texture> retired = previous;
    previous.reset();
    ClearCachedTextures(context.device);
    Require(retired.expired(), "pending snapshot: retired snapshot remained alive");
    const auto remade = CachedSampledTexture(context, words);
    Require(remade != nullptr && !remade->ViewsStorageImage(), "pending snapshot: retired snapshot was not rebuilt");
    recorder.Sync();
    memory.Check(sampleExpected, disjointExpected);
    Require(CachedSampledTexture(context, words) == remade, "pending snapshot: rebuilt snapshot was not reused");
    std::printf("snapshot_pending_retirement=PASS disjoint=%u surface_bytes=%zu\n", static_cast<unsigned>(disjoint), surfaceBytes);
}
#endif

#ifdef _WIN32
std::vector<std::uint32_t> TexelFetchShader(std::uint32_t side, std::optional<std::uint32_t> fixedIndex = std::nullopt, bool floatSample = false) {
    ShaderRecompiler::SpirvModule module;
    module.EmitCapability(spv::CapabilityShader);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto i32 = module.Type(spv::OpTypeInt, 32u, 1u);
    const auto v4 = module.Type(spv::OpTypeVector, u32, 4u);
    const auto v3 = module.Type(spv::OpTypeVector, u32, 3u);
    const auto v2 = module.Type(spv::OpTypeVector, i32, 2u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto constant = [&](std::uint32_t value) { return module.Constant(spv::OpConstant, u32, value); };
    const auto array = module.Type(spv::OpTypeRuntimeArray, v4);
    module.AddAnnotation(spv::OpDecorate, array, spv::DecorationArrayStride, 16u);
    const auto block = module.Type(spv::OpTypeStruct, array);
    module.AddAnnotation(spv::OpDecorate, block, spv::DecorationBlock);
    module.AddAnnotation(spv::OpMemberDecorate, block, 0u, spv::DecorationOffset, 0u);
    const auto buffer = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, block), spv::StorageClassStorageBuffer);
    module.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationDescriptorSet, 0u);
    module.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationBinding, 1u);
    const auto sampledType = floatSample ? module.Type(spv::OpTypeFloat, 32u) : u32;
    const auto sampledVector = module.Type(spv::OpTypeVector, sampledType, 4u);
    const auto imageType = module.Type(spv::OpTypeImage, sampledType, spv::Dim2D, 0u, 0u, 0u, 1u, spv::ImageFormatUnknown);
    const auto image = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType), spv::StorageClassUniformConstant);
    module.AddAnnotation(spv::OpDecorate, image, spv::DecorationDescriptorSet, 0u);
    module.AddAnnotation(spv::OpDecorate, image, spv::DecorationBinding, 0u);
    const auto group = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassInput, v3), spv::StorageClassInput);
    module.AddAnnotation(spv::OpDecorate, group, spv::DecorationBuiltIn, spv::BuiltInWorkgroupId);
    const auto main = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
    module.AddFunction(spv::OpLabel, module.AllocateId());
    const auto loaded = module.AllocateId();
    module.AddFunction(spv::OpLoad, v3, loaded, group);
    const auto groupIndex = module.AllocateId();
    module.AddFunction(spv::OpCompositeExtract, u32, groupIndex, loaded, 0u);
    const auto binary = [&](spv::Op op, std::uint32_t a, std::uint32_t b) {
        const auto result = module.AllocateId();
        module.AddFunction(op, u32, result, a, b);
        return result;
    };
    const auto index = fixedIndex.has_value() ? binary(spv::OpIAdd, groupIndex, constant(*fixedIndex)) : groupIndex;
    const auto signedValue = [&](std::uint32_t value) {
        const auto result = module.AllocateId();
        module.AddFunction(spv::OpBitcast, i32, result, value);
        return result;
    };
    const auto coordinate = module.AllocateId();
    module.AddFunction(spv::OpCompositeConstruct, v2, coordinate, signedValue(binary(spv::OpUMod, index, constant(side))), signedValue(binary(spv::OpUDiv, index, constant(side))));
    const auto object = module.AllocateId();
    module.AddFunction(spv::OpLoad, imageType, object, image);
    auto value = module.AllocateId();
    module.AddFunction(spv::OpImageFetch, sampledVector, value, object, coordinate, spv::ImageOperandsLodMask, constant(0u));
    if (floatSample) {
        const auto bits = module.AllocateId();
        module.AddFunction(spv::OpBitcast, v4, bits, value);
        value = bits;
    }
    const auto pointer = module.AllocateId();
    module.AddFunction(spv::OpAccessChain, module.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, v4), pointer, buffer, constant(0u), index);
    module.AddFunction(spv::OpStore, pointer, value);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {group});
    module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 1u, 1u, 1u);
    return module.Finalize();
}

class TexelFetch {
public:
    TexelFetch(const Context& context, const Texture& texture, Buffer& output, std::span<const std::uint32_t> code) : context(context) {
        try {
            const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{{0u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}}};
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = 2u;
            info.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &setLayout), "texel fetch descriptor layout");
            const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u}}};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1u;
            poolInfo.poolSizeCount = 2u;
            poolInfo.pPoolSizes = sizes.data();
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "texel fetch descriptor pool");
            const VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1u, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocate, &set), "texel fetch descriptor set");
            const VkDescriptorImageInfo image{VK_NULL_HANDLE, texture.View(), texture.Layout()};
            const VkDescriptorBufferInfo buffer{output.Handle(), 0u, output.Bytes().size()};
            const std::array<VkWriteDescriptorSet, 2> writes{{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0u, 0u, 1u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &image, nullptr, nullptr}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1u, 0u, 1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer, nullptr}}};
            context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 2u, writes.data(), 0u, nullptr);
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1u;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "texel fetch pipeline layout");
            VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shaderInfo.codeSize = code.size_bytes();
            shaderInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &shaderInfo, nullptr, &module), "texel fetch shader module");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.layout = layout;
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1u, &pipelineInfo, nullptr, &pipeline), "texel fetch pipeline");
        } catch (...) {
            release();
            throw;
        }
    }
    ~TexelFetch() { release(); }
    TexelFetch(const TexelFetch&) = delete;
    TexelFetch& operator=(const TexelFetch&) = delete;

    void Run(VkCommandBuffer commands, std::uint32_t groups) const {
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0u, 1u, &set, 0u, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1u, 1u);
    }

private:
    void release() noexcept {
        if (pipeline != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module != VK_NULL_HANDLE) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    Context context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

std::uint32_t FetchFirstTexel(const Context& context, const Texture& texture, std::uint32_t side) {
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Sync();
    Buffer output(context, 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
    const TexelFetch fetch(context, texture, output, TexelFetchShader(side));
    CommandBatch batch(context);
    fetch.Run(batch.Handle(), 1u);
    RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    batch.SubmitAndWait();
    output.Invalidate();
    std::uint32_t texel = 0;
    std::memcpy(&texel, output.Bytes().data(), sizeof(texel));
    return texel;
}

std::uint32_t FetchTexelAt(const Context& context, const Texture& texture, std::uint32_t side, std::uint32_t index) {
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->Sync();
    const auto bytes = (static_cast<std::size_t>(index) + 1u) * 16u;
    Buffer output(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
    const TexelFetch fetch(context, texture, output, TexelFetchShader(side, index));
    CommandBatch batch(context);
    fetch.Run(batch.Handle(), 1u);
    RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    batch.SubmitAndWait();
    output.Invalidate();
    std::uint32_t texel = 0;
    std::memcpy(&texel, output.Bytes().data() + static_cast<std::size_t>(index) * 16u, sizeof(texel));
    return texel;
}

class SnapshotUploadMemory {
public:
    explicit SnapshotUploadMemory(const Context& context, std::size_t bytes) : context(context), bytes(bytes) {
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
        Require(section != nullptr, "cannot create the direct snapshot shared section");
        memory = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536);
        if (memory == nullptr) {
            CloseHandle(section);
            throw std::runtime_error("cannot allocate the direct snapshot range");
        }
        try {
            GuestArena::GuestArenaMapDirect_nid_postfix(memory, bytes, section, 0, PAGE_READWRITE);
            GuestAllocations::Mutation().Add(memory, bytes, true, true, true);
            std::memset(memory, 0x5a, bytes);
        } catch (...) {
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
            CloseHandle(section);
            throw;
        }
    }
    ~SnapshotUploadMemory() {
        ClearCachedTextures(context.device);
        GuestAllocations::Mutation().Remove(memory);
        GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
        CloseHandle(section);
    }
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(memory); }
private:
    const Context& context;
    std::size_t bytes;
    HANDLE section = nullptr;
    void* memory = nullptr;
};

class UploadBufferCounter {
public:
    explicit UploadBufferCounter(Context& context) : context(context) {
        Require(!active, "nested upload buffer counters");
        active = true;
        previous = context.deviceProc;
        create = reinterpret_cast<PFN_vkCreateBuffer>(previous(context.device, "vkCreateBuffer"));
        count = 0;
        context.deviceProc = Resolve;
    }
    ~UploadBufferCounter() {
        context.deviceProc = previous;
        active = false;
    }
    std::uint64_t Count() const { return count; }
private:
    static PFN_vkVoidFunction VKAPI_CALL Resolve(VkDevice device, const char* name) {
        if (std::strcmp(name, "vkCreateBuffer") == 0) return reinterpret_cast<PFN_vkVoidFunction>(Create);
        return previous(device, name);
    }
    static VkResult VKAPI_CALL Create(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
        ++count;
        return create(device, info, allocator, buffer);
    }
    Context& context;
    static inline thread_local bool active = false;
    static inline thread_local PFN_vkGetDeviceProcAddr previous = nullptr;
    static inline thread_local PFN_vkCreateBuffer create = nullptr;
    static inline thread_local std::uint64_t count = 0;
};

void packedSnapshotTests(Context& context, bool packed, bool compressed) {
    namespace GM = AgcDriver::GuestMemory;
    constexpr std::uint32_t Side = 128u;
    constexpr std::size_t Bytes = std::size_t{Side} * Side * 4u;
    constexpr std::size_t Count = 17;
    SnapshotUploadMemory memory(context, (Count + 1u) * Bytes);
    const auto format = compressed ? 169u : Format32UInt;
    const auto value = [&](std::size_t index) { return compressed ? (index % 2u == 0 ? 0x3f800000u : 0u) : 0x17890000u + static_cast<std::uint32_t>(index); };
    for (std::size_t index = 0; index < Count; ++index) {
        const auto address = memory.Address() + index * Bytes;
        if (compressed) std::fill_n(reinterpret_cast<std::uint64_t*>(address), Bytes / 8u, index % 2u == 0 ? 0xf800ull : 0x07e0ull);
        else std::fill_n(reinterpret_cast<std::uint32_t*>(address), Bytes / 4u, value(index));
        GM::CollectWritesUncached(address, Bytes);
        Require(WatchedAfterSettle(context, address, Bytes), "packed snapshot fixture lost write tracking");
        GM::CollectWritesUncached(address, Bytes);
    }
    UploadBufferCounter buffers(context);
    Recorder recorder(context);
    recorder.Activate();
    auto draw = recorder.DrawUpload(16);
    const std::array<std::uint64_t, 3> drawStamp{1, 1, 1};
    recorder.NoteUnwrittenDrawSnapshot(memory.Address(), 16, drawStamp, draw.buffer, draw.offset);
    const auto before = buffers.Count();
    std::vector<std::shared_ptr<Texture>> textures;
    VkDeviceSize imageBytes = 0;
    for (std::size_t index = 0; index < Count; ++index) {
        textures.push_back(Sampled(context, memory.Address() + index * Bytes, Side, format));
        imageBytes += textures.back()->AllocationBytes();
    }
    const auto created = buffers.Count() - before;
    const auto retained = recorder.OpenKeptBytes();
    const auto expectedBuffers = Count + (packed ? 2u : Count);
    const auto snapshotBytes = packed ? 2u * Recorder::DrawUploadChunk : Count * Bytes;
    const auto geometry = DescribeSurface(Resource(memory.Address(), Side, format));
    const auto linearBytes = Count * geometry.sliceLinearBytes * geometry.layers;
    for (std::size_t index = 0; index < Count; ++index) {
        const auto address = memory.Address() + index * Bytes;
        std::memset(reinterpret_cast<void*>(address), 0, Bytes);
        GM::CollectWritesUncached(address, Bytes);
    }
    recorder.Sync();
    const auto check = [&](const Texture& texture, std::uint32_t expected) {
        Buffer output(context, Bytes * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const TexelFetch fetch(context, texture, output, TexelFetchShader(Side, std::nullopt, compressed));
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), Side * Side);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        output.Invalidate();
        for (std::size_t pixel = 0; pixel < Side * Side; ++pixel) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, output.Bytes().data() + pixel * 16u, sizeof(actual));
            Require(actual == expected, "packed snapshot changed before GPU completion or used the wrong slice offset");
        }
    };
    for (std::size_t index = 0; index < Count; ++index) check(*textures[index], value(index));
    auto changed = Sampled(context, memory.Address(), Side, format);
    Require(changed != textures.front(), "packed upload reused an obsolete guest snapshot");
    recorder.Sync();
    check(*changed, 0);
    check(*textures.back(), value(Count - 1u));
    auto dropped = Sampled(context, memory.Address() + Count * Bytes, Side, format);
    ClearCachedTextures(context.device);
    dropped.reset();
    recorder.Sync();
    auto laterDraw = recorder.DrawUpload(16);
    Require(laterDraw.buffer == draw.buffer && laterDraw.offset == draw.offset + std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16), "texture uploads consumed a persistent draw upload block");
    Require(recorder.UnwrittenDrawSnapshot(memory.Address(), 16, drawStamp) == draw.buffer, "texture uploads invalidated a persistent draw snapshot");
    std::printf("sampled_packed_upload packed=%u compressed=%u textures=%zu buffer_creates=%llu kept_bytes=%zu pixels_checked=%zu\n", static_cast<unsigned>(packed), static_cast<unsigned>(compressed), Count, static_cast<unsigned long long>(created), retained, (Count + 2u) * Side * Side);
    Require(created == expectedBuffers, "small texture snapshots created an unexpected number of Vulkan buffers: " + std::to_string(created));
    Require(retained == imageBytes + snapshotBytes + linearBytes, "packed snapshot buffers were not accounted once at their full size");
}

class UploadCopyCounter {
public:
    explicit UploadCopyCounter(Context& context) : context(context), previous(context.functions), functions(previous != nullptr ? *previous : DeviceFunctions{}) {
        Require(copy == nullptr, "nested snapshot copy counters");
        copy = context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer");
        count = 0;
        functions.cmdCopyBuffer = Copy;
        context.functions = &functions;
    }
    ~UploadCopyCounter() {
        context.functions = previous;
        copy = nullptr;
    }
    std::uint64_t Count() const { return count; }
private:
    static void VKAPI_CALL Copy(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t regions, const VkBufferCopy* copies) {
        ++count;
        copy(commands, source, destination, regions, copies);
    }
    Context& context;
    const DeviceFunctions* previous;
    DeviceFunctions functions;
    static inline thread_local PFN_vkCmdCopyBuffer copy = nullptr;
    static inline thread_local std::uint64_t count = 0;
};

void directSnapshotTests(Context& context, bool expectDirect, bool compressed) {
    namespace GM = AgcDriver::GuestMemory;
    constexpr std::uint32_t TestSide = 128u;
    constexpr std::size_t TestBytes = std::size_t{TestSide} * TestSide * 4u;
    const std::uint32_t Initial = compressed ? 0x3f800000u : 0x17395badu;
    const std::uint32_t Changed = compressed ? 0u : 0x2468ace1u;
    const auto format = compressed ? 169u : Format32UInt;
    const auto checkPixels = [&](const Texture& texture, std::uint32_t expected) {
        if (auto* active = Recorder::Active(); active != nullptr) active->Sync();
        Buffer output(context, TestBytes * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
        const TexelFetch fetch(context, texture, output, TexelFetchShader(TestSide, std::nullopt, compressed));
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), TestSide * TestSide);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        output.Invalidate();
        for (std::size_t pixel = 0; pixel < TestSide * TestSide; ++pixel) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, output.Bytes().data() + pixel * 16u, 4u);
            Require(actual == expected, "direct snapshot returned an incorrect texel at " + std::to_string(pixel));
        }
    };
    UploadCopyCounter copies(context);
    std::shared_ptr<Texture> original;
    std::size_t temporaryBytes = 0;
    VkDeviceSize imageBytes = 0;
    std::uint64_t bufferCopies = 0;
    {
        SnapshotUploadMemory memory(context, 3u * TestBytes);
        const auto address = memory.Address() + TestBytes;
        if (compressed) std::fill_n(reinterpret_cast<std::uint64_t*>(address), TestBytes / 8u, 0xf800ull);
        else std::fill_n(reinterpret_cast<std::uint32_t*>(address), TestBytes / 4u, Initial);
        GM::CollectWritesUncached(address, TestBytes);
        Require(WatchedAfterSettle(context, address, TestBytes), "direct snapshot fixture lost write tracking");
        GM::CollectWritesUncached(address, TestBytes);
        Recorder recorder(context);
        recorder.Activate();
        original = Sampled(context, address, TestSide, format);
        bufferCopies = copies.Count();
        Require(original != nullptr && !original->ViewsStorageImage(), "direct snapshot must own an independent image");
        temporaryBytes = recorder.OpenKeptBytes();
        imageBytes = original->AllocationBytes();
        Require(Sampled(context, address, TestSide, format) == original, "unchanged direct snapshot was rebuilt");
        if (compressed) std::fill_n(reinterpret_cast<std::uint64_t*>(address), TestBytes / 8u, 0x07e0ull);
        else std::fill_n(reinterpret_cast<std::uint32_t*>(address), TestBytes / 4u, Changed);
        GM::CollectWritesUncached(address, TestBytes);
        checkPixels(*original, Initial);
        const auto changed = Sampled(context, address, TestSide, format);
        Require(changed != original, "guest mutation reused an obsolete direct snapshot");
        checkPixels(*changed, Changed);
        checkPixels(*original, Initial);
        ClearCachedTextures(context.device);
        auto dropped = Sampled(context, memory.Address(), TestSide, format);
        ClearCachedTextures(context.device);
        dropped.reset();
        recorder.Sync();
    }
    checkPixels(*original, Initial);
    const auto limit = imageBytes + 2u * TestBytes + 4096u;
    std::printf("sampled_snapshot_upload kept_bytes=%zu image_bytes=%llu input_bytes=%zu pixels_checked=%u expect_direct=%u compressed=%u buffer_copies=%llu\n", temporaryBytes, static_cast<unsigned long long>(imageBytes), TestBytes, 4u * TestSide * TestSide, static_cast<unsigned>(expectDirect), static_cast<unsigned>(compressed), static_cast<unsigned long long>(bufferCopies));
    Require(bufferCopies == (expectDirect ? 0u : 1u), "snapshot upload issued an unexpected number of GPU buffer copies");
    if (expectDirect) Require(temporaryBytes <= limit, "direct snapshot retained a redundant tiled upload buffer");
}

constexpr std::uint32_t WeakSnapshotSide = 256u;
constexpr std::size_t WeakPartialCommitBytes = 65536u;
constexpr std::size_t WeakSnapshotBytes = std::size_t{WeakSnapshotSide} * WeakSnapshotSide * 4u;

class WeakPartialSnapshotBlock {
public:
    explicit WeakPartialSnapshotBlock(const Context& context) : context(context) {
        memory = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(WeakSnapshotBytes, WeakPartialCommitBytes));
        Require(memory != nullptr, "partial weak snapshot: cannot reserve the guest surface");
        Register();
    }
    ~WeakPartialSnapshotBlock() {
        ClearCachedTextures(context.device);
        if (registered) GuestAllocations::Mutation().Remove(memory);
        GuestArena::GuestArenaReset_nid_postfix(memory, WeakSnapshotBytes);
        GuestArena::GuestArenaRelease_nid_postfix(memory, WeakSnapshotBytes);
    }
    WeakPartialSnapshotBlock(const WeakPartialSnapshotBlock&) = delete;
    WeakPartialSnapshotBlock& operator=(const WeakPartialSnapshotBlock&) = delete;
    void Commit(std::size_t offset, std::size_t bytes) {
        GuestArena::GuestArenaCommit_nid_postfix(memory + offset, bytes, PAGE_READWRITE, bytes);
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(reinterpret_cast<std::uintptr_t>(memory + offset), bytes);
        for (std::size_t page = 0; page < bytes; page += 4096u) static_cast<void>(static_cast<volatile const std::byte*>(memory + offset)[page]);
    }
    void Decommit(std::size_t offset, std::size_t bytes) {
        GuestArena::GuestArenaReset_nid_postfix(memory + offset, bytes);
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(reinterpret_cast<std::uintptr_t>(memory + offset), bytes);
    }
    void Unregister() {
        if (!registered) return;
        GuestAllocations::Mutation().Remove(memory);
        registered = false;
    }
    void Register() {
        if (registered) return;
        GuestAllocations::Mutation().Add(memory, WeakSnapshotBytes, true, true, true);
        registered = true;
    }
    std::byte* Data() const { return memory; }
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(memory); }
private:
    const Context& context;
    std::byte* memory = nullptr;
    bool registered = false;
};

void partialCommittedWeakSnapshotTests(const Context& context) {
    namespace GM = AgcDriver::GuestMemory;
    constexpr std::uint32_t Initial = 0x1256a9efu;
    constexpr std::uint32_t CpuWrite = 0x98badcfeu;
    constexpr std::uint32_t MappedWrite = 0x71625344u;
    constexpr std::uint32_t PendingWrite = 0x392b17a5u;
    ClearCachedTextures(context.device);
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "partial weak snapshot: cache fillers did not occupy the budget");
    const auto baselineMemory = SampledTextureMemory();
    WeakPartialSnapshotBlock memory(context);
    const auto address = memory.Address();
    auto words = Descriptor(address, WeakSnapshotSide);
    words[3] &= ~(0x1fu << 20u);
    Require(DescribeSurface(DecodeTextureResource(words)).guestBytes == WeakSnapshotBytes, "partial weak snapshot: descriptor extent differs from its reservation");
    memory.Commit(0, WeakPartialCommitBytes);
    std::fill_n(reinterpret_cast<std::uint32_t*>(memory.Data()), WeakPartialCommitBytes / sizeof(std::uint32_t), Initial);
    GM::BumpCollectEpoch();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    const auto generation = GM::CollectCommittedWrites(address, WeakSnapshotBytes, runs);
    Require(generation != 0 && runs.size() == 1 && runs.front() == std::pair{address, address + WeakPartialCommitBytes}, "partial weak snapshot: exact initial committed proof unavailable");
    std::vector<std::shared_ptr<Texture>> owners;
    const auto lookup = [&] {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, words);
    };
    owners.push_back(lookup());
    auto first = owners.back();
    Require(first != nullptr && !first->ViewsStorageImage() && first->AllocationBytes() != 0 && first->AllocationBytes() <= Budget / 4u, "partial weak snapshot: partial surface was not a small owned snapshot");
    Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries, "partial weak snapshot: current-frame partial surface entered the full strong cache");
    const auto initialLive = SampledTextureMemory();
    Require(FetchFirstTexel(context, *first, WeakSnapshotSide) == Initial, "partial weak snapshot: initial sampled pixel differs");
    auto same = lookup();
    Require(same == first && SampledTextureMemory() == initialLive, "partial weak snapshot: unchanged committed runs did not reuse the held native image");
    Require(FetchFirstTexel(context, *same, WeakSnapshotSide) == Initial, "partial weak snapshot: reused image sampled the wrong initial pixel");
    first.reset();
    same.reset();
    const auto fresh = [&](const char* label, std::uint32_t expected) {
        const auto before = SampledTextureMemory();
        auto next = lookup();
        Require(next != nullptr && !next->ViewsStorageImage(), std::string("partial weak snapshot: no owned image ") + label);
        Require(std::find(owners.begin(), owners.end(), next) == owners.end(), std::string("partial weak snapshot: reused a stale image ") + label);
        owners.push_back(next);
        Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries, std::string("partial weak snapshot: image entered the strong cache ") + label);
        Require(SampledTextureMemory() == before + next->AllocationBytes(), std::string("partial weak snapshot: native allocation accounting changed unexpectedly ") + label);
        Require(FetchFirstTexel(context, *next, WeakSnapshotSide) == expected, std::string("partial weak snapshot: sampled pixel differs ") + label);
    };
    std::memcpy(memory.Data(), &CpuWrite, sizeof(CpuWrite));
    fresh("after a committed-pixel CPU write", CpuWrite);
    memory.Commit(2u * WeakPartialCommitBytes, WeakPartialCommitBytes);
    std::fill_n(reinterpret_cast<std::uint32_t*>(memory.Data() + 2u * WeakPartialCommitBytes), WeakPartialCommitBytes / sizeof(std::uint32_t), 0x55667788u);
    fresh("after a new committed run", CpuWrite);
    memory.Decommit(2u * WeakPartialCommitBytes, WeakPartialCommitBytes);
    fresh("after a committed run was decommitted", CpuWrite);
    memory.Commit(2u * WeakPartialCommitBytes, WeakPartialCommitBytes);
    fresh("after that run was recommitted", CpuWrite);
    const auto beforeMapping = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    memory.Decommit(0, WeakPartialCommitBytes);
    memory.Unregister();
    memory.Commit(0, WeakPartialCommitBytes);
    std::fill_n(reinterpret_cast<std::uint32_t*>(memory.Data()), WeakPartialCommitBytes / sizeof(std::uint32_t), MappedWrite);
    memory.Register();
    const auto afterMapping = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    Require(afterMapping[0] != beforeMapping[0] && afterMapping[1] != beforeMapping[1], "partial weak snapshot: mapping replacement did not change allocation and forget stamps");
    fresh("after same-address mapping replacement", MappedWrite);
    const auto unwatched = GM::UnwatchSerial();
    GM::Unwatch(address, WeakPartialCommitBytes);
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> rewatch{{{address, address + WeakPartialCommitBytes}}};
    GM::Rewatch(rewatch);
    Require(GM::Watched(address, WeakPartialCommitBytes) && GM::UnwatchSerial() != unwatched, "partial weak snapshot: watch transition did not change the watch stamp");
    fresh("after unwatch and rewatch", MappedWrite);
    std::array<std::byte, 4096> unrelated{};
    const auto beforeAllocation = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    {
        GuestAllocations::Mutation mutation;
        mutation.Add(unrelated.data(), unrelated.size(), true, true, false);
    }
    {
        GuestAllocations::Mutation mutation;
        mutation.Remove(unrelated.data());
    }
    const auto afterAllocation = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    Require(afterAllocation[0] != beforeAllocation[0] && afterAllocation[1] != beforeAllocation[1], "partial weak snapshot: disjoint allocation did not change allocation and forget stamps");
    fresh("after a disjoint allocation and forget event", MappedWrite);
    {
        Recorder recorder(context);
        recorder.Activate();
        auto copied = std::make_shared<Buffer>(context, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::fill(copied->Bytes().begin(), copied->Bytes().end(), std::byte{0xa6});
        const auto landed = std::make_shared<bool>(false);
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, copied->Handle(), 0, sizeof(std::uint32_t), PendingWrite);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.OnComplete([copied, landed, address, PendingWrite] {
            copied->Invalidate();
            CheckSnapshotWords(copied->Bytes(), PendingWrite, "partial weak snapshot pending GPU output");
            GM::StoreOwnBytes(address, copied->Bytes());
            *landed = true;
        });
        recorder.Keep(copied);
        recorder.NotePendingWrite(address, sizeof(std::uint32_t));
        Require(Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "partial weak snapshot: pending GPU writer was not registered");
        fresh("after a pending GPU write was published", PendingWrite);
        Require(*landed && !Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "partial weak snapshot: lookup returned before the pending writer published");
        recorder.Sync();
    }
    const auto beforeSettled = SampledTextureMemory();
    auto settled = lookup();
    Require(settled != nullptr, "partial weak snapshot: no image after the pending write settled");
    if (settled != owners.back()) {
        Require(std::find(owners.begin(), owners.end(), settled) == owners.end(), "partial weak snapshot: stale owner returned after the pending write settled");
        Require(SampledTextureMemory() == beforeSettled + settled->AllocationBytes(), "partial weak snapshot: settled capture allocation accounting differs");
        owners.push_back(settled);
    }
    Require(FetchFirstTexel(context, *settled, WeakSnapshotSide) == PendingWrite, "partial weak snapshot: settled capture sampled old writer data");
    const auto settledMemory = SampledTextureMemory();
    auto stable = lookup();
    Require(stable == settled && SampledTextureMemory() == settledMemory, "partial weak snapshot: fresh post-writer capture did not stabilize");
    Require(FetchFirstTexel(context, *stable, WeakSnapshotSide) == PendingWrite, "partial weak snapshot: stable capture sampled old writer data");
    Require(SampledTextureMemory() == std::accumulate(owners.begin(), owners.end(), baselineMemory, [](std::uint64_t bytes, const auto& texture) { return bytes + texture->AllocationBytes(); }), "partial weak snapshot: held native image accounting does not equal allocation sizes");
    const auto made = owners.size();
    stable.reset();
    settled.reset();
    ClearCachedTextures(context.device);
    owners.clear();
    Require(SampledTextureMemory() == baselineMemory, "partial weak snapshot: held native images did not retire after release");
    std::printf("partial_committed_weak_live=PASS committed_bytes=%zu distinct_owners=%zu\n", WeakPartialCommitBytes, made);
}

constexpr std::size_t UntrackedWeakSurfaceBytes = std::size_t{Side} * Side * 4u;
constexpr std::size_t UntrackedWeakSurfaceCount = 6u;
constexpr std::size_t UntrackedWeakBytes = UntrackedWeakSurfaceBytes * UntrackedWeakSurfaceCount;
constexpr std::uint32_t UntrackedInitial = 0x5a5a5a5au;
constexpr std::uint32_t UntrackedCpuWrite = 0x24136aefu;
constexpr std::uint32_t UntrackedPendingWrite = 0x9c51d782u;

class UntrackedWeakSnapshotBlock {
public:
    explicit UntrackedWeakSnapshotBlock(const Context& context, std::size_t surfaces = UntrackedWeakSurfaceCount) : context(context), bytes(UntrackedWeakSurfaceBytes * surfaces) {
        memory = static_cast<std::byte*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(memory != nullptr, "untracked weak snapshot: cannot allocate guest bytes");
        Register();
        Initialize(UntrackedInitial);
    }
    ~UntrackedWeakSnapshotBlock() {
        ClearCachedTextures(context.device);
        if (registered) GuestAllocations::Mutation().Remove(memory);
        if (memory != nullptr) VirtualFree(memory, 0, MEM_RELEASE);
    }
    UntrackedWeakSnapshotBlock(const UntrackedWeakSnapshotBlock&) = delete;
    UntrackedWeakSnapshotBlock& operator=(const UntrackedWeakSnapshotBlock&) = delete;
    std::byte* Data() const { return memory; }
    std::uint64_t Address(std::size_t surface = 0) const { return reinterpret_cast<std::uint64_t>(memory + surface * UntrackedWeakSurfaceBytes); }
    void Decommit(std::size_t offset, std::size_t bytes) {
        Require(VirtualFree(memory + offset, bytes, MEM_DECOMMIT) != 0, "untracked weak snapshot: cannot decommit guest pages");
    }
    void Commit(std::size_t offset, std::size_t bytes) {
        Require(VirtualAlloc(memory + offset, bytes, MEM_COMMIT, PAGE_READWRITE) == memory + offset, "untracked weak snapshot: cannot recommit guest pages");
    }
    void Replace(std::uint32_t firstPixel) {
        const auto old = memory;
        GuestAllocations::Mutation().Remove(old);
        registered = false;
        Require(VirtualFree(old, 0, MEM_RELEASE) != 0, "untracked weak snapshot: cannot release old guest mapping");
        memory = nullptr;
        auto* replacement = static_cast<std::byte*>(VirtualAlloc(old, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (replacement != old) {
            if (replacement != nullptr) VirtualFree(replacement, 0, MEM_RELEASE);
            throw std::runtime_error("untracked weak snapshot: guest mapping replacement did not reuse its address");
        }
        memory = replacement;
        Register();
        Initialize(UntrackedInitial);
        std::memcpy(memory, &firstPixel, sizeof(firstPixel));
    }
private:
    void Initialize(std::uint32_t pixel) {
        std::memset(memory, static_cast<int>(pixel & 0xffu), bytes);
        std::memset(memory + 4096u, 0, 4096u);
    }
    void Register() {
        GuestAllocations::Mutation().Add(memory, bytes, true, true, false);
        registered = true;
    }
    const Context& context;
    std::size_t bytes;
    std::byte* memory = nullptr;
    bool registered = false;
};

std::array<std::uint32_t, 8> LinearUntrackedDescriptor(std::uint64_t address) {
    auto words = Descriptor(address, Side, Format32UInt);
    words[3] &= ~(0x1fu << 20u);
    return words;
}

std::size_t CurrentProcessProofArenaFileCount() {
#ifdef _WIN32
    std::wstring directory(32768, L'\0');
    const auto length = GetTempPathW(static_cast<DWORD>(directory.size()), directory.data());
    Require(length != 0 && length < directory.size(), "file-backed weak snapshot: cannot query the temp directory");
    directory.resize(length);
    if (directory.back() != L'\\' && directory.back() != L'/') directory.push_back(L'\\');
    const auto pattern = directory + L"AnyPS5-proof-" + std::to_wstring(GetCurrentProcessId()) + L"-*.tmp";
    WIN32_FIND_DATAW data{};
    const auto find = FindFirstFileW(pattern.c_str(), &data);
    if (find == INVALID_HANDLE_VALUE) {
        const auto error = GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_NO_MORE_FILES) return 0;
        throw std::runtime_error("file-backed weak snapshot: cannot enumerate per-process arena files");
    }
    std::size_t count = 0;
    do {
        if ((data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) ++count;
    } while (FindNextFileW(find, &data));
    const auto error = GetLastError();
    FindClose(find);
    Require(error == ERROR_NO_MORE_FILES, "file-backed weak snapshot: arena file enumeration failed");
    return count;
#else
    return 0;
#endif
}

void untrackedWeakSnapshotTests(const Context& context, bool capacityOnly, bool requireFileBackend = false) {
    namespace GM = AgcDriver::GuestMemory;
    const auto* proofBudget = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_MIB");
    Require(proofBudget != nullptr && std::string_view(proofBudget) == "1", "untracked weak snapshot: fixture requires a 1 MiB byte-proof budget");
    if (requireFileBackend) {
        const auto* fileMode = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_FILE");
        const auto* compression = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_COMPRESS");
        Require(fileMode != nullptr && std::string_view(fileMode) == "1", "file-backed weak snapshot: fixture requires file mode");
        Require(compression != nullptr && std::string_view(compression) == "0", "file-backed weak snapshot: fixture requires raw bytes without compression");
    }
    if (!GM::WriteWatched()) return;
    ClearCachedTextures(context.device);
    const auto initialMemory = SampledTextureMemory();
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "untracked weak snapshot: strong-cache fillers did not occupy exactly 1 MiB");
    if (requireFileBackend) Require(CurrentProcessProofArenaFileCount() == 0, "file-backed weak snapshot: an arena file existed before the target weak proof");
    const auto baselineMemory = SampledTextureMemory();
    UntrackedWeakSnapshotBlock memory(context);
    const auto verifyNoProof = [&](std::size_t surface) {
        const auto address = memory.Address(surface);
        Require(!GM::Watched(address, UntrackedWeakSurfaceBytes), "untracked weak snapshot: fixture bytes unexpectedly entered the write watch");
        GM::BumpCollectEpoch();
        Require(GM::CollectWrites(address, UntrackedWeakSurfaceBytes) == 0, "untracked weak snapshot: fixture unexpectedly has a complete write generation");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        Require(GM::CollectCommittedWrites(address, UntrackedWeakSurfaceBytes, committed) == 0 && committed.empty(), "untracked weak snapshot: fixture unexpectedly has a committed-write proof");
        Require(GM::UntrackedHint(address, UntrackedWeakSurfaceBytes) == 0, "untracked weak snapshot: fixture unexpectedly has an untracked-write hint");
    };
    const auto lookup = [&](std::size_t surface) {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, LinearUntrackedDescriptor(memory.Address(surface)));
    };
    std::vector<std::shared_ptr<Texture>> owners;
    verifyNoProof(0);
    auto first = lookup(0);
    if (requireFileBackend) Require(CurrentProcessProofArenaFileCount() == 1, "file-backed weak snapshot: target proof did not create exactly one live per-process arena file");
    Require(first != nullptr && !first->ViewsStorageImage() && first->AllocationBytes() > 0 && first->AllocationBytes() <= Budget / 4u, "untracked weak snapshot: linear image is not a small owned snapshot");
    owners.push_back(first);
    Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "untracked weak snapshot: target entered the full strong cache");
    Require(FetchFirstTexel(context, *first, Side) == UntrackedInitial, "untracked weak snapshot: initial sampled texel differs");
    const auto firstBytes = SampledTextureMemory();
    auto same = lookup(0);
    Require(same == first && SampledTextureMemory() == firstBytes, "untracked weak snapshot: byte-identical no-proof lookup did not reuse its live owner");
    Require(FetchFirstTexel(context, *same, Side) == UntrackedInitial, "untracked weak snapshot: reused image sampled the wrong initial texel");
    first.reset();
    same.reset();
    const auto fresh = [&](std::size_t surface, const char* label, std::uint32_t expected) {
        const auto before = SampledTextureMemory();
        auto next = lookup(surface);
        Require(next != nullptr && !next->ViewsStorageImage(), std::string("untracked weak snapshot: no owned image ") + label);
        Require(std::find(owners.begin(), owners.end(), next) == owners.end(), std::string("untracked weak snapshot: stale image returned ") + label);
        owners.push_back(next);
        Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, std::string("untracked weak snapshot: target entered the strong cache ") + label);
        Require(SampledTextureMemory() == before + next->AllocationBytes(), std::string("untracked weak snapshot: native allocation accounting changed ") + label);
        Require(FetchFirstTexel(context, *next, Side) == expected, std::string("untracked weak snapshot: sampled texel differs ") + label);
        return next;
    };
    std::shared_ptr<Texture> pendingOwner;
    if (!capacityOnly) {
    std::uint32_t cpuWrite = UntrackedCpuWrite;
    std::memcpy(memory.Data(), &cpuWrite, sizeof(cpuWrite));
    verifyNoProof(0);
    auto cpu = fresh(0, "after an untracked CPU pixel write", UntrackedCpuWrite);
    cpu.reset();
    memory.Decommit(4096u, 4096u);
    auto decommitted = fresh(0, "after a zero page was decommitted", UntrackedCpuWrite);
    decommitted.reset();
    memory.Commit(4096u, 4096u);
    std::memset(memory.Data() + 4096u, 0, 4096u);
    auto recommitted = fresh(0, "after the same zero page was recommitted", UntrackedCpuWrite);
    recommitted.reset();
    const auto beforeMapping = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    memory.Replace(UntrackedCpuWrite);
    const auto afterMapping = std::array<std::uint64_t, 2>{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GM::ForgetSerial()};
    Require(afterMapping[0] != beforeMapping[0] && afterMapping[1] != beforeMapping[1], "untracked weak snapshot: same-address mapping replacement did not change allocation and forget stamps");
    auto remapped = fresh(0, "after same-address mapping replacement with identical bytes", UntrackedCpuWrite);
    remapped.reset();
    {
        Recorder recorder(context);
        recorder.Activate();
        auto copied = std::make_shared<Buffer>(context, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::fill(copied->Bytes().begin(), copied->Bytes().end(), std::byte{0xa6});
        const auto landed = std::make_shared<bool>(false);
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, copied->Handle(), 0, sizeof(std::uint32_t), UntrackedPendingWrite);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        const auto address = memory.Address();
        recorder.OnComplete([copied, landed, address] {
            copied->Invalidate();
            CheckSnapshotWords(copied->Bytes(), UntrackedPendingWrite, "untracked weak snapshot pending GPU output");
            GM::StoreOwnBytes(address, copied->Bytes());
            *landed = true;
        });
        recorder.Keep(copied);
        recorder.NotePendingWrite(address, sizeof(std::uint32_t));
        Require(Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "untracked weak snapshot: pending GPU writer was not registered");
        pendingOwner = fresh(0, "after a pending GPU write", UntrackedPendingWrite);
        Require(*landed && !Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "untracked weak snapshot: lookup returned before the pending GPU writer completed");
        Require(*reinterpret_cast<const std::uint32_t*>(memory.Data()) == UntrackedPendingWrite, "untracked weak snapshot: pending writer did not update guest bytes");
        recorder.Sync();
    }
    }
    if (capacityOnly) {
    std::array<std::shared_ptr<Texture>, 3> additional{};
    for (std::size_t i = 0; i < additional.size(); ++i) {
        verifyNoProof(i + 1u);
        additional[i] = fresh(i + 1u, "while filling the CPU byte-proof budget", UntrackedInitial);
    }
    auto refreshFillers = [&] {
        for (std::size_t i = 0; i < fillerOwners.size(); ++i) Require(Sampled(context, fillerMemory.Surface(i)) == fillerOwners[i], "untracked weak snapshot: strong filler changed while maintaining first-use admission");
        Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "untracked weak snapshot: filler refresh changed strong-cache capacity");
    };
    verifyNoProof(4u);
    refreshFillers();
    const auto beforeOmitted = SampledTextureMemory();
    auto omittedFirst = lookup(4u);
    Require(omittedFirst != nullptr && std::find(owners.begin(), owners.end(), omittedFirst) == owners.end(), "untracked weak snapshot: budget-pressure sample returned an existing owner");
    Require(SampledTextureMemory() == beforeOmitted + omittedFirst->AllocationBytes(), "untracked weak snapshot: budget-pressure sample did not allocate a fresh image");
    Require(FetchFirstTexel(context, *omittedFirst, Side) == UntrackedInitial, "untracked weak snapshot: budget-pressure sample has wrong pixels");
    owners.push_back(omittedFirst);
    omittedFirst.reset();
    refreshFillers();
    const auto beforeRepeatedOmitted = SampledTextureMemory();
    auto omittedAgain = lookup(4u);
    Require(omittedAgain != nullptr && std::find(owners.begin(), owners.end(), omittedAgain) == owners.end(), "untracked weak snapshot: full proof budget retained a fifth proof instead of skipping it");
    Require(SampledTextureMemory() == beforeRepeatedOmitted + omittedAgain->AllocationBytes(), "untracked weak snapshot: omitted proof lookup did not make a fresh image");
    Require(FetchFirstTexel(context, *omittedAgain, Side) == UntrackedInitial, "untracked weak snapshot: second budget-pressure sample has wrong pixels");
    owners.push_back(omittedAgain);
    omittedAgain.reset();
    refreshFillers();
    for (std::size_t i = 0; i < additional.size() + 1u; ++i) {
        const auto beforeRetained = SampledTextureMemory();
        auto retained = lookup(i);
        Require(retained == owners[i] && SampledTextureMemory() == beforeRetained, "untracked weak snapshot: adding a fifth proof evicted a live proof instead of skipping it");
        Require(FetchFirstTexel(context, *retained, Side) == UntrackedInitial, "untracked weak snapshot: retained proof sampled stale data");
    }
    additional.fill(nullptr);
    } else {
        pendingOwner.reset();
    }
    const auto heldBytes = std::accumulate(owners.begin(), owners.end(), baselineMemory, [](std::uint64_t bytes, const auto& texture) { return bytes + texture->AllocationBytes(); });
    Require(SampledTextureMemory() == heldBytes, "untracked weak snapshot: held-owner accounting differs from live native allocations");
    const auto made = owners.size();
    owners.clear();
    fillerOwners.fill(nullptr);
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == initialMemory, "untracked weak snapshot: live image owners did not retire after release");
    if (requireFileBackend) Require(CurrentProcessProofArenaFileCount() == 0, "file-backed weak snapshot: cache clear did not close and remove the arena file");
    std::printf("untracked_weak_live=PASS image_bytes=%zu distinct_owners=%zu\n", UntrackedWeakSurfaceBytes, made);
}


template <typename Device>
std::optional<VkDeviceSize> DeviceScopedReleaseForTest(VkDeviceSize bytes, bool newImage, Device device) {
    if constexpr (requires { ReleaseTextureCaches(bytes, newImage, device); }) return ReleaseTextureCaches(bytes, newImage, device);
    else return std::nullopt;
}

void fileBackedNativeRetentionTests(const Context& context) {
    namespace GM = AgcDriver::GuestMemory;
    const auto* proofBudget = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_MIB");
    const auto* fileMode = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_FILE");
    const auto* compression = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_COMPRESS");
    const auto* retentionBudget = std::getenv("APS5_LIVE_TEXTURE_RETAIN_MIB");
    Require(proofBudget != nullptr && std::string_view(proofBudget) == "2", "file-backed native retention: fixture requires a 2 MiB byte-proof budget");
    Require(fileMode != nullptr && std::string_view(fileMode) == "1", "file-backed native retention: fixture requires file mode");
    Require(compression != nullptr && std::string_view(compression) == "0", "file-backed native retention: fixture requires raw byte proofs");
    Require(retentionBudget != nullptr && std::string_view(retentionBudget) == "1", "file-backed native retention: fixture requires a 1 MiB native-retention budget");
    if (!GM::WriteWatched()) return;
    ClearCachedTextures(context.device);
    const auto initialMemory = SampledTextureMemory();
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "file-backed native retention: strong-cache fillers did not occupy exactly 1 MiB");
    Require(CurrentProcessProofArenaFileCount() == 0, "file-backed native retention: an arena file existed before weak snapshots");
    const auto baselineMemory = SampledTextureMemory();
    UntrackedWeakSnapshotBlock memory(context);
    const auto verifyNoProof = [&](std::size_t surface) {
        const auto address = memory.Address(surface);
        Require(!GM::Watched(address, UntrackedWeakSurfaceBytes), "file-backed native retention: fixture bytes unexpectedly entered the write watch");
        GM::BumpCollectEpoch();
        Require(GM::CollectWrites(address, UntrackedWeakSurfaceBytes) == 0, "file-backed native retention: fixture unexpectedly has a complete write generation");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        Require(GM::CollectCommittedWrites(address, UntrackedWeakSurfaceBytes, committed) == 0 && committed.empty(), "file-backed native retention: fixture unexpectedly has a committed-write proof");
        Require(GM::UntrackedHint(address, UntrackedWeakSurfaceBytes) == 0, "file-backed native retention: fixture unexpectedly has an untracked-write hint");
    };
    const auto lookup = [&](std::size_t surface) {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, LinearUntrackedDescriptor(memory.Address(surface)));
    };
    std::array<std::weak_ptr<Texture>, UntrackedWeakSurfaceCount> weakOwners;
    std::array<std::uint64_t, UntrackedWeakSurfaceCount> allocationBytes{};
    std::array<bool, UntrackedWeakSurfaceCount> releasedStates{};
    std::uint64_t allNativeBytes = 0;
    {
        Recorder recorder(context);
        recorder.Activate();
        for (std::size_t i = 0; i < weakOwners.size(); ++i) {
            verifyNoProof(i);
            auto texture = lookup(i);
            Require(texture != nullptr && !texture->ViewsStorageImage(), "file-backed native retention: weak lookup did not produce an owned sampled image");
            Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "file-backed native retention: weak image entered the strong cache");
            Require(SampledTextureMemory() >= baselineMemory + texture->AllocationBytes(), "file-backed native retention: sampled allocation accounting did not advance");
            allocationBytes[i] = texture->AllocationBytes();
            releasedStates[i] = texture->Released();
            allNativeBytes += allocationBytes[i];
            weakOwners[i] = texture;
            if (i == 0) {
                Require(CurrentProcessProofArenaFileCount() == 1, "file-backed native retention: target did not create one raw proof arena file");
                recorder.Keep(texture);
            }
            texture.reset();
        }
        Require(allNativeBytes > (1u << 20u), "file-backed native retention: fixture did not exceed the configured native-retention budget");
        auto inFlight = weakOwners[0].lock();
        Require(inFlight != nullptr && inFlight->Released() == releasedStates[0], "file-backed native retention: LRU pressure changed the released state of a texture still held by an in-flight recorder");
        inFlight.reset();
        recorder.Sync();
    }
    Require(weakOwners[0].expired(), "file-backed native retention: oldest native image was not demoted after recorder retirement");
    auto retainedBeforeDeviceProbe = weakOwners.back().lock();
    Require(retainedBeforeDeviceProbe != nullptr, "file-backed native retention: newest native image did not survive caller and recorder retirement");
    const auto retainedBytes = retainedBeforeDeviceProbe->AllocationBytes();
    retainedBeforeDeviceProbe.reset();
    const auto wrongDevice = reinterpret_cast<VkDevice>(std::uintptr_t{1});
    Require(wrongDevice != context.device, "file-backed native retention: wrong-device sentinel aliases the requesting device");
    const auto beforeWrongDevice = SampledTextureMemory();
    const auto wrongDeviceReleased = DeviceScopedReleaseForTest(retainedBytes, true, wrongDevice);
    if (wrongDeviceReleased.has_value()) Require(*wrongDeviceReleased == 0 && SampledTextureMemory() == beforeWrongDevice, "file-backed native retention: wrong-device pressure reclaimed or demoted a different device's image");
    auto retained = weakOwners.back().lock();
    Require(retained != nullptr && retained->Released() == releasedStates.back(), "file-backed native retention: wrong-device pressure demoted the kept image or changed its released state");
    const auto beforeReuse = SampledTextureMemory();
    auto same = lookup(UntrackedWeakSurfaceCount - 1u);
    Require(same == retained && same->Released() == releasedStates.back() && SampledTextureMemory() == beforeReuse, "file-backed native retention: new lookup did not reuse the kept native image with its released state unchanged");
    Require(FetchFirstTexel(context, *same, Side) == UntrackedInitial, "file-backed native retention: reused native image sampled the wrong initial texel");
    const auto changed = UntrackedCpuWrite;
    std::memcpy(memory.Data() + (UntrackedWeakSurfaceCount - 1u) * UntrackedWeakSurfaceBytes, &changed, sizeof(changed));
    auto updated = lookup(UntrackedWeakSurfaceCount - 1u);
    Require(updated != nullptr && updated != same, "file-backed native retention: changed guest bytes did not build a fresh native image");
    const auto updatedReleased = updated->Released();
    Require(FetchFirstTexel(context, *updated, Side) == changed, "file-backed native retention: fresh image sampled stale guest pixels");
    Require(FetchFirstTexel(context, *same, Side) == UntrackedInitial, "file-backed native retention: old native image changed after guest mutation");
    Require(same->Released() == releasedStates.back() && updated->Released() == updatedReleased, "file-backed native retention: guest mutation changed a native image released state");
    same.reset();
    retained.reset();
    updated.reset();
    fillerOwners.fill(nullptr);
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == initialMemory, "file-backed native retention: cache clear did not retire the retained native images");
    for (const auto& weak : weakOwners) Require(weak.expired(), "file-backed native retention: cache clear left a weak native image alive");
    Require(CurrentProcessProofArenaFileCount() == 0, "file-backed native retention: cache clear did not remove the proof arena file");
    std::printf("file_raw_native_retention=PASS image_bytes=%llu newest_image_bytes=%llu entries=%zu\n", static_cast<unsigned long long>(allNativeBytes), static_cast<unsigned long long>(retainedBytes), weakOwners.size());
}

void fileBackedNativeArenaPressureTests(const Context& context) {
    namespace GM = AgcDriver::GuestMemory;
    constexpr std::size_t SurfaceCount = 9;
    const auto* proofBudget = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_MIB");
    const auto* fileMode = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_FILE");
    const auto* compression = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_COMPRESS");
    const auto* retentionBudget = std::getenv("APS5_LIVE_TEXTURE_RETAIN_MIB");
    Require(proofBudget != nullptr && std::string_view(proofBudget) == "2", "file-backed native arena pressure: fixture requires a 2 MiB byte-proof budget");
    Require(fileMode != nullptr && std::string_view(fileMode) == "1", "file-backed native arena pressure: fixture requires file mode");
    Require(compression != nullptr && std::string_view(compression) == "0", "file-backed native arena pressure: fixture requires raw byte proofs");
    Require(retentionBudget != nullptr && std::string_view(retentionBudget) == "4", "file-backed native arena pressure: fixture requires a 4 MiB native-retention budget");
    if (!GM::WriteWatched()) return;
    ClearCachedTextures(context.device);
    const auto initialMemory = SampledTextureMemory();
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "file-backed native arena pressure: strong-cache fillers did not occupy exactly 1 MiB");
    const auto baselineMemory = SampledTextureMemory();
    UntrackedWeakSnapshotBlock memory(context, SurfaceCount);
    const auto verifyNoProof = [&](std::size_t surface) {
        const auto address = memory.Address(surface);
        Require(!GM::Watched(address, UntrackedWeakSurfaceBytes), "file-backed native arena pressure: fixture bytes unexpectedly entered the write watch");
        GM::BumpCollectEpoch();
        Require(GM::CollectWrites(address, UntrackedWeakSurfaceBytes) == 0, "file-backed native arena pressure: fixture unexpectedly has a complete write generation");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        Require(GM::CollectCommittedWrites(address, UntrackedWeakSurfaceBytes, committed) == 0 && committed.empty(), "file-backed native arena pressure: fixture unexpectedly has a committed-write proof");
        Require(GM::UntrackedHint(address, UntrackedWeakSurfaceBytes) == 0, "file-backed native arena pressure: fixture unexpectedly has an untracked-write hint");
    };
    const auto lookup = [&](std::size_t surface) {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, LinearUntrackedDescriptor(memory.Address(surface)));
    };
    std::array<std::weak_ptr<Texture>, SurfaceCount> weakOwners;
    std::array<bool, SurfaceCount> releasedStates{};
    {
        Recorder recorder(context);
        recorder.Activate();
        for (std::size_t i = 0; i < weakOwners.size(); ++i) {
            verifyNoProof(i);
            auto texture = lookup(i);
            Require(texture != nullptr && !texture->ViewsStorageImage(), "file-backed native arena pressure: weak lookup did not produce an owned sampled image");
            Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "file-backed native arena pressure: weak image entered the strong cache");
            releasedStates[i] = texture->Released();
            weakOwners[i] = texture;
            if (i == 0) {
                Require(CurrentProcessProofArenaFileCount() == 1, "file-backed native arena pressure: target did not create one raw proof arena file");
                recorder.Keep(texture);
            }
            texture.reset();
        }
        auto inFlight = weakOwners[0].lock();
        Require(inFlight != nullptr && inFlight->Released() == releasedStates[0], "file-backed native arena pressure: arena reclaim changed the released state of an in-flight texture");
        inFlight.reset();
        recorder.Sync();
    }
    Require(weakOwners[0].expired(), "file-backed native arena pressure: in-flight oldest keeper did not retire after synchronization");
    Require(weakOwners[1].expired(), "file-backed native arena pressure: oldest non-in-flight keeper did not yield one arena block");
    auto retained = weakOwners[SurfaceCount - 2u].lock();
    Require(retained != nullptr && retained->Released() == releasedStates[SurfaceCount - 2u], "file-backed native arena pressure: an arena allocation miss demoted every keeper instead of reclaiming oldest blocks incrementally");
    const auto beforeReuse = SampledTextureMemory();
    auto same = lookup(SurfaceCount - 2u);
    Require(same == retained && SampledTextureMemory() == beforeReuse, "file-backed native arena pressure: the newest surviving native image was not reused");
    Require(FetchFirstTexel(context, *same, Side) == UntrackedInitial, "file-backed native arena pressure: the reused native image sampled the wrong texel");
    same.reset();
    retained.reset();
    fillerOwners.fill(nullptr);
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == initialMemory, "file-backed native arena pressure: retained images did not retire after cache clear");
    for (const auto& weak : weakOwners) Require(weak.expired(), "file-backed native arena pressure: cache clear left a native image alive");
    Require(CurrentProcessProofArenaFileCount() == 0, "file-backed native arena pressure: cache clear did not remove the proof arena file");
    std::puts("file_raw_native_arena_pressure=PASS");
}

void compressedUntrackedSnapshotTests(const Context& context) {
    namespace GM = AgcDriver::GuestMemory;
    const auto* proofBudget = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_MIB");
    const auto* compression = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_COMPRESS");
    Require(proofBudget != nullptr && std::string_view(proofBudget) == "1", "compressed weak snapshot: fixture requires a 1 MiB byte-proof budget");
    Require(compression != nullptr && std::string_view(compression) == "1", "compressed weak snapshot: fixture requires opt-in compression");
    if (!GM::WriteWatched()) return;
    ClearCachedTextures(context.device);
    const auto initialMemory = SampledTextureMemory();
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "compressed weak snapshot: strong-cache fillers did not occupy exactly 1 MiB");
    UntrackedWeakSnapshotBlock memory(context);
    const auto verifyNoProof = [&](std::size_t surface) {
        const auto address = memory.Address(surface);
        Require(!GM::Watched(address, UntrackedWeakSurfaceBytes), "compressed weak snapshot: fixture bytes entered the write watch");
        GM::BumpCollectEpoch();
        Require(GM::CollectWrites(address, UntrackedWeakSurfaceBytes) == 0, "compressed weak snapshot: fixture has a full write proof");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        Require(GM::CollectCommittedWrites(address, UntrackedWeakSurfaceBytes, committed) == 0 && committed.empty(), "compressed weak snapshot: fixture has a partial write proof");
        Require(GM::UntrackedHint(address, UntrackedWeakSurfaceBytes) == 0, "compressed weak snapshot: fixture has an untracked hint");
    };
    const auto lookup = [&](std::size_t surface) {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, LinearUntrackedDescriptor(memory.Address(surface)));
    };
    constexpr std::size_t tailBytes = 64u << 10u;
    constexpr std::size_t tailOffset = UntrackedWeakSurfaceBytes - tailBytes;
    constexpr std::size_t tailMarkerOffset = tailOffset + 256u;
    constexpr std::uint32_t initialTailMarker = 0x6a5b4c3du;
    constexpr std::uint32_t changedTailMarker = 0x1e2d3c4bu;
    for (std::size_t surface = 0; surface < 5u; ++surface) {
        auto state = 0x9e3779b9u ^ static_cast<std::uint32_t>(surface * 0x85ebca6bu);
        for (std::size_t offset = 0; offset < tailBytes; offset += sizeof(state)) {
            state ^= state << 13u;
            state ^= state >> 17u;
            state ^= state << 5u;
            std::memcpy(memory.Data() + surface * UntrackedWeakSurfaceBytes + tailOffset + offset, &state, sizeof(state));
        }
        std::memcpy(memory.Data() + surface * UntrackedWeakSurfaceBytes + tailMarkerOffset, &initialTailMarker, sizeof(initialTailMarker));
    }
    constexpr std::array<std::size_t, 3> boundaryBytes{(64u << 10u) - 4u, 64u << 10u, (64u << 10u) + 4u};
    constexpr std::array<std::uint32_t, 3> markers{0x143625efu, 0x9a7b6c5du, 0x0c1d2e3fu};
    for (std::size_t i = 0; i < boundaryBytes.size(); ++i) std::memcpy(memory.Data() + boundaryBytes[i], &markers[i], sizeof(markers[i]));
    std::array<std::shared_ptr<Texture>, 5> owners{};
    for (std::size_t surface = 0; surface < owners.size(); ++surface) {
        verifyNoProof(surface);
        owners[surface] = lookup(surface);
        Require(owners[surface] != nullptr && !owners[surface]->ViewsStorageImage(), "compressed weak snapshot: no owned sampled image");
        Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "compressed weak snapshot: target entered the strong cache");
        Require(FetchFirstTexel(context, *owners[surface], Side) == UntrackedInitial, "compressed weak snapshot: initial GPU texel differs");
        Require(FetchTexelAt(context, *owners[surface], Side, static_cast<std::uint32_t>(tailMarkerOffset / sizeof(std::uint32_t))) == initialTailMarker, "compressed weak snapshot: initial raw-tail marker differs");
    }
    for (std::size_t surface = 0; surface < owners.size(); ++surface) {
        const auto before = SampledTextureMemory();
        auto same = lookup(surface);
        Require(same == owners[surface] && SampledTextureMemory() == before, std::string("compressed weak snapshot: compressible surface did not reuse its live owner: ") + std::to_string(surface));
        Require(FetchFirstTexel(context, *same, Side) == UntrackedInitial, "compressed weak snapshot: reused GPU image has the wrong first texel");
    }
    for (std::size_t i = 0; i < boundaryBytes.size(); ++i) {
        const auto pixel = static_cast<std::uint32_t>(boundaryBytes[i] / sizeof(std::uint32_t));
        Require(FetchTexelAt(context, *owners[0], Side, pixel) == markers[i], "compressed weak snapshot: initial chunk-boundary marker differs");
    }
    auto oldBoundaryOwner = owners[0];
    for (std::size_t i = 0; i < boundaryBytes.size(); ++i) std::memcpy(memory.Data() + boundaryBytes[i], &markers[(i + 1u) % markers.size()], sizeof(markers[0]));
    std::memcpy(memory.Data() + tailMarkerOffset, &changedTailMarker, sizeof(changedTailMarker));
    verifyNoProof(0);
    auto boundaryChanged = lookup(0);
    Require(boundaryChanged != oldBoundaryOwner, "compressed weak snapshot: CPU writes across a 64 KiB chunk boundary returned a stale owner");
    owners[0] = boundaryChanged;
    Require(FetchFirstTexel(context, *boundaryChanged, Side) == UntrackedInitial, "compressed weak snapshot: boundary rewrite changed the first texel");
    for (std::size_t i = 0; i < boundaryBytes.size(); ++i) {
        const auto pixel = static_cast<std::uint32_t>(boundaryBytes[i] / sizeof(std::uint32_t));
        Require(FetchTexelAt(context, *boundaryChanged, Side, pixel) == markers[(i + 1u) % markers.size()], "compressed weak snapshot: changed chunk-boundary texel differs");
    }
    Require(FetchTexelAt(context, *boundaryChanged, Side, static_cast<std::uint32_t>(tailMarkerOffset / sizeof(std::uint32_t))) == changedTailMarker, "compressed weak snapshot: changed raw-tail marker differs");
    Require(lookup(0) == boundaryChanged, "compressed weak snapshot: boundary-updated bytes did not stabilize for reuse");
    auto oldTopologyOwner = owners[1];
    memory.Decommit(UntrackedWeakSurfaceBytes + 4096u, 4096u);
    verifyNoProof(1);
    auto decommitted = lookup(1);
    Require(decommitted != oldTopologyOwner, "compressed weak snapshot: page decommit with unchanged zero pixels returned a stale owner");
    owners[1] = decommitted;
    Require(FetchTexelAt(context, *decommitted, Side, 1024u) == 0u, "compressed weak snapshot: decommitted zero page sampled incorrectly");
    memory.Commit(UntrackedWeakSurfaceBytes + 4096u, 4096u);
    std::memset(memory.Data() + UntrackedWeakSurfaceBytes + 4096u, 0, 4096u);
    auto recommitted = lookup(1);
    Require(recommitted != decommitted, "compressed weak snapshot: identical-byte recommit reused a different topology proof");
    owners[1] = recommitted;
    Require(FetchTexelAt(context, *recommitted, Side, 1024u) == 0u, "compressed weak snapshot: recommitted zero page sampled incorrectly");
    Require(lookup(1) == recommitted, "compressed weak snapshot: recommitted topology did not stabilize");
    {
        Recorder recorder(context);
        recorder.Activate();
        auto copied = std::make_shared<Buffer>(context, sizeof(std::uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::fill(copied->Bytes().begin(), copied->Bytes().end(), std::byte{0xa6});
        const auto landed = std::make_shared<bool>(false);
        const auto commands = recorder.Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, copied->Handle(), 0, sizeof(std::uint32_t), UntrackedPendingWrite);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        const auto address = memory.Address(2);
        recorder.OnComplete([copied, landed, address] {
            copied->Invalidate();
            CheckSnapshotWords(copied->Bytes(), UntrackedPendingWrite, "compressed weak snapshot pending GPU output");
            GM::StoreOwnBytes(address, copied->Bytes());
            *landed = true;
        });
        recorder.Keep(copied);
        recorder.NotePendingWrite(address, sizeof(std::uint32_t));
        Require(Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "compressed weak snapshot: pending GPU write was not registered");
        const auto oldPendingOwner = owners[2];
        auto pending = lookup(2);
        Require(pending != oldPendingOwner, "compressed weak snapshot: pending GPU write returned a stale owner");
        Require(*landed && !Recorder::SnapshotWriteOverlaps(address, sizeof(std::uint32_t)), "compressed weak snapshot: lookup did not wait for the pending GPU writer");
        Require(*reinterpret_cast<const std::uint32_t*>(memory.Data() + 2u * UntrackedWeakSurfaceBytes) == UntrackedPendingWrite, "compressed weak snapshot: pending writer did not land guest bytes");
        auto stablePending = lookup(2);
        if (stablePending != pending) pending = stablePending;
        owners[2] = pending;
        Require(lookup(2) == pending, "compressed weak snapshot: post-writer bytes did not stabilize for reuse");
        Require(FetchFirstTexel(context, *pending, Side) == UntrackedPendingWrite, "compressed weak snapshot: post-writer GPU texel differs");
        recorder.Sync();
    }
    for (const auto surface : {3u, 4u}) {
        const auto before = SampledTextureMemory();
        Require(lookup(surface) == owners[surface] && SampledTextureMemory() == before, "compressed weak snapshot: live proof was lost after boundary or topology updates");
    }
    const auto made = owners.size();
    oldBoundaryOwner.reset();
    oldTopologyOwner.reset();
    decommitted.reset();
    boundaryChanged.reset();
    recommitted.reset();
    owners.fill(nullptr);
    fillerOwners.fill(nullptr);
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == initialMemory, "compressed weak snapshot: image owners did not retire after release");
    std::printf("compressed_untracked_weak_live=PASS images=%zu bytes=%zu chunk_bytes=%u\n", made, UntrackedWeakSurfaceBytes, 64u << 10u);
}

void incompressibleUntrackedSnapshotTests(const Context& context) {
    namespace GM = AgcDriver::GuestMemory;
    const auto* proofBudget = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_MIB");
    const auto* compression = std::getenv("APS5_LIVE_TEXTURE_SNAPSHOT_COMPRESS");
    Require(proofBudget != nullptr && std::string_view(proofBudget) == "1", "incompressible weak snapshot: fixture requires a 1 MiB byte-proof budget");
    Require(compression != nullptr && std::string_view(compression) == "1", "incompressible weak snapshot: fixture requires opt-in compression");
    if (!GM::WriteWatched()) return;
    ClearCachedTextures(context.device);
    const auto initialMemory = SampledTextureMemory();
    Block fillerMemory(context);
    std::array<std::shared_ptr<Texture>, 4> fillerOwners;
    Recorder::CountPresent();
    for (std::size_t i = 0; i < fillerOwners.size(); ++i) fillerOwners[i] = Sampled(context, fillerMemory.Surface(i));
    const auto cacheBaseline = TextureCacheUsage();
    Require(cacheBaseline.sampledEntries == fillerOwners.size() && cacheBaseline.sampledBytes == Budget, "incompressible weak snapshot: strong-cache fillers did not occupy exactly 1 MiB");
    UntrackedWeakSnapshotBlock memory(context);
    for (std::size_t surface = 0; surface < 5u; ++surface) {
        std::uint32_t state = 0x9e3779b9u ^ static_cast<std::uint32_t>(surface * 0x85ebca6bu);
        auto* bytes = memory.Data() + surface * UntrackedWeakSurfaceBytes;
        for (std::size_t offset = 0; offset < UntrackedWeakSurfaceBytes; offset += sizeof(state)) {
            state ^= state << 13u;
            state ^= state >> 17u;
            state ^= state << 5u;
            std::memcpy(bytes + offset, &state, sizeof(state));
        }
    }
    const auto verifyNoProof = [&](std::size_t surface) {
        const auto address = memory.Address(surface);
        Require(!GM::Watched(address, UntrackedWeakSurfaceBytes), "incompressible weak snapshot: fixture bytes entered the write watch");
        GM::BumpCollectEpoch();
        Require(GM::CollectWrites(address, UntrackedWeakSurfaceBytes) == 0, "incompressible weak snapshot: fixture has a full write proof");
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        Require(GM::CollectCommittedWrites(address, UntrackedWeakSurfaceBytes, committed) == 0 && committed.empty(), "incompressible weak snapshot: fixture has a partial write proof");
        Require(GM::UntrackedHint(address, UntrackedWeakSurfaceBytes) == 0, "incompressible weak snapshot: fixture has an untracked hint");
    };
    const auto lookup = [&](std::size_t surface) {
        GM::BumpCollectEpoch();
        return CachedSampledTexture(context, LinearUntrackedDescriptor(memory.Address(surface)));
    };
    std::array<std::shared_ptr<Texture>, 4> proofs{};
    std::vector<std::shared_ptr<Texture>> extra;
    for (std::size_t surface = 0; surface < proofs.size(); ++surface) {
        verifyNoProof(surface);
        proofs[surface] = lookup(surface);
        Require(proofs[surface] != nullptr && !proofs[surface]->ViewsStorageImage(), "incompressible weak snapshot: no owned sampled image");
        Require(TextureCacheUsage().sampledEntries == cacheBaseline.sampledEntries && TextureCacheUsage().sampledBytes == cacheBaseline.sampledBytes, "incompressible weak snapshot: target entered the strong cache");
        std::uint32_t expected = 0;
        std::memcpy(&expected, memory.Data() + surface * UntrackedWeakSurfaceBytes, sizeof(expected));
        Require(FetchFirstTexel(context, *proofs[surface], Side) == expected, "incompressible weak snapshot: captured GPU pixel differs");
    }
    for (std::size_t surface = 0; surface < proofs.size(); ++surface) {
        const auto before = SampledTextureMemory();
        Require(lookup(surface) == proofs[surface] && SampledTextureMemory() == before, "incompressible weak snapshot: one of four raw proofs did not reuse its owner");
    }
    verifyNoProof(4);
    const auto fresh = [&](const char* label) {
        const auto before = SampledTextureMemory();
        auto texture = lookup(4);
        Require(texture != nullptr && !texture->ViewsStorageImage(), std::string("incompressible weak snapshot: no image ") + label);
        Require(std::find(proofs.begin(), proofs.end(), texture) == proofs.end(), std::string("incompressible weak snapshot: stale proof owner returned ") + label);
        Require(SampledTextureMemory() == before + texture->AllocationBytes(), std::string("incompressible weak snapshot: missing fresh allocation ") + label);
        std::uint32_t expected = 0;
        std::memcpy(&expected, memory.Data() + 4u * UntrackedWeakSurfaceBytes, sizeof(expected));
        Require(FetchFirstTexel(context, *texture, Side) == expected, std::string("incompressible weak snapshot: wrong GPU pixel ") + label);
        extra.push_back(texture);
    };
    fresh("on first over-budget fallback");
    fresh("on repeated over-budget fallback");
    for (std::size_t surface = 0; surface < proofs.size(); ++surface) {
        const auto before = SampledTextureMemory();
        Require(lookup(surface) == proofs[surface] && SampledTextureMemory() == before, "incompressible weak snapshot: attempting raw fallback evicted a live proof");
    }
    const auto made = extra.size();
    extra.clear();
    proofs.fill(nullptr);
    fillerOwners.fill(nullptr);
    ClearCachedTextures(context.device);
    Require(SampledTextureMemory() == initialMemory, "incompressible weak snapshot: image owners did not retire after release");
    std::printf("incompressible_untracked_weak_live=PASS raw_slots=%zu over_budget_owners=%zu\n", proofs.size(), made);
}
#endif

bool resourceCacheTests(Context context, const std::string& mode) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    Block block(context);
    if (context.hostImportAlignment == 0 || HostImportFor(context, block.Surface(0), BlockBytes) == nullptr) {
        Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "resource cache tests require host imports");
        std::puts("skipped, resource cache tests require host imports");
        return false;
    }
    auto& cache = SharedResourceCache();
    cache.Clear();
    try {
        if (mode == "resource-clear") {
            auto storage = ImageResources(context, block, 0, 1, false);
            auto sampled = ImageResources(context, block, 0, 1, true);
            Require(sampled->SampledTextures().front()->ViewsStorageImage(), "the sampled resource is not a storage-image view");
            const std::weak_ptr<StorageTexture> image = StorageTexture::FindLive(block.Surface(0), SurfaceBytes);
            const std::weak_ptr<ShaderResources> storageSet = storage;
            const std::weak_ptr<ShaderResources> sampledSet = sampled;
            cache.Insert({1}, storage);
            cache.Insert({2}, sampled);
            Require(cache.Size() == 2, "the resource cache did not accept the live descriptor sets");
            sampled.reset();
            ClearCachedTextures(context.device);
            Require(cache.Find({1}) == nullptr && cache.Find({2}) == nullptr && cache.Size() == 0, "clearing storage images left their resources cached");
            Require(sampledSet.expired(), "cache clear kept the sampled descriptor set alive");
            Require(!image.expired() && !storageSet.expired(), "cache clear destroyed an active caller's resources");
            storage.reset();
            Require(image.expired() && storageSet.expired(), "cleared storage image outlived its last active resource");
        } else if (mode == "resource-oversized") {
            for (const bool sampledImages : {false, true}) {
                const std::weak_ptr<StorageTexture> early = CachedStorageSurface(context, Resource(block.Surface(0)));
                auto oversized = ImageResources(context, block, 0, static_cast<std::uint32_t>(Surfaces), sampledImages);
                const std::weak_ptr<ShaderResources> rejected = oversized;
                auto image = early.lock();
                Require(oversized->Reusable(), "the oversized descriptor set is not eligible for resource caching");
                Require(image != nullptr && !image->Cached(), "the oversized descriptor set did not evict an early image while building");
                if (sampledImages) Require(oversized->SampledTextures().size() == Surfaces && oversized->SampledTextures().front()->StorageSource() == image.get(), "the oversized sampled descriptor set does not retain the early storage image");
                std::vector<std::shared_ptr<ShaderResources>> displaced;
                cache.Insert({1}, oversized, &displaced);
                Require(cache.Size() == 0 && displaced.empty(), sampledImages ? "an oversized sampled descriptor set entered the resource cache" : "an oversized storage descriptor set entered the resource cache");
                image.reset();
                oversized.reset();
                Require(rejected.expired() && early.expired(), "the rejected oversized descriptor set retained an evicted storage image");
                std::puts(sampledImages ? "resource cache oversized sampled lifetime passed" : "resource cache oversized storage lifetime passed");
                ClearCachedTextures(context.device);
            }
        } else {
            auto storage = ImageResources(context, block, 0, 1, false);
            auto sampled = ImageResources(context, block, 0, 1, true);
            Require(storage->Reusable() && sampled->Reusable(), "the stale descriptor sets are not eligible for resource caching");
            Require(sampled->SampledTextures().front()->ViewsStorageImage(), "the sampled resource is not a storage-image view");
            auto image = StorageTexture::FindLive(block.Surface(0), SurfaceBytes);
            const std::weak_ptr<StorageTexture> staleImage = image;
            const std::weak_ptr<ShaderResources> storageSet = storage;
            const std::weak_ptr<ShaderResources> sampledSet = sampled;
            Require(image != nullptr && image->Cached(), "the resource has no cached storage image");
            for (std::size_t surface = 1; surface < Surfaces && image->Cached(); ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
            Require(!image->Cached(), "the test never evicted the oldest storage image");
            auto live = ImageResources(context, block, Surfaces - 1, 1, false);
            cache.Insert({1}, live);
            Require(cache.Find({1}) == live, "the resource cache rejected a live descriptor set");
            std::vector<std::shared_ptr<ShaderResources>> displaced;
            for (const auto& stale : {storage, sampled}) {
                cache.Insert({2}, stale, &displaced);
                Require(cache.Find({2}) == nullptr && cache.Size() == 1 && displaced.empty(), "an uncached image was admitted to the resource cache");
                cache.Insert({1}, stale, &displaced);
                Require(cache.Find({1}) == live && cache.Size() == 1 && displaced.empty(), "an uncached image replaced a live resource-cache entry");
            }
            image.reset();
            storage.reset();
            Require(storageSet.expired() && !sampledSet.expired() && !staleImage.expired(), "stale storage release did not preserve the sampled caller's image");
            std::puts("resource cache stale storage caller release passed");
            sampled.reset();
            Require(sampledSet.expired() && staleImage.expired(), "the rejected descriptor sets retained an evicted storage image");
            std::puts("resource cache stale sampled caller release passed");
        }
        if (mode == "resource-admission") recorderEvictionTests(context, block);
    } catch (...) {
        cache.Clear();
        throw;
    }
    cache.Clear();
    return true;
}

void viewTests(const Context& context) {
    if (context.hostImportAlignment == 0) {
        std::puts("host imports unavailable: views of storage images not tested");
        return;
    }
    Block block(context);
    if (HostImportFor(context, block.Surface(0), BlockBytes) == nullptr) {
        std::puts("host import of the texture block refused: views of storage images not tested");
        return;
    }
    const auto before = TextureCacheUsage();
    auto view = Sampled(context, block.Surface(0));
    Require(view->ViewsStorageImage(), "a sampled texture in host-imported memory is not a view of its storage image");
    const std::weak_ptr<Texture> viewed = view;
    const std::weak_ptr<StorageTexture> storage = view->SharedStorageSource();
    const auto after = TextureCacheUsage();
    Require(after.sampledEntries == before.sampledEntries + 1u, "the view did not enter the sampled texture cache");
    Require(after.sampledBytes == before.sampledBytes, "a view of a cached storage image counted " + std::to_string(after.sampledBytes - before.sampledBytes) + " bytes against the sampled texture cache");
    Require(after.storageBytes > before.storageBytes, "the storage image of the view is not counted by the storage image cache");
    Require(Sampled(context, block.Surface(0)) == view, "the second lookup of the surface did not hit its view");
    view.reset();
    for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
    Require(viewed.expired() && storage.expired(), "the sampled texture cache kept the view of an evicted storage image");
    const auto evicted = TextureCacheUsage();
    Require(evicted.sampledEntries == after.sampledEntries - 1u && evicted.sampledBytes == before.sampledBytes, "dropping the view of the evicted storage image changed other sampled entries");
    auto again = Sampled(context, block.Surface(0));
    Require(again->ViewsStorageImage() && again->StorageSource() != nullptr && StorageImageCached(context, again->StorageSource()), "the surface's next lookup does not view its new cached storage image");
    const std::weak_ptr<Texture> heldView = again;
    const std::weak_ptr<StorageTexture> heldStorage = again->SharedStorageSource();
    for (std::size_t surface = 1; surface < Surfaces; ++surface) CachedStorageSurface(context, Resource(block.Surface(surface)));
    Require(!StorageImageCached(context, again->StorageSource()), "the held view's source was not evicted");
    Require(!heldView.expired() && !heldStorage.expired(), "eviction destroyed an active caller's view or image");
    again.reset();
    Require(heldView.expired() && heldStorage.expired(), "the evicted image outlived its last caller without a sampled lookup");
}

ShaderRecompiler::RecompileResult SampledProgram(std::uint64_t address) {
    ShaderRecompiler::DescriptorBinding binding{};
    binding.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    binding.role = ShaderRecompiler::DescriptorRole::GuestImages;
    binding.descriptorSet = 0;
    binding.binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding + PlaceholderDimensionsPerClass + 2u;
    binding.count = 1;
    binding.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
    binding.imageWritten.assign(1, false);
    binding.imageSamplers.assign(1, 0u);
    const auto words = Descriptor(address);
    binding.guestDescriptor.assign(words.begin(), words.end());
    ShaderRecompiler::RecompileResult program;
    program.bindings.push_back(std::move(binding));
    return program;
}

void replacedEntryTests(Context context) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    WatchedSurface surface(context);
    ClearCachedTextures(context.device);
    const auto program = SampledProgram(surface.Address());
    const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto first = std::make_shared<ShaderResources>(context, shader);
    Require(first->Reusable(), "the sampled snapshot's descriptor set is not eligible for resource caching");
    Require(first->SampledTextures().size() == 1, "the sampled descriptor set binds no texture");
    const auto bound = first->SampledTextures().front();
    Require(!bound->ViewsStorageImage(), "the replaced-entry fixture is not a snapshot");
    Require(Sampled(context, surface.Address()) == bound, "the descriptor set does not bind the cache's texture");
    ShaderResources::ProofReport report{};
    Require(first->Revalidate(shader), "a template over an unchanged, still cached texture did not prove current");
    Require(first->Revalidate(shader, &report), "a template over an unchanged, still cached texture did not prove current twice");
    Require(report.path == ShaderResources::ProofPath::Fast, "the fixture's template did not take the fast proof");
    Require(first->SampledTextures().front() == bound, "the proof changed the bound texture");
    ClearCachedTextures(context.device);
    Require(TextureCacheUsage().sampledEntries == 0, "clearing the sampled cache left entries");
    Require(!first->Revalidate(shader, &report), "a template binding a texture whose cache entry left still proved current");
    const auto replacement = Sampled(context, surface.Address());
    Require(replacement != bound, "the lookup after the clear handed back the old texture");
    Require(!first->Revalidate(shader), "a template binding the replaced texture proved current against the new entry");
    auto rebuilt = std::make_shared<ShaderResources>(context, shader);
    Require(rebuilt->SampledTextures().front() == replacement, "the rebuilt descriptor set does not bind the cache's new texture");
    Require(rebuilt->Revalidate(shader), "the rebuilt template did not prove current over its cached texture");
    first.reset();
    rebuilt.reset();
    ClearCachedTextures(context.device);
    std::puts("template over a replaced sampled cache entry passed");
}

bool lookupChangedTests(Context context) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    if (context.hostImportAlignment == 0) {
        Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "the lookup-changed test requires host imports");
        std::puts("skipped, host imports are unavailable");
        return false;
    }
    WatchedSurface surface(context);
    ClearCachedTextures(context.device);
    const auto program = SampledProgram(surface.Address());
    const CompiledShader shader{ShaderRecompiler::ShaderStage::Compute, &program, 0};
    auto first = std::make_shared<ShaderResources>(context, shader);
    Require(first->Reusable() && first->SampledTextures().size() == 1, "the lookup-changed template is not reusable");
    const auto bound = first->SampledTextures().front();
    Require(!bound->ViewsStorageImage(), "the lookup-changed fixture is not a snapshot before the import");
    ShaderResources::ProofReport report{};
    Require(first->Revalidate(shader), "the lookup-changed template did not prove current");
    Require(first->Revalidate(shader, &report) && report.path == ShaderResources::ProofPath::Fast, "the lookup-changed template did not take the fast proof");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    Require(first->Revalidate(shader, &report), "the audit refused an unchanged template");
    Require(report.path == ShaderResources::ProofPath::Full, "a proof past the audit interval did not take the full walk");
    Require(first->SampledTextures().front() == bound, "the audit changed the bound texture");
    Require(first->Revalidate(shader, &report) && report.path == ShaderResources::ProofPath::Fast, "the proof right after an audit did not take the fast proof");
    if (HostImportFor(context, surface.Address(), SurfaceBytes) == nullptr) {
        Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "the host import of the watched surface was refused");
        std::puts("skipped, host imports are unavailable");
        return false;
    }
    Require(!bound->Released(), "the snapshot left the cache without a lookup");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    Require(!first->Revalidate(shader, &report), "a template binding a snapshot of a surface its lookup now views through a storage image still proved current past the audit interval");
    const auto view = Sampled(context, surface.Address());
    Require(view != bound && view->ViewsStorageImage(), "the lookup of the imported surface does not view its storage image");
    auto rebuilt = std::make_shared<ShaderResources>(context, shader);
    Require(rebuilt->SampledTextures().front() == view, "the rebuilt template does not bind the storage view");
    first.reset();
    rebuilt.reset();
    ClearCachedTextures(context.device);
    std::puts("template over a changed sampled lookup passed");
    return true;
}

void reportedBudgetTests(const Context& context) {
    if (context.memoryProperties2 == nullptr) {
        std::puts("VK_EXT_memory_budget unavailable: the sampled texture cache keeps a quarter of the device-local heap");
        return;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto budget = SampledTextureBudget(context.memory, &reported, 0);
    Require(budget >= (2048ull << 20u), "the sampled texture cache budget is below the 2 GiB floor");
    for (std::uint32_t heap = 0; heap < context.memory.memoryHeapCount; ++heap) {
        if ((context.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        std::printf("device-local heap %u: %llu MiB, VK_EXT_memory_budget budget %llu MiB, usage %llu MiB\n", heap, static_cast<unsigned long long>(context.memory.memoryHeaps[heap].size >> 20u), static_cast<unsigned long long>(reported.heapBudget[heap] >> 20u), static_cast<unsigned long long>(reported.heapUsage[heap] >> 20u));
    }
    std::printf("sampled texture cache budget %llu MiB, storage image cache budget %llu MiB\n", static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(TextureCacheBudget(context.memory) >> 20u));
}

void snapshotTests(const Context& context) {
    Block block(context);
    auto first = Sampled(context, block.Surface(0));
    Require(!first->ViewsStorageImage(), "a sampled texture outside host-imported memory is not a snapshot");
    const auto held = TextureCacheUsage().sampledBytes;
    Require(held == first->AllocationBytes(), "a snapshot counted " + std::to_string(held) + " bytes, not its image allocation of " + std::to_string(first->AllocationBytes()));
    const std::weak_ptr<Texture> oldest = first;
    first.reset();
    const auto capacity = static_cast<std::size_t>(Budget / held);
    Require(capacity >= 2u && capacity + 1u < Surfaces, "a snapshot of the 256 KiB surface counts " + std::to_string(held) + " bytes, which the test's 1 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Require(!oldest.expired() && TextureCacheUsage().sampledEntries == capacity, "the sampled texture cache did not keep " + std::to_string(capacity) + " snapshots within its budget");
    Sampled(context, block.Surface(capacity));
    Require(oldest.expired(), "the least recently used snapshot was not evicted past the budget");
    Require(TextureCacheUsage().sampledBytes <= Budget, "the sampled texture cache holds more than its budget");
    ClearCachedTextures(context.device);
    constexpr std::uint32_t TinySide = 16;
    const auto tiny = Sampled(context, block.Surface(0), TinySide);
    const auto guestBytes = DescribeSurface(Resource(block.Surface(0), TinySide)).guestBytes;
    if (tiny->AllocationBytes() >= guestBytes) {
        std::printf("the 16x16 snapshot's image takes %llu bytes, not less than its %llu guest bytes: counting the allocation alone not tested\n", static_cast<unsigned long long>(tiny->AllocationBytes()), static_cast<unsigned long long>(guestBytes));
        return;
    }
    Require(TextureCacheUsage().sampledBytes == tiny->AllocationBytes(), "a snapshot whose " + std::to_string(tiny->AllocationBytes()) + "-byte image is smaller than its " + std::to_string(guestBytes) + " guest bytes counted " + std::to_string(TextureCacheUsage().sampledBytes) + " bytes, not its allocation");
}

void recycleTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    const auto address = block.Surface(0);
    std::memset(reinterpret_cast<void*>(address), 0xa1, SurfaceBytes);
    const auto first = Sampled(context, address);
    Require(Sampled(context, address) == first, "an unchanged surface did not serve its snapshot");
    std::memset(reinterpret_cast<void*>(address), 0xb2, SurfaceBytes);
    const auto second = Sampled(context, address);
    Require(second != first, "a changed surface did not replace its snapshot");
    std::memset(reinterpret_cast<void*>(address), 0xa1, SurfaceBytes);
    const auto third = Sampled(context, address);
    Require(third != second, "the replaced snapshot's size-equal buffer was not reread from guest memory");
    Require(Sampled(context, address) == third, "an unchanged surface did not serve its remade snapshot");
    constexpr std::uint32_t BigSide = Side * 2;
    constexpr std::size_t BigBytes = std::size_t{BigSide} * BigSide * 4u;
    const WatchedSurface watchedFirst(context, BigBytes);
    const WatchedSurface watchedSecond(context, BigBytes);
    const auto bigFirst = Sampled(context, watchedFirst.Address(), BigSide);
    Require(TextureCacheUsage().spareEntries == 1 && TextureCacheUsage().spareBytes == BigBytes, "the first big snapshot did not park its dropped " + std::to_string(BigBytes) + "-byte buffer in the spare pool");
    const auto bigSecond = Sampled(context, watchedSecond.Address(), BigSide);
    Require(TextureCacheUsage().spareEntries == 1 && TextureCacheUsage().spareBytes == BigBytes, "the second big snapshot did not take the parked spare of its size");
}

void spareAgeTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    constexpr std::uint32_t BigSide = Side * 2;
    constexpr std::size_t BigBytes = std::size_t{BigSide} * BigSide * 4u;
    {
        const WatchedSurface watched(context, BigBytes);
        const auto first = Sampled(context, watched.Address(), BigSide);
        Require(TextureCacheUsage().spareEntries == 1 && TextureCacheUsage().spareBytes == BigBytes, "the big snapshot did not park its dropped buffer in the spare pool");
        for (std::uint64_t present = 0; present <= SpareSnapshotPresents; ++present) Recorder::CountPresent();
        const auto small = Sampled(context, block.Surface(0));
        Require(TextureCacheUsage().spareEntries == 0 && TextureCacheUsage().spareBytes == 0, "a spare idle for " + std::to_string(SpareSnapshotPresents + 1u) + " presents was not released");
    }
    ClearCachedTextures(context.device);
    {
        const WatchedSurface watchedFirst(context, BigBytes);
        const WatchedSurface watchedSecond(context, BigBytes);
        const auto second = Sampled(context, watchedFirst.Address(), BigSide);
        Require(TextureCacheUsage().spareEntries == 1 && TextureCacheUsage().spareBytes == BigBytes, "the second big snapshot did not park its dropped buffer in the spare pool");
        for (std::uint64_t present = 0; present < SpareSnapshotPresents; ++present) Recorder::CountPresent();
        const auto third = Sampled(context, watchedSecond.Address(), BigSide);
        Require(TextureCacheUsage().spareEntries == 1 && TextureCacheUsage().spareBytes == BigBytes, "a spare taken within " + std::to_string(SpareSnapshotPresents) + " presents was released instead of reused");
    }
}

void heldTests(const Context& context) {
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    const auto bytes = TextureCacheUsage().sampledBytes;
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require((capacity + 1u) * bytes <= Ceiling && capacity + 2u < Surfaces, "a snapshot of the 256 KiB surface counts " + std::to_string(bytes) + " bytes, which the test's ceiling does not fit as planned");
    std::vector<std::weak_ptr<Texture>> released;
    for (std::size_t surface = 1; surface < capacity; ++surface) released.push_back(Sampled(context, block.Surface(surface)));
    Sampled(context, block.Surface(capacity));
    Require(Sampled(context, block.Surface(0)) == held, "the sampled texture cache evicted a texture its caller still holds");
    Require(released.front().expired(), "the least recently used texture nobody holds was not evicted in place of the held one");
    held.reset();
    ClearCachedTextures(context.device);
    std::vector<std::shared_ptr<Texture>> all;
    for (std::size_t surface = 0; surface <= capacity; ++surface) all.push_back(Sampled(context, block.Surface(surface)));
    Require(TextureCacheUsage().sampledEntries == capacity + 1u && TextureCacheUsage().sampledBytes > Budget, "textures their callers hold left a cache over its budget");
    for (std::size_t surface = 0; surface < all.size(); ++surface) Require(Sampled(context, block.Surface(surface)) == all[surface], "surface " + std::to_string(surface) + " was made again while its texture was held");
    all.push_back(Sampled(context, block.Surface(capacity + 1u)));
    Require(TextureCacheUsage().sampledEntries == capacity + 1u && TextureCacheUsage().sampledBytes <= Ceiling, "held textures took the sampled texture cache past a quarter over its budget");
    for (std::size_t surface = 1; surface < all.size(); ++surface) Require(Sampled(context, block.Surface(surface)) == all[surface], "past the ceiling, surface " + std::to_string(surface) + " left in place of the least recently used held texture");
    all.clear();
    Sampled(context, block.Surface(capacity + 2u));
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "released textures were not evicted down to the budget");
}

void releasedTests(Context context) {
    PlaceholderImages placeholders(context);
    context.placeholderImages = &placeholders;
    TextureCache textures(context);
    context.textureCache = &textures;
    ClearCachedTextures(context.device);
    auto& cache = SharedResourceCache();
    cache.Clear();
    try {
        Block block(context);
        auto pinned = ImageResources(context, block, 0, 1, true);
        Require(pinned->SampledTextures().size() == 1 && !pinned->SampledTextures().front()->ViewsStorageImage(), "the pinned descriptor set does not hold a sampled snapshot");
        const std::weak_ptr<Texture> texture = pinned->SampledTextures().front();
        const std::weak_ptr<ShaderResources> pinnedSet = pinned;
        cache.Insert({1}, pinned);
        Require(cache.Find({1}) == pinned, "the resource cache rejected a descriptor set whose texture is cached");
        pinned.reset();
        ClearCachedTextures(context.device);
        Require(!texture.expired() && texture.lock()->Released(), "the sampled texture cache dropped a texture the resource cache holds without marking it released");
        auto live = ImageResources(context, block, 1, 1, true);
        std::vector<std::shared_ptr<ShaderResources>> displaced;
        cache.Insert({2}, live, &displaced);
        Require(cache.Find({1}) == nullptr && cache.Find({2}) == live && cache.Size() == 1, "the resource cache kept an entry holding a released texture");
        Require(displaced.size() == 1 && displaced.front() == pinnedSet.lock(), "the entry holding a released texture was not handed back to the caller");
        displaced.clear();
        Require(pinnedSet.expired() && texture.expired(), "the resource cache kept a released texture alive");
        auto stale = ImageResources(context, block, 2, 1, true);
        ClearCachedTextures(context.device);
        Require(stale->SampledTextures().front()->Released(), "a held texture the sampled cache dropped is not marked released");
        cache.Insert({3}, stale, &displaced);
        Require(cache.Find({3}) == nullptr, "a descriptor set holding a released texture entered the resource cache");
        Require(cache.Size() == 0 && displaced.size() == 1 && displaced.front() == live, "the live entry whose texture was released stayed in the resource cache");
        displaced.clear();
    } catch (...) {
        cache.Clear();
        throw;
    }
    cache.Clear();
    ClearCachedTextures(context.device);
    std::puts("resource cache released texture tests passed");
}

void rescanTests(const Context& context) {
    constexpr std::uint32_t SmallSide = 128;
    constexpr std::size_t SmallBytes = std::size_t{SmallSide} * SmallSide * 4u;
    Block block(context);
    const auto small = [&](std::size_t surface) { return Sampled(context, block.Surface(0) + surface * SmallBytes, SmallSide); };
    std::vector<std::shared_ptr<Texture>> all{small(0)};
    const auto bytes = TextureCacheUsage().sampledBytes;
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require((capacity + 3u) * bytes <= Ceiling && (capacity + 3u) * SmallBytes <= BlockBytes, "a snapshot of the 64 KiB surface counts " + std::to_string(bytes) + " bytes, which the test's ceiling does not fit as planned");
    for (std::size_t surface = 1; surface <= capacity; ++surface) all.push_back(small(surface));
    const std::weak_ptr<Texture> newest = all.back();
    all.pop_back();
    all.push_back(small(capacity + 1u));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    all.push_back(small(capacity + 2u));
    Require(newest.expired(), "a released texture behind held ones was still cached a second later");
    Require(TextureCacheUsage().sampledEntries == capacity + 2u, "the rescan left " + std::to_string(TextureCacheUsage().sampledEntries) + " entries, not the " + std::to_string(capacity + 2u) + " textures still held");
}

void viewEvictionTests(const Context& context) {
    if (context.hostImportAlignment == 0) return;
    Block snapshots(context);
    Block imported(context);
    if (HostImportFor(context, imported.Surface(0), BlockBytes) == nullptr) return;
    const std::weak_ptr<Texture> view = Sampled(context, imported.Surface(0));
    Require(!view.expired() && TextureCacheUsage().sampledBytes == 0, "the view of a cached storage image did not enter the sampled texture cache at 0 bytes");
    Sampled(context, snapshots.Surface(0));
    const auto snapshotBytes = TextureCacheUsage().sampledBytes;
    Require(snapshotBytes != 0, "the snapshot fixture unexpectedly became a view of a cached storage image");
    const auto capacity = static_cast<std::size_t>(Budget / snapshotBytes);
    for (std::size_t surface = 1; surface <= capacity; ++surface) Sampled(context, snapshots.Surface(surface));
    Require(!view.expired(), "the sampled texture cache evicted a view of a cached storage image, which frees no memory");
}

void frameTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    const auto frame = capacity + 2u;
    Require(capacity >= 2u && 2u * frame <= Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> made(frame);
    for (std::size_t surface = 0; surface < frame; ++surface) made[surface] = Sampled(context, block.Surface(surface));
    Require(TextureCacheUsage().sampledEntries == capacity, "a frame of " + std::to_string(frame) + " textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " cached, not the " + std::to_string(capacity) + " the budget holds");
    for (std::size_t surface = capacity; surface < frame; ++surface) Require(made[surface].expired(), "surface " + std::to_string(surface) + ", used once in a frame whose textures fill the cache, was cached in place of an earlier one");
    Recorder::CountPresent();
    std::size_t hits = 0;
    for (std::size_t surface = 0; surface < frame; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        hits += again == made[surface].lock();
    }
    Require(hits == capacity, "the next frame over the same " + std::to_string(frame) + " textures found " + std::to_string(hits) + " of them cached, not the " + std::to_string(capacity) + " the cache kept");
    auto refused = Sampled(context, block.Surface(frame - 1u));
    Require(Sampled(context, block.Surface(frame - 1u)) != refused, "a texture used a second time in the frame was cached in place of one of the frame's textures");
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "the texture used a second time left the cache over its budget");
    refused.reset();
    Recorder::CountPresent();
    std::vector<std::weak_ptr<Texture>> moved(frame);
    for (std::size_t surface = 0; surface < frame; ++surface) moved[surface] = Sampled(context, block.Surface(frame + surface));
    for (std::size_t surface = 0; surface < frame; ++surface) Require(moved[surface].expired(), "surface " + std::to_string(frame + surface) + ", first used in the frame after the cached textures were, was cached in place of one of them");
    Require(TextureCacheUsage().sampledEntries == capacity, "a frame over other textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " of the previous frame's textures cached");
    Recorder::CountPresent();
    for (std::size_t surface = 0; surface < frame; ++surface) moved[surface] = Sampled(context, block.Surface(frame + surface));
    for (std::size_t surface = 0; surface < capacity; ++surface) Require(!moved[surface].expired(), "surface " + std::to_string(frame + surface) + " of a second frame over other textures did not replace the textures last used two frames before");
    Require(TextureCacheUsage().sampledEntries == capacity, "a second frame over other textures left " + std::to_string(TextureCacheUsage().sampledEntries) + " entries cached");
}

void storageBudgetIsolationTests(const Context& context) {
    ClearCachedTextures(context.device);
    Block block(context);
    std::vector<std::shared_ptr<StorageTexture>> images;
    std::uint64_t allocated = 0;
    for (std::size_t surface = 0; surface < Surfaces; ++surface) {
        auto image = CachedStorageSurface(context, Resource(block.Surface(surface)));
        allocated += image->BudgetBytes();
        images.push_back(std::move(image));
        Require(TextureCacheUsage().storageBytes <= Budget, "sampled-only budget: storage cache admission passed the legacy 1 MiB cap");
    }
    Require(allocated > Budget, "sampled-only budget: the storage fixture did not request more than 1 MiB of native images");
    const auto usage = TextureCacheUsage();
    Require(usage.storageBytes != 0 && usage.storageBytes <= Budget, "sampled-only budget: the storage cache did not remain within the legacy 1 MiB cap after over-budget admission");
    auto view = Sampled(context, block.Surface(Surfaces - 1u));
    Require(view->ViewsStorageImage() && view->StorageSource() == images.back().get() && StorageImageCached(context, images.back().get()), "sampled-only budget: the final storage image did not retain its sampled view");
    Require(FetchFirstTexel(context, *view, Side) == 0x5a5a5a5au, "sampled-only budget: storage image readback changed during cache eviction");
    Require(TextureCacheUsage().storageBytes <= Budget, "sampled-only budget: storage cache readback exceeded the legacy 1 MiB cap");
}

void sampledBudgetIsolationTests(const Context& context) {
    constexpr std::uint64_t sampledLimit = 2u * Budget;
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    Require(StorageTextureCacheBudget(context) == Budget, "sampled-only budget: the legacy 1 MiB limit did not remain on the storage cache");
    Block block(context);
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(sampledLimit / bytes);
    const auto frame = capacity + 2u;
    Require(capacity > Budget / bytes && frame <= Surfaces, "sampled-only budget: the 2 MiB test limit does not fit the fixture");
    std::vector<std::weak_ptr<Texture>> made(frame);
    for (std::size_t surface = 0; surface < frame; ++surface) made[surface] = Sampled(context, block.Surface(surface));
    const auto retained = TextureCacheUsage();
    Require(retained.sampledEntries == capacity && retained.sampledBytes > Budget && retained.sampledBytes <= sampledLimit, "sampled-only budget: sampled entries did not retain more than the legacy 1 MiB cap within the 2 MiB cap");
    Require(SampledTextureCacheBudget(context) == sampledLimit, "sampled-only budget: the sampled-specific setting was not applied");
    OverrideCacheBudgetForTest(Budget);
    Require(SampledTextureCacheBudget(context) == Budget, "sampled-only budget: the test override lost priority over the sampled-specific setting");
    OverrideCacheBudgetForTest(0);
    for (std::size_t surface = capacity; surface < frame; ++surface) Require(made[surface].expired(), "sampled-only budget: a first-use texture displaced an older cache entry");
    Recorder::CountPresent();
    std::size_t hits = 0;
    for (std::size_t surface = 0; surface < frame; ++surface) hits += Sampled(context, block.Surface(surface)) == made[surface].lock();
    Require(hits == capacity, "sampled-only budget: the next frame did not reuse exactly the retained entries");
    Recorder::CountPresent();
    auto firstUse = Sampled(context, block.Surface(capacity));
    Require(firstUse.use_count() == 1, "sampled-only budget: a first-use texture entered the strong cache");
    const auto admitted = Sampled(context, block.Surface(capacity));
    Require(Sampled(context, block.Surface(capacity)) == admitted, "sampled-only budget: a second-use texture did not enter the strong cache");
    Require(made[0].expired(), "sampled-only budget: admission did not evict the least-recently-used texture");
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= sampledLimit, "sampled-only budget: LRU admission exceeded the sampled limit");
    const auto beforeWrite = Sampled(context, block.Surface(1));
    Require(beforeWrite == made[1].lock(), "sampled-only budget: the CPU-write target was not still cached");
    constexpr std::uint32_t changed = 0xb2b2b2b2u;
    std::memset(reinterpret_cast<void*>(block.Surface(1)), 0xb2, SurfaceBytes);
    const auto afterWrite = Sampled(context, block.Surface(1));
    Require(afterWrite != beforeWrite, "sampled-only budget: a CPU write reused a stale sampled image");
    Require(FetchFirstTexel(context, *afterWrite, Side) == changed, "sampled-only budget: the refreshed image does not contain the CPU-written pixel");
    ClearCachedTextures(context.device);
    Block heldBlock(context);
    Recorder::CountPresent();
    std::vector<std::shared_ptr<Texture>> held(capacity);
    for (std::size_t surface = 0; surface < capacity; ++surface) held[surface] = Sampled(context, heldBlock.Surface(surface));
    Require(TextureCacheUsage().sampledBytes == sampledLimit, "sampled-only budget: held entries did not fill the 2 MiB budget");
    Recorder::CountPresent();
    auto firstUseHeld = Sampled(context, heldBlock.Surface(capacity));
    const auto firstAdmitted = Sampled(context, heldBlock.Surface(capacity));
    Require(Sampled(context, heldBlock.Surface(capacity)) == firstAdmitted, "sampled-only budget: a repeated first texture was not admitted around held entries");
    auto secondUseHeld = Sampled(context, heldBlock.Surface(capacity + 1u));
    const auto secondAdmitted = Sampled(context, heldBlock.Surface(capacity + 1u));
    Require(Sampled(context, heldBlock.Surface(capacity + 1u)) == secondAdmitted, "sampled-only budget: a second texture was not admitted around held entries");
    const auto heldUsage = TextureCacheUsage();
    Require(heldUsage.sampledBytes <= sampledLimit + sampledLimit / 4u, "sampled-only budget: held entries did not stay within the defined 25 percent cache ceiling");
    storageBudgetIsolationTests(context);
}

void sampledBudgetFallbackTests(const Context& context) {
    ClearCachedTextures(context.device);
    Require(StorageTextureCacheBudget(context) == Budget, "sampled-only fallback: the legacy storage budget is not 1 MiB");
    Require(SampledTextureCacheBudget(context) == Budget, "sampled-only fallback: zero, invalid, or unset did not inherit the legacy 1 MiB budget");
}

void scanTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    const auto working = 2u * capacity;
    Require(capacity >= 2u && working <= Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    constexpr std::size_t Presents = 6;
    constexpr std::size_t Uses = 2;
    std::vector<std::weak_ptr<Texture>> last(working);
    std::size_t hits = 0;
    std::size_t lookups = 0;
    for (std::size_t present = 0; present < Presents; ++present) {
        Recorder::CountPresent();
        for (std::size_t surface = 0; surface < working; ++surface) {
            for (std::size_t use = 0; use < Uses; ++use) {
                const auto texture = Sampled(context, block.Surface(surface));
                if (present != 0u) {
                    hits += texture == last[surface].lock();
                    ++lookups;
                }
                last[surface] = texture;
            }
        }
    }
    Require(TextureCacheUsage().sampledBytes <= Budget, "the sampled texture cache holds more than its budget after a scan");
    const auto wanted = (Presents - 1u) * Uses * capacity / 2u;
    Require(hits >= wanted, "a cyclic scan using each of " + std::to_string(working) + " textures twice in a row, twice the " + std::to_string(capacity) + " the budget holds, found " + std::to_string(hits) + " of " + std::to_string(lookups) + " lookups cached, fewer than " + std::to_string(wanted));
}

void shrinkTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    OverrideCacheBudgetForTest(2u * Budget);
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>((2u * Budget) / bytes);
    Require(capacity >= 4u && capacity + 1u <= Surfaces, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 2 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Require(TextureCacheUsage().sampledEntries == capacity, "the 2 MiB budget did not keep " + std::to_string(capacity) + " snapshots");
    Recorder::CountPresent();
    OverrideCacheBudgetForTest(Budget);
    for (std::size_t surface = 0; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Sampled(context, block.Surface(capacity));
    Require(TextureCacheUsage().sampledBytes <= Ceiling, "after the budget fell to 1 MiB, one insert left " + std::to_string(TextureCacheUsage().sampledBytes) + " bytes cached with every entry in use this frame, past a quarter over the budget");
    OverrideCacheBudgetForTest(0);
}

void secondUseTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    for (std::size_t surface = 1; surface < capacity; ++surface) Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    auto first = Sampled(context, block.Surface(capacity));
    Require(TextureCacheUsage().sampledEntries == capacity && first.use_count() == 1, "the first use in a frame of a texture the cache does not hold was cached in place of one the last frame used");
    const auto second = Sampled(context, block.Surface(capacity));
    Require(Sampled(context, block.Surface(capacity)) == second, "a texture used a second time in a frame whose least recently used entry is from the last frame was not cached");
    Require(TextureCacheUsage().sampledEntries == capacity && TextureCacheUsage().sampledBytes <= Budget, "the texture admitted on its second use left the cache over its budget");
}

void earlyMissTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> cached(capacity);
    for (std::size_t surface = 0; surface < capacity; ++surface) cached[surface] = Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    for (std::size_t surface = 0; surface < capacity; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        Require(again == cached[surface].lock(), "surface " + std::to_string(surface) + " missed in a frame over the textures the cache holds");
    }
    Recorder::CountPresent();
    Sampled(context, block.Surface(capacity));
    std::size_t hits = 0;
    for (std::size_t surface = 0; surface < capacity; ++surface) {
        const auto again = Sampled(context, block.Surface(surface));
        hits += again == cached[surface].lock();
    }
    Require(hits == capacity, "a frame that first uses a new texture before its " + std::to_string(capacity) + " cached ones found " + std::to_string(hits) + " of them cached");
}

constexpr std::uint64_t KeyHashBasis = 14695981039346656037ull;
constexpr std::uint64_t KeyHashPrime = 1099511628211ull;

std::array<std::array<std::uint32_t, 8>, 2> CollidingDescriptors(VkDevice device, std::uint64_t address) {
    const auto words = Descriptor(address);
    auto prefix = KeyHashBasis ^ reinterpret_cast<std::uintptr_t>(device);
    for (std::size_t word = 0; word < 6; ++word) prefix = (prefix ^ words[word]) * KeyHashPrime;
    std::unordered_map<std::uint32_t, std::uint32_t> word6ByHighHalf;
    for (std::uint32_t candidate = 0; candidate < 0x200u; ++candidate) {
        const auto word6 = ((candidate >> 1u) << 24u) | (candidate & 1u);
        const auto hash = (prefix ^ word6) * KeyHashPrime;
        const auto [match, added] = word6ByHighHalf.emplace(static_cast<std::uint32_t>(hash >> 32u), word6);
        if (added) continue;
        auto first = words;
        auto second = words;
        first[6] = match->second;
        second[6] = word6;
        second[7] = static_cast<std::uint32_t>(hash ^ ((prefix ^ match->second) * KeyHashPrime));
        return {first, second};
    }
    throw std::runtime_error("no two descriptors of the surface that differ only in word 6's statistics counter and the unused meta address collide in TextureKeyHash");
}

void collidingKeyTests(const Context& context) {
    Block block(context);
    ClearCachedTextures(context.device);
    Recorder::CountPresent();
    const auto bytes = Sampled(context, block.Surface(0))->AllocationBytes();
    const auto capacity = static_cast<std::size_t>(Budget / bytes);
    Require(capacity >= 2u && capacity < Surfaces && TextureCacheUsage().sampledEntries == 1u, "a snapshot of the 256 KiB surface holds " + std::to_string(bytes) + " bytes, which the test's 1 MiB budget does not fit as planned");
    std::vector<std::weak_ptr<Texture>> cached(capacity);
    for (std::size_t surface = 0; surface < capacity; ++surface) cached[surface] = Sampled(context, block.Surface(surface));
    Recorder::CountPresent();
    const auto [first, second] = CollidingDescriptors(context.device, block.Surface(capacity));
    CachedSampledTexture(context, first);
    CachedSampledTexture(context, second);
    std::size_t kept = 0;
    for (const auto& texture : cached) kept += !texture.expired();
    Require(kept == capacity && TextureCacheUsage().sampledEntries == capacity, "the first use of a descriptor whose cache key hash equals that of another descriptor first used in the frame left " + std::to_string(kept) + " of the " + std::to_string(capacity) + " textures the last frame used cached");
}

constexpr std::uint64_t ReportedBudget = 64ull << 30u;
constexpr std::uint64_t ReportedUsage = 12ull << 30u;

void VKAPI_PTR ReportBudget(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    for (std::uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        reported->heapBudget[heap] = ReportedBudget;
        reported->heapUsage[heap] = ReportedUsage;
    }
}

std::uint64_t ExpectedBudget(const Context& context) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
    ReportBudget(context.physical, &properties);
    return SampledTextureBudget(context.memory, &reported, SampledTextureMemory() + TextureCacheUsage().storageBytes, TextureCacheUsage().storageBytes);
}

std::uint64_t ReadBudget(const Context& context) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    return SampledTextureCacheBudget(context);
}

constexpr std::uint64_t PressedUsage = ReportedBudget - 9ull * (1ull << 30u);

void VKAPI_PTR ReportPressure(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties2* properties) {
    auto* reported = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
    for (std::uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        reported->heapBudget[heap] = ReportedBudget;
        reported->heapUsage[heap] = PressedUsage;
    }
}

std::uint64_t ReadStorageBudget(const Context& context) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    return StorageTextureCacheBudget(context);
}

void memoryTests(Context context) {
    context.memoryProperties2 = &ReportBudget;
    const auto before = SampledTextureMemory();
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    Require(!held->ViewsStorageImage() && held->AllocationBytes() != 0, "the test texture is not a snapshot with an image of its own");
    Require(SampledTextureMemory() == before + held->AllocationBytes(), "the sampled texture memory grew by " + std::to_string(SampledTextureMemory() - before) + " bytes, not by the snapshot's allocation of " + std::to_string(held->AllocationBytes()));
    Require(ReadBudget(context) == ExpectedBudget(context), "the sampled texture cache budget does not count its cached texture as the cache's own memory");
    ClearCachedTextures(context.device);
    Require(TextureCacheUsage().sampledEntries == 0 && SampledTextureMemory() == before + held->AllocationBytes(), "a texture its caller still holds stopped counting as sampled texture memory when it left the cache");
    const auto budget = ReadBudget(context);
    Require(budget == ExpectedBudget(context), "the sampled texture cache budget counts a texture its caller still holds as memory in use outside the texture caches");
    std::printf("reported heap budget %llu MiB with %llu MiB in use: sampled texture cache budget %llu MiB with a %llu KiB texture held outside the cache\n", static_cast<unsigned long long>(ReportedBudget >> 20u), static_cast<unsigned long long>(ReportedUsage >> 20u), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(held->AllocationBytes() >> 10u));
    held.reset();
    Require(SampledTextureMemory() == before, "a released texture still counts as sampled texture memory");
}

void storageMemoryTests(Context context) {
    constexpr std::uint64_t GiB = 1ull << 30u;
    context.memoryProperties2 = &ReportBudget;
    Block block(context);
    auto held = Sampled(context, block.Surface(0));
    const auto heap = LargestDeviceLocalHeap(context.memory);
    Require(heap.has_value(), "the device has no device-local heap");
    context.memory.memoryHeaps[*heap].size = std::max<VkDeviceSize>(context.memory.memoryHeaps[*heap].size, ReportedBudget);
    const auto half = std::min<std::uint64_t>(context.memory.memoryHeaps[*heap].size / 2u, ReportedBudget / 2u);
    Require(ReadStorageBudget(context) == half, "52 GiB free of a 64 GiB budget does not leave the storage image cache half of the heap or of the budget");
    context.memoryProperties2 = &ReportPressure;
    const auto pressed = ReadStorageBudget(context);
    const auto expected = GiB + TextureCacheUsage().storageBytes;
    Require(pressed == expected, "55 GiB in use of a 64 GiB budget leaves the storage image cache " + std::to_string(pressed >> 20u) + " MiB, not the " + std::to_string(expected >> 20u) + " MiB left after the use outside the texture caches, the sampled textures and 8 GiB of headroom");
    std::printf("reported heap budget %llu MiB with %llu MiB in use: storage image cache budget %llu MiB\n", static_cast<unsigned long long>(ReportedBudget >> 20u), static_cast<unsigned long long>(PressedUsage >> 20u), static_cast<unsigned long long>(pressed >> 20u));
}

struct RecoveryProbe {
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    VkDevice expectedDevice = VK_NULL_HANDLE;
    std::uint64_t released = 0;
    unsigned requests = 0;
    unsigned failures = 0;
    bool pressure = false;
};

RecoveryProbe recovery;

template <typename Device>
VkDeviceSize ReleaseCachesForTest(VkDeviceSize bytes, bool newImage, Device device) {
    if constexpr (requires { ReleaseTextureCaches(bytes, newImage, device); }) return ReleaseTextureCaches(bytes, newImage, device);
    else return ReleaseTextureCaches(bytes, newImage);
}

template <typename Release>
VkDeviceSize ReclaimForTestImpl(VkDeviceSize bytes, bool newImage, Release&& release) {
    Require(newImage, "an image allocation did not identify itself to the recovery callback");
    ++recovery.requests;
    Require(release(bytes, false) == 0, "a buffer allocation was allowed to reclaim the cache held by image creation");
    const auto released = release(bytes, newImage);
    recovery.released += released;
    return released;
}

VkDeviceSize ReclaimForTest(VkDeviceSize bytes, bool newImage) {
    return ReclaimForTestImpl(bytes, newImage, [](VkDeviceSize request, bool image) { return ReleaseTextureCaches(request, image); });
}

VkDeviceSize ReclaimForTest(VkDeviceSize bytes, bool newImage, VkDevice device) {
    Require(device == recovery.expectedDevice, "the allocation-recovery callback received a different device");
    return ReclaimForTestImpl(bytes, newImage, [device](VkDeviceSize request, bool image) { return ReleaseCachesForTest(request, image, device); });
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateForTest(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) {
    if (recovery.pressure && recovery.released == 0) {
        ++recovery.failures;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return reinterpret_cast<PFN_vkAllocateMemory>(recovery.deviceProc(device, "vkAllocateMemory"))(device, info, allocator, memory);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL RecoveryProc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkAllocateMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(AllocateForTest);
    return recovery.deviceProc(device, name);
}

void recoveryTests(Context context, const std::string& mode) {
    recovery = RecoveryProbe{};
    recovery.deviceProc = context.deviceProc;
    recovery.expectedDevice = context.device;
    context.deviceProc = RecoveryProc;
    context.cacheReclaim = ReclaimForTest;
    context.bufferPool.reset();
    Block block(context);
    const bool storage = mode == "recover-storage";
    if (storage) {
        auto oldest = CachedStorageSurface(context, Resource(block.Surface(0)));
        const auto allocated = oldest->AllocationBytes();
        std::weak_ptr<StorageTexture> released = oldest;
        oldest.reset();
        auto pending = CachedStorageSurface(context, Resource(block.Surface(1)));
        pending->MarkDirty();
        recovery.pressure = true;
        auto made = CachedStorageSurface(context, Resource(block.Surface(2)));
        recovery.pressure = false;
        Require(made != nullptr && released.expired(), "storage allocation recovery did not release its own oldest clean image");
        Require(pending->Dirty() && StorageImageCached(context, pending.get()), "storage allocation recovery changed an image with pending results");
        Require(recovery.released == allocated, "storage allocation recovery counted memory still held by its caller");
        pending->Flush();
        Require(!pending->Dirty(), "pending storage results could not be stored after memory recovery");
    } else if (mode == "recover-sampled") {
        auto oldest = Sampled(context, block.Surface(0));
        Require(!oldest->ViewsStorageImage(), "the sampled recovery fixture needs a snapshot allocation");
        const auto allocated = oldest->AllocationBytes();
        std::weak_ptr<Texture> released = oldest;
        oldest.reset();
        auto held = Sampled(context, block.Surface(1));
        recovery.pressure = true;
        auto made = Sampled(context, block.Surface(2));
        recovery.pressure = false;
        Require(made != nullptr && released.expired(), "sampled allocation recovery did not release its own oldest unused texture");
        Require(Sampled(context, block.Surface(1)) == held, "sampled allocation recovery removed a texture still held by its caller");
        Require(recovery.released == allocated, "sampled allocation recovery counted memory still held by its caller");
    } else {
        const bool srgb = mode == "recover-srgb";
        if (srgb) {
            context.srgbDecodeFormats = SrgbDecodeFormats(context.formatProperties, context.physical);
            context.srgbCopyFormats = SrgbCopyFormats(context.formatProperties, context.physical, context.srgbDecodeFormats);
            Require(SrgbExpandsToRgba8(context, 128u), "the device cannot serve the sRGB expansion copy fixture");
        }
        auto oldest = Sampled(context, block.Surface(0));
        Require(!oldest->ViewsStorageImage(), "the sampled recovery fixture needs a snapshot allocation");
        const auto allocated = oldest->AllocationBytes();
        std::weak_ptr<Texture> released = oldest;
        oldest.reset();
        auto held = Sampled(context, block.Surface(1));
        const auto side = srgb ? 1024u : 512u;
        const auto format = srgb ? 128u : 43u;
        const auto sourceWords = Descriptor(block.Surface(4), side, format);
        const auto sourceResource = DecodeTextureResource(sourceWords);
        auto source = CachedStorageSurface(context, sourceResource);
        source->MarkDirty();
        if (srgb) {
            Require(Texture::CanSrgbExpandFrom(context, *source, sourceResource), "the storage surface cannot serve an sRGB expansion copy");
        } else {
            Require(Texture::CanRepackCopyFrom(*source, sourceResource), "the storage surface cannot serve a repacked copy");
        }
        recovery.pressure = true;
        auto made = CachedSampledTexture(context, sourceWords);
        recovery.pressure = false;
        Require(made != nullptr && made->SharedStorageSource() == source, "sampled copy allocation recovery did not serve the pending storage image");
        Require(released.expired(), "sampled copy allocation recovery did not release its own oldest unused texture");
        Require(source->Dirty() && StorageImageCached(context, source.get()), "sampled copy allocation recovery changed or evicted an image with pending results");
        Require(!held->Released() && held.use_count() == 2, "sampled copy allocation recovery removed a texture still held by its caller");
        const auto refreshed = Sampled(context, block.Surface(1));
        Require(refreshed != nullptr && (refreshed == held || refreshed->ViewsStorageImage()), "sampled copy cache refresh did not serve the surface");
        Require(recovery.released == allocated, "sampled copy allocation recovery counted memory still held by its caller");
        source->Flush();
        Require(!source->Dirty(), "pending storage results could not be stored after memory recovery");
    }
    Require(recovery.requests == 1 && recovery.failures >= 1 && recovery.failures <= 2, "image allocation recovery was skipped or retried without a bound");
    const char* label = storage ? "storage" : mode == "recover-srgb" ? "sRGB" : mode == "recover-repack" ? "repacked" : "sampled";
    std::printf("%s image allocation recovered after %u injected failures: %llu bytes released\n", label, recovery.failures, static_cast<unsigned long long>(recovery.released));
}

}

int main(int argc, char** argv) {
    try {
        const bool memory = argc > 1 && std::string(argv[1]) == "memory";
        const std::string mode = argc > 1 ? argv[1] : "";
        const bool sampledBudgetMode = mode == "sampled-budget-separated" || mode == "sampled-budget-fallback";
        if (mode == "device-selection-cpu") {
            deviceSelectionTests();
            return 0;
        }
        const bool pendingSnapshot = mode == "snapshot-pending-write" || mode == "snapshot-pending-write-disjoint";
        const bool partialLive = mode == "snapshot-partial-weak-live";
        const bool untrackedLive = mode == "snapshot-untracked-weak-live";
        const bool fileRawWeakLive = mode == "snapshot-file-raw-weak-live";
        const bool fileRawNativeRetention = mode == "snapshot-file-raw-native-retention";
        const bool fileRawNativeArenaPressure = mode == "snapshot-file-raw-native-arena-pressure";
        const bool untrackedBudget = mode == "snapshot-untracked-weak-budget";
        const bool compressedWeak = mode == "snapshot-compressed-weak-live";
        const bool incompressibleWeak = mode == "snapshot-incompressible-weak-live";
        const bool resources = mode == "resource-admission" || mode == "resource-oversized" || mode == "resource-clear";
        const bool disableStorageImageMultisample = mode == "without-storage-image-multisample";
        const bool recover = mode == "recover-sampled" || mode == "recover-storage" || mode == "recover-srgb" || mode == "recover-repack";
#ifdef _WIN32
        if (!memory) _putenv_s("APS5_TEXTURE_CACHE_MIB", recover ? "16" : "1");
        if (!sampledBudgetMode) _putenv_s("APS5_SAMPLED_TEXTURE_CACHE_MIB", "");
        if (std::getenv("APS5_TEMPLATE_AUDIT_MS") == nullptr) _putenv_s("APS5_TEMPLATE_AUDIT_MS", "250");
#else
        if (!memory) setenv("APS5_TEXTURE_CACHE_MIB", recover ? "16" : "1", 1);
        if (!sampledBudgetMode) unsetenv("APS5_SAMPLED_TEXTURE_CACHE_MIB");
        setenv("APS5_TEMPLATE_AUDIT_MS", "250", 0);
#endif
        std::unique_ptr<Device> device;
        try {
            device = std::make_unique<Device>(!disableStorageImageMultisample);
        } catch (const std::exception& error) {
            const char* requestedName = std::getenv("ANYPS5_GPU");
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr || (requestedName != nullptr && *requestedName != '\0')) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        if (mode == "device-selection") {
            device->VerifySelection();
            return 0;
        }
        auto context = device->GetContext();
        if (disableStorageImageMultisample) {
            PlaceholderImages placeholders(context);
            for (std::uint32_t heap = ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
                 heap < ShaderRecompiler::RuntimeAbi::FirstImageBinding + ShaderRecompiler::RuntimeAbi::ImageBindingCount; ++heap) {
                const auto shape = PlaceholderImageShapeOf(heap);
                if (shape.samples == VK_SAMPLE_COUNT_1_BIT) continue;
                bool refused = false;
                try {
                    placeholders.View(heap);
                } catch (const std::runtime_error&) {
                    refused = true;
                }
                Require(refused, "a device without storage-image MSAA returned a placeholder for typed multisample storage heap " + std::to_string(heap));
            }
        }
        TextureDetiler detiler(context);
        context.detiler = &detiler;
        if (mode == "snapshot-packed-upload" || mode == "snapshot-unpacked-upload" || mode == "snapshot-packed-bc-upload" || mode == "snapshot-unpacked-bc-upload") {
#ifdef _WIN32
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            packedSnapshotTests(context, mode == "snapshot-packed-upload" || mode == "snapshot-packed-bc-upload", mode == "snapshot-packed-bc-upload" || mode == "snapshot-unpacked-bc-upload");
            return 0;
#else
            throw std::runtime_error("packed snapshot upload fixture requires Windows");
#endif
        }
        if (mode == "snapshot-direct-upload" || mode == "snapshot-copy-upload" || mode == "snapshot-direct-bc-upload" || mode == "snapshot-copy-bc-upload") {
#ifdef _WIN32
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            directSnapshotTests(context, mode == "snapshot-direct-upload" || mode == "snapshot-direct-bc-upload", mode == "snapshot-direct-bc-upload" || mode == "snapshot-copy-bc-upload");
            return 0;
#else
            throw std::runtime_error("direct snapshot upload fixture requires Windows");
#endif
        }
        if (mode == "sampled-budget-separated" || mode == "sampled-budget-fallback") {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (mode == "sampled-budget-separated") sampledBudgetIsolationTests(context);
            else sampledBudgetFallbackTests(context);
            return 0;
        }
        if (partialLive) {
#ifdef _WIN32
            if (!AgcDriver::GuestMemory::WriteWatched()) return VulkanTestSkipped;
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            partialCommittedWeakSnapshotTests(context);
            return 0;
#else
            std::puts("skipped, partial committed snapshot write-watch control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (compressedWeak || incompressibleWeak) {
#ifdef _WIN32
            if (!AgcDriver::GuestMemory::WriteWatched()) return VulkanTestSkipped;
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (compressedWeak) compressedUntrackedSnapshotTests(context);
            else incompressibleUntrackedSnapshotTests(context);
            return 0;
#else
            std::puts("skipped, compressed untracked snapshot control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (fileRawNativeRetention) {
#ifdef _WIN32
            if (!AgcDriver::GuestMemory::WriteWatched()) return VulkanTestSkipped;
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            fileBackedNativeRetentionTests(context);
            return 0;
#else
            std::puts("skipped, file-backed native retention control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (fileRawNativeArenaPressure) {
#ifdef _WIN32
            if (!AgcDriver::GuestMemory::WriteWatched()) return VulkanTestSkipped;
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            fileBackedNativeArenaPressureTests(context);
            return 0;
#else
            std::puts("skipped, file-backed native arena pressure control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (untrackedLive || untrackedBudget || fileRawWeakLive) {
#ifdef _WIN32
            if (!AgcDriver::GuestMemory::WriteWatched()) return VulkanTestSkipped;
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            untrackedWeakSnapshotTests(context, untrackedBudget, fileRawWeakLive);
            return 0;
#else
            std::puts("skipped, untracked weak snapshot byte-proof control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (pendingSnapshot) {
#ifdef _WIN32
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            for (const auto side : {512u, Side}) pendingSnapshotTests(context, mode == "snapshot-pending-write-disjoint", side);
            std::puts("direct sampled snapshot pending write tests passed");
            return 0;
#else
            std::puts("skipped, pending snapshot write-watch control requires Windows");
            return VulkanTestSkipped;
#endif
        }
        if (resources) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!resourceCacheTests(context, mode)) return VulkanTestSkipped;
            std::puts("resource cache tests passed");
            return 0;
        }
        if (mode == "resource-replaced") {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            replacedEntryTests(context);
            std::puts("resource cache replaced entry tests passed");
            return 0;
        }
        if (mode == "resource-lookup-changed") {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            if (!lookupChangedTests(context)) return VulkanTestSkipped;
            std::puts("resource cache lookup-changed tests passed");
            return 0;
        }
        if (memory) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            memoryTests(context);
            storageMemoryTests(context);
            ClearCachedTextures(context.device);
            std::puts("sampled texture memory budget tests passed");
            return 0;
        }
        if (recover) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            recoveryTests(context, mode);
            return 0;
        }
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            reportedBudgetTests(context);
            snapshotTests(context);
            recycleTests(context);
            heldTests(context);
            releasedTests(context);
            rescanTests(context);
            viewEvictionTests(context);
            viewTests(context);
            scanTests(context);
            frameTests(context);
            shrinkTests(context);
            secondUseTests(context);
            earlyMissTests(context);
            collidingKeyTests(context);
            spareAgeTests(context);
            ClearCachedTextures(context.device);
        }
        std::puts("sampled texture cache budget tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
