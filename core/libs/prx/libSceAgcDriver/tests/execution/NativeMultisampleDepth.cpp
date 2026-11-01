#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTextureLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include "VulkanTestDevice.hpp"
#include "FixtureDeviceSelection.hpp"
#include <SDL_loadso.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <iterator>
#endif
#endif

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}

namespace AgcDriver::DriverDetail {
struct DriverTestAccess {
    static std::shared_ptr<VulkanDevice> GetDevice() { return Driver::Get().device.Load(); }
};
}

namespace {

using namespace AgcDriver::Graphics;

class Device {
public:
    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#elif defined(__APPLE__)
        library = SDL_LoadObject("libvulkan.1.dylib");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
        Require(proc != nullptr, "Vulkan instance procedure is missing");
        try {
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            app.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &app;
#ifdef __APPLE__
            const char* portability = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
            info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            info.enabledExtensionCount = 1;
            info.ppEnabledExtensionNames = &portability;
#endif
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0u;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            context.physical = SelectFixturePhysicalDevice(devices, function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties"));
            VkPhysicalDeviceProperties selectedProperties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &selectedProperties);
            context.limits = selectedProperties.limits;
            std::printf("Selected GPU name=%s vendor=0x%x device=0x%x driver=%u\n", selectedProperties.deviceName, selectedProperties.vendorID, selectedProperties.deviceID, selectedProperties.driverVersion);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            std::uint32_t family = 0u;
            constexpr auto queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT;
            while (family < count && (families[family].queueFlags & queueFlags) != queueFlags) ++family;
            Require(family < count, "no graphics and compute queue");
            VkPhysicalDeviceFeatures available{};
            function<PFN_vkGetPhysicalDeviceFeatures>("vkGetPhysicalDeviceFeatures")(context.physical, &available);
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderStorageImageReadWithoutFormat = available.shaderStorageImageReadWithoutFormat;
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1u;
            queue.pQueuePriorities = &priority;
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device.queueCreateInfoCount = 1u;
            device.pQueueCreateInfos = &queue;
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0u, &context.queue);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
        } catch (...) { release(); throw; }
    }
    ~Device() { release(); }
    Context GetContext() const { return context; }
private:
    template<typename TFunction> TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(proc(instance, name));
        Require(result != nullptr, std::string("missing Vulkan function ") + name);
        return result;
    }
    void release() noexcept {
        if (context.device != VK_NULL_HANDLE) {
            context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device);
            ClearCachedTextures(context.device);
            ClearDepthSurfaces(context.device);
            context.bufferPool.reset();
            if (context.pool) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
            function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        }
        if (instance) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library) SDL_UnloadObject(library);
    }
    void* library = nullptr;
    PFN_vkGetInstanceProcAddr proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
};

#ifdef __linux__
struct SharedBacking {
    int file;
    std::size_t bytes;
    std::uint64_t offset;
};
std::map<std::uintptr_t, SharedBacking> sharedBlocks;

bool ResolveSharedBlock(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
    if (bytes == 0) return false;
    const auto next = sharedBlocks.upper_bound(address);
    if (next == sharedBlocks.begin()) return false;
    const auto entry = std::prev(next);
    const auto lead = address - entry->first;
    if (lead >= entry->second.bytes || bytes > entry->second.bytes - lead) return false;
    *file = fcntl(entry->second.file, F_DUPFD_CLOEXEC, 0);
    *offset = entry->second.offset + lead;
    return *file >= 0;
}

struct SharedBackingScope {
    SharedBackingScope() { GuestArena::GuestArenaSetSharedBacking_nid_postfix(ResolveSharedBlock); }
    ~SharedBackingScope() { GuestArena::GuestArenaSetSharedBacking_nid_postfix(nullptr); }
};
#endif

class TargetMemory {
public:
    explicit TargetMemory(std::size_t bytes, bool registerAllocation = false) : bytes(bytes), registered(registerAllocation) {
#ifdef _WIN32
        pointer = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536u);
        GuestArena::GuestArenaCommit_nid_postfix(pointer, bytes, PAGE_READWRITE, bytes);
#else
#ifdef __linux__
        backing = memfd_create("native-depth-htile-clear", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        Require(backing >= 0, "cannot create the native HTILE backing");
        if (ftruncate(backing, bytes + 65536u) != 0 || fcntl(backing, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
            close(backing);
            Require(false, "cannot size or seal the native HTILE backing");
        }
        void* raw = mmap(nullptr, bytes + 65536u, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
        if (raw == MAP_FAILED) close(backing);
#else
        void* raw = mmap(nullptr, bytes + 65536u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
        Require(raw != MAP_FAILED, "cannot allocate multisample target fixture");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + 65535u) & ~std::uintptr_t{65535u};
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + bytes != begin + bytes + 65536u) munmap(reinterpret_cast<void*>(aligned + bytes), begin + 65536u - aligned);
        pointer = reinterpret_cast<void*>(aligned);
#ifdef __linux__
        sharedBlocks.emplace(aligned, SharedBacking{backing, bytes, aligned - begin});
#endif
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(pointer, bytes);
#endif
        Require(pointer != nullptr, "multisample target fixture allocation failed");
        std::memset(pointer, 0xa5, bytes);
        if (registered) GuestAllocations::Mutation().Add(pointer, bytes, true, true, true);
    }
    ~TargetMemory() {
        if (registered) GuestAllocations::Mutation().Remove(pointer);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(pointer, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(pointer, bytes);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(pointer, bytes);
#ifdef __linux__
        sharedBlocks.erase(reinterpret_cast<std::uintptr_t>(pointer));
        close(backing);
#endif
        munmap(pointer, bytes);
#endif
    }
    std::byte* Data() const { return static_cast<std::byte*>(pointer); }
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(pointer); }
private:
    void* pointer = nullptr;
    std::size_t bytes;
    bool registered = false;
#ifdef __linux__
    int backing = -1;
#endif
};

alignas(256) constexpr std::array<std::uint32_t, 17> HtileMaskKernel{
    0xd7460000u, 0x04010c06u, 0x34000084u, 0xdc388000u, 0x04020000u, 0xbf8c3f70u, 0xd7710008u, 0x00120805u,
    0xd7710009u, 0x00120a05u, 0xd771000au, 0x00120c05u, 0xd771000bu, 0x00120e05u, 0xdc788000u, 0x00020800u,
    0xbf810000u,
};

std::vector<std::uint32_t> DriverPm4Packet(std::uint32_t opcode, std::initializer_list<std::uint32_t> body) {
    std::vector<std::uint32_t> words{0xc0000000u | (static_cast<std::uint32_t>(body.size() - 1u) << 16u) | (opcode << 8u)};
    words.insert(words.end(), body);
    return words;
}

Context SubmitDriverHtileClear(std::uint64_t address, std::unique_lock<AgcDriver::GuestMemory::GpuMutexType>& gpuLock) {
    const auto shader = reinterpret_cast<std::uintptr_t>(HtileMaskKernel.data());
    std::vector<std::uint32_t> commands;
    const auto append = [&commands](std::vector<std::uint32_t> packet) { commands.insert(commands.end(), packet.begin(), packet.end()); };
    append(DriverPm4Packet(0x76u, {0x20cu, static_cast<std::uint32_t>(shader >> 8u)}));
    append(DriverPm4Packet(0x76u, {0x20du, static_cast<std::uint32_t>(shader >> 40u) & 0xffu}));
    append(DriverPm4Packet(0x76u, {0x207u, 64u}));
    append(DriverPm4Packet(0x76u, {0x208u, 1u}));
    append(DriverPm4Packet(0x76u, {0x209u, 1u}));
    append(DriverPm4Packet(0x76u, {0x213u, (6u << 1u) | 0x80u}));
    append(DriverPm4Packet(0x76u, {0x240u, 0u}));
    append(DriverPm4Packet(0x76u, {0x241u, 0u}));
    append(DriverPm4Packet(0x76u, {0x242u, static_cast<std::uint32_t>(address)}));
    append(DriverPm4Packet(0x76u, {0x243u, static_cast<std::uint32_t>(address >> 32u)}));
    append(DriverPm4Packet(0x76u, {0x244u, 0xfffc00f0u}));
    append(DriverPm4Packet(0x76u, {0x245u, 0u}));
    append(DriverPm4Packet(0x15u, {32u, 1u, 1u, 0x41u}));
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0u, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "native depth HTILE clear: PM4 submission failed");
    gpuLock.unlock();
    try { AgcDriverWaitIdle_nid_postfix(); }
    catch (...) { gpuLock.lock(); throw; }
    gpuLock.lock();
    const auto driverDevice = AgcDriver::DriverDetail::DriverTestAccess::GetDevice();
    Require(driverDevice != nullptr, "HTILE clear driver device is missing");
    return driverDevice->graphicsContext();
}

template<typename Action>
void TargetRefuses(Action action, std::string_view expected) {
    std::string message;
    try { action(); }
    catch (const std::exception& error) { message = error.what(); }
    Require(message.find(expected) != std::string::npos, "multisample target refusal did not match " + std::string(expected) + ": " + message);
}

std::string_view PublicationRefusal(VkFormat format) {
    return format == VK_FORMAT_D16_UNORM ? "guest publication of a written native multisample D16 depth is not implemented" : "guest publication of a written native multisample depth needs the texture detiler";
}

void QueueTargetByteWrite(const Context& context, Recorder& recorder, std::uint64_t address, std::byte value) {
    auto buffer = std::make_shared<Buffer>(context, 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder.Commands();
    const auto byte = std::to_integer<std::uint32_t>(value);
    context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, buffer->Handle(), 0u, 4u, byte * 0x01010101u);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    recorder.OnComplete([buffer, address] {
        buffer->Invalidate();
        AgcDriver::GuestMemory::StoreOwnBytes(address, buffer->Bytes().first(1u));
    });
    recorder.Keep(buffer);
    recorder.NotePendingWrite(address, 1u);
}

std::vector<std::uint32_t> FetchShader(std::uint32_t width, std::uint32_t height, std::uint32_t fragments, bool arrayed, bool floating) {
    ShaderRecompiler::SpirvModule module;
    module.EmitCapability(spv::CapabilityShader);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto i32 = module.Type(spv::OpTypeInt, 32u, 1u);
    const auto scalar = floating ? module.Type(spv::OpTypeFloat, 32u) : u32;
    const auto v4 = module.Type(spv::OpTypeVector, scalar, 4u);
    const auto v3 = module.Type(spv::OpTypeVector, u32, 3u);
    const auto coordinateType = module.Type(spv::OpTypeVector, i32, arrayed ? 3u : 2u);
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
    const auto imageType = module.Type(spv::OpTypeImage, scalar, spv::Dim2D, 0u, arrayed ? 1u : 0u, 1u, 1u, spv::ImageFormatUnknown);
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
    const auto index = module.AllocateId();
    module.AddFunction(spv::OpCompositeExtract, u32, index, loaded, 0u);
    const auto binary = [&](spv::Op op, std::uint32_t a, std::uint32_t b) {
        const auto result = module.AllocateId();
        module.AddFunction(op, u32, result, a, b);
        return result;
    };
    const auto pixel = binary(spv::OpUDiv, index, constant(fragments));
    const auto x = binary(spv::OpUMod, pixel, constant(width));
    const auto y = binary(spv::OpUMod, binary(spv::OpUDiv, pixel, constant(width)), constant(height));
    const auto z = binary(spv::OpUDiv, pixel, constant(width * height));
    const auto signedValue = [&](std::uint32_t value) {
        const auto result = module.AllocateId();
        module.AddFunction(spv::OpBitcast, i32, result, value);
        return result;
    };
    const auto coordinate = module.AllocateId();
    if (arrayed) module.AddFunction(spv::OpCompositeConstruct, coordinateType, coordinate, signedValue(x), signedValue(y), signedValue(z));
    else module.AddFunction(spv::OpCompositeConstruct, coordinateType, coordinate, signedValue(x), signedValue(y));
    const auto object = module.AllocateId();
    module.AddFunction(spv::OpLoad, imageType, object, image);
    const auto value = module.AllocateId();
    module.AddFunction(spv::OpImageFetch, v4, value, object, coordinate, spv::ImageOperandsSampleMask, binary(spv::OpUMod, index, constant(fragments)));
    const auto pointer = module.AllocateId();
    module.AddFunction(spv::OpAccessChain, module.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, v4), pointer, buffer, constant(0u), binary(spv::OpIAdd, index, constant(1u)));
    module.AddFunction(spv::OpStore, pointer, value);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", {group});
    module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 1u, 1u, 1u);
    return module.Finalize();
}

class Fetch {
public:
    Fetch(const Context& context, const Texture& texture, Buffer& output, std::span<const std::uint32_t> code, bool firstLayer = false) : context(context) {
        try {
            const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{{0u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, {1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}}};
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = 2u;
            info.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &setLayout), "fetch descriptor layout");
            const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1u}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u}}};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1u;
            poolInfo.poolSizeCount = 2u;
            poolInfo.pPoolSizes = sizes.data();
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "fetch descriptor pool");
            const VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1u, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocate, &set), "fetch descriptor set");
            const VkDescriptorImageInfo image{VK_NULL_HANDLE, firstLayer ? texture.FirstLayerView() : texture.View(), texture.Layout()};
            const VkDescriptorBufferInfo buffer{output.Handle(), 0u, output.Bytes().size()};
            const std::array<VkWriteDescriptorSet, 2> writes{{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0u, 0u, 1u, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &image, nullptr, nullptr}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1u, 0u, 1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer, nullptr}}};
            context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 2u, writes.data(), 0u, nullptr);
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1u;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "fetch pipeline layout");
            VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shaderInfo.codeSize = code.size_bytes();
            shaderInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &shaderInfo, nullptr, &module), "fetch shader module");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.layout = layout;
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1u, &pipelineInfo, nullptr, &pipeline), "fetch pipeline");
        } catch (...) { release(); throw; }
    }
    ~Fetch() { release(); }
    void Run(VkCommandBuffer commands, std::uint32_t groups) const {
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0u, 1u, &set, 0u, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1u, 1u);
    }
private:
    void release() noexcept {
        if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }
    Context context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

std::vector<std::uint32_t> DepthVertexShader() {
    ShaderRecompiler::SpirvModule module;
    module.EmitCapability(spv::CapabilityShader);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto f32 = module.Type(spv::OpTypeFloat, 32u);
    const auto i32 = module.Type(spv::OpTypeInt, 32u, 1u);
    const auto boolean = module.Type(spv::OpTypeBool);
    const auto v4 = module.Type(spv::OpTypeVector, f32, 4u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto index = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassInput, i32), spv::StorageClassInput);
    const auto position = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassOutput, v4), spv::StorageClassOutput);
    module.AddAnnotation(spv::OpDecorate, index, spv::DecorationBuiltIn, spv::BuiltInVertexIndex);
    module.AddAnnotation(spv::OpDecorate, position, spv::DecorationBuiltIn, spv::BuiltInPosition);
    const auto main = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
    module.AddFunction(spv::OpLabel, module.AllocateId());
    const auto loaded = module.AllocateId();
    module.AddFunction(spv::OpLoad, i32, loaded, index);
    const auto coordinate = [&](std::uint32_t vertex) {
        const auto equal = module.AllocateId(), value = module.AllocateId();
        module.AddFunction(spv::OpIEqual, boolean, equal, loaded, module.Constant(spv::OpConstant, i32, vertex));
        module.AddFunction(spv::OpSelect, f32, value, equal, module.Constant(spv::OpConstant, f32, std::bit_cast<std::uint32_t>(3.0f)), module.Constant(spv::OpConstant, f32, std::bit_cast<std::uint32_t>(-1.0f)));
        return value;
    };
    const auto value = module.AllocateId();
    module.AddFunction(spv::OpCompositeConstruct, v4, value, coordinate(1u), coordinate(2u), module.Constant(spv::OpConstant, f32, 0u), module.Constant(spv::OpConstant, f32, std::bit_cast<std::uint32_t>(1.0f)));
    module.AddFunction(spv::OpStore, position, value);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelVertex, main, "main", {index, position});
    return module.Finalize();
}

std::vector<std::uint32_t> DepthFragmentShader() {
    ShaderRecompiler::SpirvModule module;
    module.EmitCapability(spv::CapabilityShader);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto f32 = module.Type(spv::OpTypeFloat, 32u);
    const auto u32 = module.Type(spv::OpTypeInt, 32u, 0u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto block = module.Type(spv::OpTypeStruct, f32);
    module.AddAnnotation(spv::OpDecorate, block, spv::DecorationBlock);
    module.AddAnnotation(spv::OpMemberDecorate, block, 0u, spv::DecorationOffset, 0u);
    const auto push = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassPushConstant, block), spv::StorageClassPushConstant);
    const auto depth = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassOutput, f32), spv::StorageClassOutput);
    module.AddAnnotation(spv::OpDecorate, depth, spv::DecorationBuiltIn, spv::BuiltInFragDepth);
    const auto main = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
    module.AddFunction(spv::OpLabel, module.AllocateId());
    const auto pointer = module.AllocateId(), value = module.AllocateId();
    module.AddFunction(spv::OpAccessChain, module.Type(spv::OpTypePointer, spv::StorageClassPushConstant, f32), pointer, push, module.Constant(spv::OpConstant, u32, 0u));
    module.AddFunction(spv::OpLoad, f32, value, pointer);
    module.AddFunction(spv::OpStore, depth, value);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelFragment, main, "main", {depth});
    module.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
    module.AddExecutionMode(main, spv::ExecutionModeDepthReplacing);
    return module.Finalize();
}

std::vector<std::uint32_t> DepthFeedbackFragmentShader(std::uint32_t binding, bool sampling, float constantDepth = 0.0f, bool multisampledImage = true, bool constantResult = false) {
    ShaderRecompiler::SpirvModule module;
    module.EmitCapability(spv::CapabilityShader);
    module.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
    const auto f32 = module.Type(spv::OpTypeFloat, 32u);
    const auto i32 = module.Type(spv::OpTypeInt, 32u, 1u);
    const auto v4 = module.Type(spv::OpTypeVector, f32, 4u);
    const auto v2 = module.Type(spv::OpTypeVector, i32, 2u);
    const auto voidType = module.Type(spv::OpTypeVoid);
    const auto imageType = module.Type(spv::OpTypeImage, f32, spv::Dim2D, 0u, 0u, multisampledImage ? 1u : 0u, 1u, spv::ImageFormatUnknown);
    const auto imageArray = module.Type(spv::OpTypeArray, imageType, module.Constant(spv::OpConstant, i32, 1u));
    const auto image = sampling ? module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageArray), spv::StorageClassUniformConstant) : 0u;
    if (sampling) {
        module.AddAnnotation(spv::OpDecorate, image, spv::DecorationDescriptorSet, ShaderRecompiler::RuntimeAbi::DescriptorSet);
        module.AddAnnotation(spv::OpDecorate, image, spv::DecorationBinding, binding);
    }
    const auto depth = module.DefineGlobalVariable(module.Type(spv::OpTypePointer, spv::StorageClassOutput, f32), spv::StorageClassOutput);
    module.AddAnnotation(spv::OpDecorate, depth, spv::DecorationBuiltIn, spv::BuiltInFragDepth);
    const auto zero = module.Constant(spv::OpConstant, i32, 0u);
    const auto coordinate = module.Constant(spv::OpConstantComposite, v2, zero, zero);
    const auto main = module.AllocateId();
    module.AddFunction(spv::OpFunction, voidType, main, spv::FunctionControlMaskNone, module.Type(spv::OpTypeFunction, voidType));
    module.AddFunction(spv::OpLabel, module.AllocateId());
    auto value = module.Constant(spv::OpConstant, f32, std::bit_cast<std::uint32_t>(constantDepth));
    if (sampling) {
        const auto element = module.AllocateId(), object = module.AllocateId(), fetched = module.AllocateId();
        value = module.AllocateId();
        module.AddFunction(spv::OpAccessChain, module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType), element, image, zero);
        module.AddFunction(spv::OpLoad, imageType, object, element);
        module.AddFunction(spv::OpImageFetch, v4, fetched, object, coordinate, multisampledImage ? spv::ImageOperandsSampleMask : spv::ImageOperandsLodMask, zero);
        module.AddFunction(spv::OpCompositeExtract, f32, value, fetched, 0u);
    }
    if (constantResult) value = module.Constant(spv::OpConstant, f32, std::bit_cast<std::uint32_t>(constantDepth));
    module.AddFunction(spv::OpStore, depth, value);
    module.AddFunction(spv::OpReturn);
    module.AddFunction(spv::OpFunctionEnd);
    module.EmitEntryPoint(spv::ExecutionModelFragment, main, "main", {depth});
    module.AddExecutionMode(main, spv::ExecutionModeOriginUpperLeft);
    module.AddExecutionMode(main, spv::ExecutionModeDepthReplacing);
    return module.Finalize();
}

class DepthRaster {
public:
    DepthRaster(const Context& context, const DepthTarget& target, VkImageView view) : context(context), extent(target.extent), samples(target.samples) {
        try {
            const VkAttachmentDescription attachment{0u, target.format, static_cast<VkSampleCountFlagBits>(samples), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            const VkAttachmentReference reference{0u, VK_IMAGE_LAYOUT_GENERAL};
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.pDepthStencilAttachment = &reference;
            VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            passInfo.attachmentCount = 1u;
            passInfo.pAttachments = &attachment;
            passInfo.subpassCount = 1u;
            passInfo.pSubpasses = &subpass;
            Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &pass), "depth sample render pass");
            VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebufferInfo.renderPass = pass;
            framebufferInfo.attachmentCount = 1u;
            framebufferInfo.pAttachments = &view;
            framebufferInfo.width = extent.width;
            framebufferInfo.height = extent.height;
            framebufferInfo.layers = 1u;
            Check(context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer), "depth sample framebuffer");
            const VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0u, 4u};
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.pushConstantRangeCount = 1u;
            layoutInfo.pPushConstantRanges = &push;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "depth sample pipeline layout");
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            const std::array codes{DepthVertexShader(), DepthFragmentShader()};
            for (std::uint32_t i = 0u; i < 2u; ++i) {
                VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                info.codeSize = codes[i].size() * 4u;
                info.pCode = codes[i].data();
                Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &info, nullptr, &modules[i]), "depth sample shader");
                stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u, i == 0u ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, modules[i], "main", nullptr};
            }
            VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
            const VkRect2D scissor{{0, 0}, extent};
            VkPipelineViewportStateCreateInfo viewportInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewportInfo.viewportCount = viewportInfo.scissorCount = 1u;
            viewportInfo.pViewports = &viewport;
            viewportInfo.pScissors = &scissor;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = static_cast<VkSampleCountFlagBits>(samples);
            VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
            VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            info.stageCount = 2u;
            info.pStages = stages.data();
            info.pVertexInputState = &input;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewportInfo;
            info.pRasterizationState = &raster;
            info.pMultisampleState = &multisample;
            info.pDepthStencilState = &depth;
            info.pColorBlendState = &blend;
            info.layout = layout;
            info.renderPass = pass;
            pipelines.resize(samples, VK_NULL_HANDLE);
            for (std::uint32_t sample = 0u; sample < samples; ++sample) {
                const auto mask = 1u << sample;
                multisample.pSampleMask = &mask;
                Check(context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, VK_NULL_HANDLE, 1u, &info, nullptr, &pipelines[sample]), "depth sample pipeline");
            }
        } catch (...) { release(); throw; }
    }
    ~DepthRaster() { release(); }
    void Run(VkCommandBuffer commands, float offset) const {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        begin.renderPass = pass;
        begin.framebuffer = framebuffer;
        begin.renderArea = {{0, 0}, extent};
        context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass")(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
        for (std::uint32_t sample = 0u; sample < samples; ++sample) {
            const auto value = offset + static_cast<float>(sample + 1u) / 16.0f;
            context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[sample]);
            context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0u, 4u, &value);
            context.Function<PFN_vkCmdDraw>("vkCmdDraw")(commands, 3u, 1u, 0u, 0u);
        }
        context.Function<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass")(commands);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
private:
    void release() noexcept {
        for (const auto pipeline : pipelines) if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        for (const auto module : modules) if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (framebuffer) context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(context.device, framebuffer, nullptr);
        if (pass) context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(context.device, pass, nullptr);
    }
    Context context;
    VkExtent2D extent;
    std::uint32_t samples;
    VkRenderPass pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::array<VkShaderModule, 2> modules{};
    std::vector<VkPipeline> pipelines;
};

std::array<std::uint32_t, 8> DepthSampleWords(const DepthTarget& target) {
    const auto base = (target.address | target.pipeBankXor) >> 8u;
    const auto metadata = target.htileAddress >> 8u;
    const auto format = target.format == VK_FORMAT_D16_UNORM ? 7u : 22u;
    const auto count = static_cast<std::uint32_t>(std::countr_zero(target.samples));
    const auto width = target.extent.width - 1u, height = target.extent.height - 1u;
    return {static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 255u) | (format << 20u) | ((width & 3u) << 30u), (width >> 2u) | (height << 14u), (14u << 28u) | (24u << 20u) | (count << 16u) | 0x204u, 0u, count << 4u, (target.iterate256 ? 1u << 10u : 0u) | (1u << 19u) | (1u << 21u) | (static_cast<std::uint32_t>(metadata & 255u) << 24u), static_cast<std::uint32_t>(metadata >> 8u)};
}

std::array<std::uint32_t, 8> HtileAliasWords(const DepthTarget& target) {
    constexpr std::uint32_t width = 16u;
    constexpr std::uint32_t height = 16u;
    constexpr std::uint32_t format8888Unorm = 56u;
    constexpr std::uint32_t type2D = 9u;
    const auto address = target.htileAddress;
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format8888Unorm << 20u) | (((width - 1u) & 3u) << 30u),
        ((width - 1u) >> 2u) | ((height - 1u) << 14u),
        0xfacu | (type2D << 28u),
        0u, 0u, 0u, 0u,
    };
}

bool DepthSupported(const Context& context, VkFormat format, std::uint32_t fragments) {
    VkImageFormatProperties properties{};
    const auto result = context.imageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0u, &properties);
    return result == VK_SUCCESS && (properties.sampleCounts & fragments) != 0u && (context.limits.framebufferDepthSampleCounts & fragments) != 0u;
}

void CheckDrawFeedback() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "native depth draw feedback requires Vulkan");
    constexpr VkExtent2D extent{5u, 3u};
    constexpr std::uint32_t samples = 4u;
    const auto depthBytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    TargetMemory memory(depthBytes), metadata(65536u);
    std::memset(metadata.Data(), 0, 65536u);
    DepthTarget depth{memory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, metadata.Address(), false, samples};
    depth.depthSwizzle = 24u;
    depth.iterate256 = depth.htilePipeAligned = true;
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.depth = depth;
    state.depthTest = state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.renderExtent = extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.samples = samples;
    ShaderRecompiler::RecompileResult vertex, producer, reader;
    vertex.spirv = DepthVertexShader();
    vertex.variantId = 1u;
    constexpr auto binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    producer.spirv = DepthFeedbackFragmentShader(binding, false);
    producer.variantId = 2u;
    const auto pixelOffset = PixelPushOffset(0u, device->Target());
    const AgcDriver::Pm4::DrawParameters draw{0u, 3u, 0u, 1u, 0u, false};
    const std::array<CompiledShader, 2> initial{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &producer, pixelOffset}}};
    device->Draw(state, draw, initial);
    device->WaitIdle();
    std::array<std::byte, 1> probe{};
    Require(DepthSurfaceAt(depth.address), "a draw left no resident native depth surface");
    reader.spirv = DepthFeedbackFragmentShader(binding, true);
    reader.variantId = 3u;
    const auto words = DepthSampleWords(depth);
    ShaderRecompiler::DescriptorBinding image{};
    image.kind = ShaderRecompiler::DescriptorKind::SampledImage;
    image.role = ShaderRecompiler::DescriptorRole::GuestImages;
    image.descriptorSet = ShaderRecompiler::RuntimeAbi::DescriptorSet;
    image.binding = binding;
    image.count = 1u;
    image.guestDescriptor.assign(words.begin(), words.end());
    image.imageShape = ShaderRecompiler::DescriptorImageShape::Image2DMultisample;
    image.imageSamplers = {0u};
    reader.bindings.push_back(std::move(image));
    const std::array<CompiledShader, 2> shaders{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &reader, pixelOffset}}};
    TargetRefuses([&] { device->Draw(state, draw, shaders); }, "sampling a native multisampled depth target the draw attaches");
    device->WaitIdle();
    TargetMemory testedMemory(depthBytes), testedMetadata(65536u);
    std::memset(testedMetadata.Data(), 0, 65536u);
    const auto guestByte = [](std::size_t offset) { return static_cast<std::byte>(offset * 7u + 3u); };
    for (std::size_t offset = 0u; offset < depthBytes; ++offset) testedMemory.Data()[offset] = guestByte(offset);
    auto tested = depth;
    tested.address = testedMemory.Address();
    tested.htileAddress = testedMetadata.Address();
    state.depth = tested;
    state.depthWrite = false;
    device->Draw(state, draw, initial);
    device->WaitIdle();
    Require(DepthSurfaceAt(tested.address), "a depth-test-only draw left no resident native depth surface");
    AgcDriver::GuestMemory::Read(tested.address + depthBytes - 1u, probe);
    Require(probe[0] == guestByte(depthBytes - 1u), "a guest read after a depth-test-only draw did not return its guest bytes");
    Require(!DepthSurfaceAt(tested.address), "a guest read left a native depth surface only depth-tested resident");
    std::puts("Native depth feedback refused through the draw resource path; a depth-tested surface retired on guest read");
}

void CheckCustomLocationHtileRead() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "custom-location HTILE alias requires Vulkan");
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    constexpr VkExtent2D extent{5u, 3u};
    constexpr std::uint32_t samples = 4u;
    Require(DepthSupported(context, VK_FORMAT_D32_SFLOAT, samples) && (context.sampleLocationSampleCounts & samples) != 0u && SampleLocationsCompatibleDepth(context, VK_FORMAT_D32_SFLOAT, samples), "the regression requires sampleable 4x D32 custom sample locations");
    const auto depthBytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    const auto metadataBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    const AgcDriver::Pm4::DrawParameters draw{0u, 3u, 0u, 1u, 0u, false};
    ShaderRecompiler::RecompileResult vertex;
    vertex.spirv = DepthVertexShader();
    vertex.variantId = 101u;
    constexpr auto binding = ShaderRecompiler::RuntimeAbi::FirstImageBinding;
    constexpr std::array<std::uint8_t, 4> locations{0x6au, 0xa6u, 0xeeu, 0x22u};

    for (const bool previouslyWritten : {false, true}) {
        TargetMemory memory(depthBytes), metadata(65536u);
        const std::uint32_t htile = previouslyWritten ? 0u : 0xfffffff0u;
        for (std::size_t offset = 0u; offset < metadataBytes; offset += 4u) std::memcpy(metadata.Data() + offset, &htile, sizeof(htile));
        DepthTarget depth{memory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, previouslyWritten ? 0.0f : 1.0f, 0u, metadata.Address(), false, samples};
        depth.depthSwizzle = 24u;
        depth.htileDepthOnly = true;
        depth.iterate256 = depth.htilePipeAligned = true;

        State state{};
        state.stages = {ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
        state.depth = depth;
        state.depthTest = true;
        state.depthWrite = previouslyWritten;
        state.depthCompare = VK_COMPARE_OP_ALWAYS;
        state.renderExtent = extent;
        state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        state.viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
        state.negativeOneToOne = false;
        state.scissor = {{0, 0}, extent};
        state.cullMode = VK_CULL_MODE_NONE;
        state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        state.samples = samples;
        state.customSampleLocations = true;
        for (std::uint32_t pixel = 0u; pixel < 4u; ++pixel) std::copy(locations.begin(), locations.end(), state.sampleLocations.begin() + pixel * 8u);

        if (!previouslyWritten) {
            Require(DepthSurfaceView(context, depth, false) != VK_NULL_HANDLE && DepthSurfaceAt(depth.address), "the HTILE clear fixture did not create an active guarded depth surface");
        } else {
            ShaderRecompiler::RecompileResult producer;
            producer.spirv = DepthFeedbackFragmentShader(binding, false, 0.375f);
            producer.variantId = 102u;
            const std::array<CompiledShader, 2> initial{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &producer, PixelPushOffset(0u, device->Target())}}};
            device->Draw(state, draw, initial);
            device->WaitIdle();
            Require(DepthSurfaceAt(depth.address), "the previously written depth surface lost its native owner");
            state.depthWrite = false;
        }

        ShaderRecompiler::RecompileResult reader;
        reader.spirv = DepthFeedbackFragmentShader(binding, true, 0.0f, false);
        reader.variantId = previouslyWritten ? 104u : 103u;
        ShaderRecompiler::DescriptorBinding image{};
        image.kind = ShaderRecompiler::DescriptorKind::SampledImage;
        image.role = ShaderRecompiler::DescriptorRole::GuestImages;
        image.descriptorSet = ShaderRecompiler::RuntimeAbi::DescriptorSet;
        image.binding = binding;
        image.count = 1u;
        const auto htileWords = HtileAliasWords(depth);
        image.guestDescriptor.assign(htileWords.begin(), htileWords.end());
        image.imageShape = ShaderRecompiler::DescriptorImageShape::Image2D;
        image.imageSamplers = {0u};
        reader.bindings.push_back(std::move(image));
        const std::array<CompiledShader, 2> shaders{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &reader, PixelPushOffset(0u, device->Target())}}};

        std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
        device->Draw(state, draw, shaders, {}, &recipe);
        device->WaitIdle();
        Require(DepthSurfaceAt(depth.address), "the custom-location draw left no active native depth surface");
        Require(recipe != nullptr, "the custom-location HTILE draw produced no recipe");
        std::array<std::byte, 1> recipeProbe{};
        AgcDriver::GuestMemory::Read(depth.htileAddress, recipeProbe);
        Require(!DepthSurfaceAt(depth.address), "the recipe probe left the native depth surface active");
        const auto replayed = device->DrawFromRecipe(state, draw, shaders, {}, recipe);
        device->WaitIdle();
        Require(replayed == AgcDriver::RecipeOutcome::Recorded, "the custom-location HTILE draw did not replay its recipe");

        auto words = DepthSampleWords(depth);
        auto texture = CachedSampledTexture(context, words);
        Require(texture != nullptr && DepthSurfaceHolds(context, DecodeTextureResource(words), texture.get()), "the custom-location depth output has no native sampled view");
        const auto count = extent.width * extent.height * samples;
        Buffer output(context, (count + 2u) * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
        const Fetch fetch(context, *texture, output, FetchShader(extent.width, extent.height, samples, false, true));
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), count);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        output.Invalidate();
        const auto expected = previouslyWritten ? 0.375f : 1.0f;
        for (std::uint32_t index = 0u; index < count; ++index) {
            std::array<float, 4> value{};
            std::memcpy(value.data(), output.Bytes().data() + (index + 1u) * 16u, sizeof(value));
            Require(value == std::array<float, 4>{expected, 0.0f, 0.0f, 1.0f}, "the custom-location sampled-resource draw lost an initial clear or previously written sample");
        }
        texture.reset();
        const auto retainedView = DepthSurfaceView(context, depth, false);
        std::array<std::byte, 1> probe{};
        AgcDriver::GuestMemory::Read(depth.htileAddress, probe);
        Require(!DepthSurfaceAt(depth.address), "reading HTILE did not retire the restored depth owner");
        Require(DepthSurfaceView(context, depth, false) == retainedView, "rebinding discarded unchanged native depth samples");
        ShaderRecompiler::RecompileResult writer;
        writer.spirv = DepthFeedbackFragmentShader(binding, true, 0.625f, false, true);
        writer.variantId = previouslyWritten ? 106u : 105u;
        writer.bindings = reader.bindings;
        state.depthWrite = true;
        const std::array<CompiledShader, 2> writing{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &writer, PixelPushOffset(0u, device->Target())}}};
        device->Draw(state, draw, writing);
        device->WaitIdle();
        AgcDriver::GuestMemory::Read(depth.address, probe);
        Require(!DepthSurfaceAt(depth.address), "reading newly written depth did not retire its restored owner");
        const MultisampleTextureLayout layout(TextureTileMode::kZ64KBX, 4u, samples, extent.width, extent.height, 1u, 0u);
        for (std::uint32_t y = 0u; y < extent.height; ++y) {
            for (std::uint32_t x = 0u; x < extent.width; ++x) {
                for (std::uint32_t sample = 0u; sample < samples; ++sample) {
                    float value = 0.0f;
                    std::memcpy(&value, memory.Data() + layout.Offset(x, y, 0u, sample), sizeof(value));
                    Require(value == 0.625f, "a write after depth restoration was lost on guest publication");
                }
            }
        }
        const std::array<std::byte, 4> changed{};
        AgcDriver::GuestMemory::Write(depth.htileAddress, changed, 4u);
        Require(DepthSurfaceView(context, depth, false) != retainedView, "changed guest metadata restored a stale native depth owner");
        device->WaitIdle();
        ClearCachedTextures(device->Device());
        ClearDepthSurfaces(device->Device());
        std::printf("Custom-location HTILE alias preserved %s samples, published later writes and refused changed metadata\n", previouslyWritten ? "previously written" : "initially cleared");
    }
}

void CheckWrittenPublication(std::uint32_t samples, VkExtent2D extent, bool depthOnly, std::uint32_t pipeBankXor) {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "native depth publication requires Vulkan");
    const auto depthBytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    const auto metadataBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    TargetMemory memory(depthBytes), metadata(65536u);
    std::memset(metadata.Data(), 0, 65536u);
    const auto guestByte = [](std::size_t offset) { return static_cast<std::byte>(offset * 7u + 3u); };
    for (std::size_t offset = 0u; offset < depthBytes; ++offset) memory.Data()[offset] = guestByte(offset);
    DepthTarget depth{memory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, metadata.Address(), false, samples};
    depth.depthSwizzle = 24u;
    depth.pipeBankXor = pipeBankXor;
    depth.htileDepthOnly = depthOnly;
    depth.iterate256 = depth.htilePipeAligned = true;
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    State state{};
    state.stages = {ShaderPath::Vertex, 0u, 32u, 32u, std::nullopt, std::nullopt};
    state.depth = depth;
    state.depthTest = state.depthWrite = true;
    state.depthCompare = VK_COMPARE_OP_ALWAYS;
    state.renderExtent = extent;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, extent};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.samples = samples;
    ShaderRecompiler::RecompileResult vertex, producer;
    vertex.spirv = DepthVertexShader();
    vertex.variantId = 1u;
    constexpr float written = 0.375f;
    producer.spirv = DepthFeedbackFragmentShader(ShaderRecompiler::RuntimeAbi::FirstImageBinding, false, written);
    producer.variantId = 2u;
    const AgcDriver::Pm4::DrawParameters draw{0u, 3u, 0u, 1u, 0u, false};
    const std::array<CompiledShader, 2> shaders{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0u}, {ShaderRecompiler::ShaderStage::Fragment, &producer, PixelPushOffset(0u, device->Target())}}};
    device->Draw(state, draw, shaders);
    device->WaitIdle();
    Require(DepthSurfaceAt(depth.address), "a draw left no resident native depth surface");
    std::array<std::byte, 1> probe{};
    AgcDriver::GuestMemory::Read(depth.address, probe);
    Require(!DepthSurfaceAt(depth.address), "publishing a written native depth surface left it resident");
    const MultisampleTextureLayout layout(TextureTileMode::kZ64KBX, 4u, samples, extent.width, extent.height, 1u, pipeBankXor);
    Require(layout.Bytes() == depthBytes, "published native depth layout differs from its guarded range");
    std::vector<bool> covered(depthBytes);
    const auto expected = std::bit_cast<std::uint32_t>(written);
    for (std::uint32_t y = 0u; y < extent.height; ++y) {
        for (std::uint32_t x = 0u; x < extent.width; ++x) {
            for (std::uint32_t sample = 0u; sample < samples; ++sample) {
                const auto offset = static_cast<std::size_t>(layout.Offset(x, y, 0u, sample));
                std::uint32_t value = 0u;
                std::memcpy(&value, memory.Data() + offset, sizeof(value));
                Require(value == expected, "a published native depth sample lost its drawn value");
                for (std::size_t byte = 0u; byte < sizeof(value); ++byte) covered[offset + byte] = true;
            }
        }
    }
    for (std::size_t offset = 0u; offset < depthBytes; ++offset) Require(covered[offset] || memory.Data()[offset] == guestByte(offset), "publishing native depth changed guest bytes outside its samples");
    const std::uint32_t htile = depthOnly ? 0xfffc000fu : 0xfffff00fu;
    for (std::size_t offset = 0u; offset < metadataBytes; offset += 4u) {
        std::uint32_t word = 0u;
        std::memcpy(&word, metadata.Data() + offset, sizeof(word));
        Require(word == htile, "publishing native depth did not expand its HTILE");
    }
    std::printf("Native depth publication samples=%u extent=%ux%u depthOnly=%u xor=0x%x: samples, HTILE and padding checked\n", samples, extent.width, extent.height, depthOnly, pipeBankXor);
}

void CheckDepthAttachmentAlias() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "native depth attachment alias requires Vulkan");
    const auto context = device->graphicsContext();
    constexpr VkExtent2D extent{5u, 3u};
    constexpr std::uint32_t samples = 4u;
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    TargetMemory memory(bytes), firstMetadata(65536u), secondMetadata(65536u);
    std::memset(memory.Data(), 0x3c, bytes);
    std::memset(firstMetadata.Data(), 0, 65536u);
    std::memset(secondMetadata.Data(), 0, 65536u);
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    DepthTarget first{memory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, firstMetadata.Address(), false, samples};
    first.depthSwizzle = 24u;
    first.iterate256 = first.htilePipeAligned = first.htileDepthOnly = true;
    const auto firstView = DepthSurfaceView(context, first);
    const auto words = DepthSampleWords(first);
    auto kept = DepthSurfaceTexture(context, words, DecodeTextureResource(words), VkComponentMapping{});
    Require(kept != nullptr, "attachment alias fixture has no held native depth view");
    auto second = first;
    second.extent.width = 6u;
    second.htileAddress = secondMetadata.Address();
    Require(MultisampledDepthBytes_nid_no_patch(second.extent, 4u, samples) <= bytes, "attachment alias fixture exceeds its backing");
    const auto secondView = DepthSurfaceView(context, second);
    Require(secondView != VK_NULL_HANDLE && secondView != firstView, "an aliased depth attachment reused the previous layout's view");
    Require(DepthSurfaceView(context, second) == secondView, "an unchanged native depth attachment was not reused");
    for (std::size_t offset = 0u; offset < bytes; ++offset) Require(memory.Data()[offset] == std::byte{0x3c}, "retiring an unwritten attachment alias changed guest bytes");
    std::puts("native depth attachment alias changes layout while retaining held GPU views");
}

void CheckStorageAliasRetirement() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "native depth storage alias requires Vulkan");
    const auto context = device->graphicsContext();
    constexpr VkExtent2D extent{5u, 3u};
    constexpr std::uint32_t samples = 4u;
    constexpr std::size_t prefix = 65536u;
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    TargetMemory memory(prefix + bytes), metadata(65536u);
    std::memset(memory.Data(), 0x3c, prefix + bytes);
    std::memset(metadata.Data(), 0, 65536u);
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    GuestTextureResource alias{};
    alias.baseAddress = memory.Address();
    alias.width = 256u;
    alias.height = static_cast<std::uint32_t>((prefix + bytes) / 1024u);
    alias.mipCount = alias.allocatedMipCount = 1u;
    alias.dimension = TextureDimension::k2D;
    alias.tileMode = TextureTileMode::kLinear;
    alias.format = 22u;
    alias.dstSelX = 0u;
    alias.dstSelY = alias.dstSelZ = 4u;
    alias.dstSelW = 5u;
    auto cached = CachedStorageSurface(context, alias);
    Require(cached != nullptr, "storage alias was not cached before native depth creation");
    DepthTarget depth{memory.Address() + prefix, 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, metadata.Address(), false, samples};
    depth.depthSwizzle = 24u;
    depth.iterate256 = depth.htilePipeAligned = depth.htileDepthOnly = true;
    Require(DepthSurfaceView(context, depth) != VK_NULL_HANDLE && DepthSurfaceAt(depth.address), "storage alias fixture has no native depth owner");
    auto disjoint = alias;
    disjoint.height = static_cast<std::uint32_t>(prefix / 1024u);
    Require(CachedStorageSurface(context, disjoint) != nullptr, "disjoint storage alias was not created");
    Require(DepthSurfaceAt(depth.address), "storage access ending at native depth retired a disjoint owner");
    Require(CachedStorageSurface(context, alias) != nullptr, "overlapping cached storage alias was not returned");
    Require(!DepthSurfaceAt(depth.address), "a cached storage view kept a native depth owner inside its range");
    for (std::size_t offset = 0u; offset < prefix + bytes; ++offset) Require(memory.Data()[offset] == std::byte{0x3c}, "retiring unwritten native depth changed guest bytes");
    std::puts("cached storage span retires only intersecting native depth owners");
}

enum class ClearFault { None, PartialStore, CpuWrite, DepthWrite, BadMetadata, UnprovenDispatch, PendingWrite, Overlap, PossibleStore, CompletedPossibleStore, DriverDispatch, ImportWindowWrite };

void CheckClearDispatch() {
    const std::array<std::uint32_t, 17> code{
        0xd7460000, 0x04010c06, 0x34000084, 0xdc388000, 0x04020000,
        0xbf8c3f70, 0xd7710008, 0x00120805, 0xd7710009, 0x00120a05,
        0xd771000a, 0x00120c05, 0xd771000b, 0x00120e05,
        0xdc788000, 0x00020800, 0xbf810000,
    };
    const std::array<std::uint32_t, 6> userData{0u, 0u, 65536u, 0u, 0xfffc00f0u, 0u};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {true, false, false}, false, 1u};
    const std::array<std::uint32_t, 5> packet{0u, 32u, 1u, 1u, 1u};
    Require(HtileDepthClearBytes_nid_no_patch(code, userData, compute, packet, 0u) == 32768u, "complete HTILE mask dispatch was not recognized");
    for (const auto groups : {0x3fffffu, 0x400000u, 0x400001u}) {
        auto boundary = packet;
        boundary[1] = groups;
        const auto expected = groups > 0x400000u ? 0u : static_cast<std::uint64_t>(groups) * 1024u;
        Require(HtileDepthClearBytes_nid_no_patch(code, userData, compute, boundary, 0u) == expected, "HTILE mask extent crossed its 32-bit byte-offset bound");
    }
    Require(HtileDepthClearBytes_nid_no_patch(code, userData, compute, packet, 65536u) == 0u, "indirect HTILE clear was accepted");
    Require(HtileDepthClearBytes_nid_no_patch(code, userData, compute, std::span(packet).first(4u), 0u) == 0u, "short HTILE dispatch packet was accepted");
    Require(HtileDepthClearBytes_nid_no_patch(code, std::span(userData).first(5u), compute, packet, 0u) == 0u, "short HTILE user data was accepted");
    auto changedCode = code;
    changedCode[0] ^= 1u;
    Require(HtileDepthClearBytes_nid_no_patch(changedCode, userData, compute, packet, 0u) == 0u, "another HTILE producer was accepted");
    for (std::uint32_t fault = 0u; fault < 14u; ++fault) {
        auto changedCompute = compute;
        auto changedPacket = packet;
        auto changedUserData = userData;
        switch (fault) {
            case 0u: changedCompute.partialThreads[0] = 1u; break;
            case 1u: changedCompute.groupIdEnable[0] = false; break;
            case 2u: changedCompute.threadIdComponentCount = 0u; break;
            case 3u: changedCompute.threadIdComponentCount = 4u; break;
            case 4u: changedCompute.numThreads[0] = 32u; break;
            case 5u: changedCompute.numThreads[1] = 2u; break;
            case 6u: changedPacket[1] = 0u; break;
            case 7u: changedPacket[2] = 2u; break;
            case 8u: changedPacket[3] = 2u; break;
            case 9u: changedPacket[4] = 0u; break;
            case 10u: changedPacket[4] = 0x21u; break;
            case 11u: changedUserData[2] = 65537u; break;
            case 12u: changedUserData[4] = 1u; break;
            case 13u: changedUserData[2] = 0xfffffff0u; changedUserData[3] = 0xffffffffu; break;
        }
        Require(HtileDepthClearBytes_nid_no_patch(code, changedUserData, changedCompute, changedPacket, 0u) == 0u, "unproven HTILE mask dispatch was accepted");
    }
}

void CheckDepthTarget(Context context, VkFormat format, std::uint32_t fragments, bool one, bool clearing, std::optional<bool> depthOnlyOverride, ClearFault fault, std::unique_lock<AgcDriver::GuestMemory::GpuMutexType>& gpuLock, bool iterate256 = true) {
    const VkExtent2D extent{5u, 3u};
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, format == VK_FORMAT_D16_UNORM ? 2u : 4u, fragments));
    TargetMemory memory(bytes), metadata(65536u, fault == ClearFault::DriverDispatch);
    const auto metadataBytes = HtileMetadataBytes(extent);
    const bool depthOnly = depthOnlyOverride.value_or(one);
    const std::uint32_t pattern = one ? (depthOnly ? 0xfffffff0u : 0xfffc00f0u) : 0u;
    for (std::size_t i = 0u; i < metadataBytes / 4u; ++i) std::memcpy(metadata.Data() + i * 4u, &pattern, 4u);
    DepthTarget target{memory.Address(), 0u, extent, format, one ? 1.0f : 0.0f, 0u, metadata.Address(), false, fragments};
    target.depthSwizzle = 24u;
    target.pipeBankXor = one ? 0x8c00u : 0u;
    target.htileDepthOnly = depthOnly;
    target.iterate256 = iterate256;
    target.htilePipeAligned = true;
    TargetMemory otherMemory(static_cast<std::size_t>(DepthSliceBytes(extent, 4u)));
    DepthTarget otherTarget{otherMemory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, 0u, false, 1u};
    Recorder recorder(context);
    recorder.Activate();
    struct Cleanup {
        Recorder& recorder;
        Context context;
        ~Cleanup() { recorder.Sync(); ClearCachedTextures(context.device); ClearDepthSurfaces(context.device); }
    } cleanup{recorder, context};
    if (!one) {
        metadata.Data()[metadataBytes - 1u] = std::byte{0x7f};
        QueueTargetByteWrite(context, recorder, target.htileAddress + metadataBytes - 1u, std::byte{0});
        Require(recorder.PendingWriteOverlaps(target.htileAddress, metadataBytes), "initial depth producer was not registered");
    }
    const auto view = DepthSurfaceView(context, target);
    Require(DepthSurfaceView(context, target) == view, "native depth attachment did not reuse its image");
    NoteDepthSurfaceWrite(context, target);
    std::array<std::byte, 1> beforeSampling{};
    TargetRefuses([&] { AgcDriver::GuestMemory::Read(target.address + bytes - 1u, beforeSampling); }, PublicationRefusal(format));
    TargetRefuses([&] { RequireNoMultisampleTarget(target.htileAddress + metadataBytes - 1u, 1u, "a DMA copy"); }, PublicationRefusal(format));
    const auto words = DepthSampleWords(target);
    auto texture = CachedSampledTexture(context, words);
    auto arrayTexture = CachedSampledTexture(context, words, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    Require(texture != nullptr && CachedSampledTexture(context, words) == texture && texture->SampledViewRange(false).type == VK_IMAGE_VIEW_TYPE_2D, "native depth sampled view was not reused");
    Require(arrayTexture != nullptr && arrayTexture != texture && CachedSampledTexture(context, words, VK_IMAGE_VIEW_TYPE_2D_ARRAY) == arrayTexture && arrayTexture->SampledViewRange(false).type == VK_IMAGE_VIEW_TYPE_2D_ARRAY, "native depth arrayed sampled view was not created as VK_IMAGE_VIEW_TYPE_2D_ARRAY or reused");
    Require(DepthSurfaceHolds(context, DecodeTextureResource(words), texture.get()) && DepthSurfaceHolds(context, DecodeTextureResource(words), arrayTexture.get()), "native depth sampled view is not held by its attachment");
    const std::array<std::shared_ptr<Texture>, 1> sampled{texture};
    TargetRefuses([&] { RequireNoNativeDepthFeedback_nid_no_patch(context, target, sampled); }, "sampling a native multisampled depth target the draw attaches");
    const std::array<std::shared_ptr<Texture>, 1> unused{};
    RequireNoNativeDepthFeedback_nid_no_patch(context, target, unused);
    auto unattached = target;
    unattached.address += bytes;
    RequireNoNativeDepthFeedback_nid_no_patch(context, unattached, sampled);
    auto singleSampled = target;
    singleSampled.samples = 1u;
    RequireNoNativeDepthFeedback_nid_no_patch(context, singleSampled, sampled);
    auto otherDevice = context;
    otherDevice.device = VK_NULL_HANDLE;
    RequireNoNativeDepthFeedback_nid_no_patch(otherDevice, target, sampled);
    recorder.Sync();
    const auto count = extent.width * extent.height * fragments;
    Buffer output(context, (count + 2u) * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const Fetch fetch(context, *texture, output, FetchShader(extent.width, extent.height, fragments, false, true));
    const Fetch arrayFetch(context, *arrayTexture, output, FetchShader(extent.width, extent.height, fragments, true, true));
    const auto check = [&](bool rendered, float offset) {
        output.Invalidate();
        for (std::uint32_t index = 0u; index < count; ++index) {
            std::array<float, 4> result{};
            std::memcpy(result.data(), output.Bytes().data() + (index + 1u) * 16u, 16u);
            const auto expected = rendered ? offset + static_cast<float>(index % fragments + 1u) / 16.0f : target.clearDepth;
            const auto tolerance = format == VK_FORMAT_D16_UNORM && rendered ? 1.0f / 65535.0f : 0.0f;
            Require(std::isfinite(result[0]) && std::abs(result[0] - expected) <= tolerance && result[1] == 0.0f && result[2] == 0.0f && result[3] == 1.0f, "native depth sampled fetch lost a raster sample or its component mapping");
        }
        Require(std::all_of(output.Bytes().begin(), output.Bytes().begin() + 16u, [](auto byte) { return byte == std::byte{0xa5}; }) && std::all_of(output.Bytes().end() - 16u, output.Bytes().end(), [](auto byte) { return byte == std::byte{0xa5}; }), "native depth sampled fetch changed output guards");
    };
    std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
    {
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), count);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        check(false, 0.0f);
    }
    std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
    {
        CommandBatch batch(context);
        arrayFetch.Run(batch.Handle(), count);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        check(false, 0.0f);
    }
    {
        const DepthRaster raster(context, target, view);
        for (const auto offset : {0.0f, 0.25f}) {
            Require(DepthSurfaceView(context, target) == view, "rebinding the native depth attachment replaced its image");
            Require(CachedSampledTexture(context, words) == texture && CachedSampledTexture(context, words, VK_IMAGE_VIEW_TYPE_2D_ARRAY) == arrayTexture, "native sampled view changed between raster writes");
            std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
            {
                CommandBatch batch(context);
                raster.Run(batch.Handle(), offset);
                batch.SubmitAndWait();
            }
            if (offset == 0.0f) DepthSurfaceView(context, otherTarget);
            Require(CachedSampledTexture(context, words) == texture, "native depth shadow view changed after a raster write");
            recorder.Sync();
            CommandBatch batch(context);
            fetch.Run(batch.Handle(), count);
            RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            batch.SubmitAndWait();
            check(true, offset);
            std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
            CommandBatch arrayBatch(context);
            arrayFetch.Run(arrayBatch.Handle(), count);
            RecordMemoryBarrier(context, arrayBatch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            arrayBatch.SubmitAndWait();
            check(true, offset);
        }
        if (clearing) {
            for (std::uint32_t iteration = 0u; iteration < 3u; ++iteration) {
                {
                    CommandBatch batch(context);
                    raster.Run(batch.Handle(), static_cast<float>(iteration) / 16.0f);
                    batch.SubmitAndWait();
                }
                const bool clearOne = iteration == 1u ? !one : one;
                target.clearDepth = clearOne ? 1.0f : 0.0f;
                Require(DepthSurfaceView(context, target) == view, "updating the native clear value replaced its attachment");
                const std::uint32_t clearPattern = clearOne ? (depthOnly ? 0xfffffff0u : 0xfffc00f0u) : 0u;
                if (fault == ClearFault::DriverDispatch && clearOne) {
                    struct ShutdownDriver {
                        ~ShutdownDriver() {
                            try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
                        }
                    } shutdownDriver;
                    const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
                    const auto driverContext = SubmitDriverHtileClear(target.htileAddress, gpuLock);
                    recorder.Activate();
                    Require(Recorder::Active() == &recorder, "driver idle did not restore the native depth fixture recorder");
                    if (!AgcDriver::GuestMemory::Watched(target.htileAddress, metadataBytes)) {
                        Require(PrepareImportWatch(driverContext) == ImportWatch::Unwatch,
                                "the HTILE metadata range lost write-watch coverage on a device whose host imports stay watched");
                        std::puts("skipped native HTILE write-coverage assertion: the write tracker does not cover the imported metadata range; the production depth-sampling path may still decline this clear");
                        TargetRefuses([&] { Require(DepthSurfaceHolds(context, DecodeTextureResource(words), texture.get()), "native depth owner was not retained"); },
                                      "native multisample depth clear requires tracked HTILE generations");
                        return;
                    } else {
                        const bool storedAll = AgcDriver::GuestMemory::StoredAllSince_nid_no_patch(target.htileAddress, metadataBytes, generation);
                        Require(storedAll, "driver idle returned before recording the completed HTILE dispatch");
                        std::puts("Native HTILE tracked dispatch completed");
                    }
                } else {
                    const auto nativeBytes = fault == ClearFault::UnprovenDispatch ? 0u : static_cast<std::size_t>(metadataBytes);
                    if (fault == ClearFault::PartialStore || fault == ClearFault::PossibleStore || fault == ClearFault::CompletedPossibleStore) {
                        for (std::size_t i = 0u; i < metadataBytes / 4u; ++i) std::memcpy(metadata.Data() + i * 4u, &clearPattern, 4u);
                    }
                    const bool completionCandidate = NoteHtileDepthClear(target.htileAddress, static_cast<std::size_t>(metadataBytes), nativeBytes);
                    if (fault == ClearFault::ImportWindowWrite) {
                        const auto beforeImport = AgcDriver::GuestMemory::TrackerGeneration();
                        Require(AgcDriver::GuestMemory::ImportWatched(target.htileAddress, static_cast<std::size_t>(metadataBytes), [&] {
                            metadata.Data()[0] ^= std::byte{0x01};
                            return true;
                        }), "HTILE host import simulation failed");
                        Require(AgcDriver::GuestMemory::CollectWritesUncached(target.htileAddress, static_cast<std::size_t>(metadataBytes)) > beforeImport, "HTILE import window did not stamp its changed page");
                    }
                    if (completionCandidate) ArmHtileDepthClear(target.htileAddress, static_cast<std::size_t>(metadataBytes));
                    if (fault == ClearFault::Overlap) NoteHtileDepthClear(target.htileAddress, static_cast<std::size_t>(metadataBytes), nativeBytes);
                    const auto storedBytes = fault == ClearFault::PartialStore ? 4u : static_cast<std::size_t>(metadataBytes);
                    if (fault == ClearFault::PossibleStore || fault == ClearFault::CompletedPossibleStore) {
                        const auto generation = AgcDriver::GuestMemory::TrackerGeneration();
                        AgcDriver::GuestMemory::MarkPossiblyWritten(target.htileAddress, metadataBytes);
                        Require(!AgcDriver::GuestMemory::StoredAllSince_nid_no_patch(target.htileAddress, metadataBytes, generation), "a possible HTILE range proved complete coverage");
                        if (fault == ClearFault::CompletedPossibleStore) CompleteHtileDepthClear(target.htileAddress, static_cast<std::size_t>(metadataBytes));
                    } else AgcDriver::GuestMemory::StoreOwnBytes(target.htileAddress, storedBytes, [&] {
                        for (std::size_t i = 0u; i < storedBytes / 4u; ++i) std::memcpy(metadata.Data() + i * 4u, &clearPattern, 4u);
                        if (fault == ClearFault::BadMetadata) metadata.Data()[0] = std::byte{0x0f};
                    });
                    if (fault == ClearFault::CpuWrite) metadata.Data()[metadataBytes - 1u] ^= std::byte{0x01};
                    if (fault == ClearFault::DepthWrite) memory.Data()[bytes - 1u] ^= std::byte{0x01};
                    if (fault == ClearFault::PendingWrite) {
                        QueueTargetByteWrite(context, recorder, target.htileAddress, std::byte{0});
                        std::string pendingRefusal;
                        try { CachedSampledTexture(context, words); }
                        catch (const std::exception& error) { pendingRefusal = error.what(); }
                        const bool refused = pendingRefusal.find("external GPU writes") != std::string::npos;
                        if (!refused) std::fprintf(stderr, "Pending HTILE texture refusal: %s\n", pendingRefusal.c_str());
                        Require(refused, "an unrecognized pending HTILE write was accepted or refused for another reason");
                    }
                }
                Require(DepthSurfaceHolds(context, DecodeTextureResource(words), texture.get()), "completed native depth clear lost its sampled view");
                recorder.Sync();
                Require(DepthSurfaceView(context, target) == view && CachedSampledTexture(context, words) == texture, "completed native depth clear replaced an attachment or cached view");
                std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
                CommandBatch batch(context);
                fetch.Run(batch.Handle(), count);
                RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
                batch.SubmitAndWait();
                check(false, 0.0f);
            }
            std::printf("Native depth completed clear format=%d samples=%u depthOnly=%u value=%u cycles=3\n", format, fragments, depthOnly, one);
        }
    }
    Require(std::all_of(memory.Data(), memory.Data() + bytes, [](auto value) { return value == std::byte{0xa5}; }), "native depth sampling silently published guest memory");
    std::array<std::byte, 1> probe{};
    for (const auto address : {target.address, target.address + bytes - 1u, target.htileAddress, target.htileAddress + metadataBytes - 1u}) {
        TargetRefuses([&] { RequireNoResidentDepthRead_nid_no_patch(address, 1u); }, PublicationRefusal(format));
        TargetRefuses([&] { AgcDriver::GuestMemory::Read(address, probe); }, PublicationRefusal(format));
        TargetRefuses([&] { RequireNoMultisampleTarget(address, 1u, "a DMA copy"); }, PublicationRefusal(format));
    }
    TargetMemory unrelated(65536u);
    AgcDriver::GuestMemory::Read(unrelated.Address(), probe);
    Require(probe[0] == std::byte{0xa5}, "resident depth guard blocked an unrelated read");
    auto alias = target;
    alias.htileAddress += 65536u;
    TargetRefuses([&] { DepthSurfaceView(context, alias); }, PublicationRefusal(format));
    alias = target;
    alias.htileDepthOnly = !target.htileDepthOnly;
    TargetRefuses([&] { DepthSurfaceView(context, alias); }, PublicationRefusal(format));
    auto resource = DecodeTextureResource(words);
    for (std::uint32_t field = 0u; field < 18u; ++field) {
        auto changed = resource;
        switch (field) {
            case 0u: changed.samples = fragments == 2u ? 4u : 2u; break;
            case 1u: ++changed.width; break;
            case 2u: ++changed.height; break;
            case 3u: changed.format = 20u; break;
            case 4u: changed.dimension = TextureDimension::k2DArray; break;
            case 5u: changed.depthOrLastArray = 1u; break;
            case 6u: changed.baseArray = 1u; break;
            case 7u: changed.minLod = 1u; break;
            case 8u: changed.pipeBankXor ^= 0x100u; break;
            case 9u: changed.dccAddress += 256u; break;
            case 10u: changed.dccPipeAligned = false; break;
            case 11u: changed.iterate256 = !changed.iterate256; break;
            case 12u: changed.allocatedMipCount = 2u; break;
            case 13u: changed.mipCount = 2u; break;
            case 14u: changed.baseLevel = 1u; break;
            case 15u: changed.lastLevel = 1u; break;
            case 16u: changed.dccAlphaOnMsb = true; break;
            case 17u: changed.tileMode = TextureTileMode::kR64KBX; break;
        }
        TargetRefuses([&] { DepthSurfaceTexture(context, words, changed, VkComponentMapping{}); }, "native multisample depth sampled");
    }
    memory.Data()[bytes - 1u] = std::byte{0x5a};
    TargetRefuses([&] { CachedSampledTexture(context, words); }, "guest writes");
    TargetRefuses([&] { DepthSurfaceHolds(context, resource, texture.get()); }, "guest writes");
    texture.reset();
    arrayTexture.reset();
    std::printf("Native depth format=%u samples=%u clear=%u iterate256=%u: imported, rasterized twice, sampled and refused conflicts\n", static_cast<std::uint32_t>(format), fragments, one, iterate256);
}

void CheckUnwrittenGuestAccess(Context context, VkFormat format, std::uint32_t fragments) {
    const VkExtent2D extent{5u, 3u};
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, format == VK_FORMAT_D16_UNORM ? 2u : 4u, fragments));
    const auto metadataBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    TargetMemory memory(bytes), metadata(65536u), unrelated(65536u);
    const auto guestByte = [](std::size_t offset) { return static_cast<std::byte>(offset * 7u + 3u); };
    for (std::size_t offset = 0u; offset < bytes; ++offset) memory.Data()[offset] = guestByte(offset);
    std::memset(metadata.Data(), 0, metadataBytes);
    DepthTarget target{memory.Address(), 0u, extent, format, 0.0f, 0u, metadata.Address(), false, fragments};
    target.depthSwizzle = 24u;
    target.iterate256 = target.htilePipeAligned = target.htileDepthOnly = true;
    Recorder recorder(context);
    recorder.Activate();
    struct Cleanup {
        Recorder& recorder;
        Context context;
        ~Cleanup() { recorder.Sync(); ClearCachedTextures(context.device); ClearDepthSurfaces(context.device); }
    } cleanup{recorder, context};
    const auto resident = [&] { return HasResidentDepthGuards_nid_no_patch() && DepthSurfaceAt(target.address); };
    auto view = DepthSurfaceView(context, target);
    Require(view != VK_NULL_HANDLE && resident(), "unwritten native depth fixture has no resident owner");
    auto otherDevice = context;
    otherDevice.device = VK_NULL_HANDLE;
    NoteDepthSurfaceWrite(otherDevice, target);
    auto unattached = target;
    unattached.address += 65536u;
    NoteDepthSurfaceWrite(context, unattached);
    std::array<std::byte, 1> probe{};
    AgcDriver::GuestMemory::Read(unrelated.Address(), probe);
    Require(resident(), "an unrelated guest read retired a native depth surface");
    for (const bool readMetadata : {false, true}) {
        if (readMetadata) RequireNoMultisampleTarget(target.htileAddress + metadataBytes - 1u, 1u, "a DMA copy");
        else {
            AgcDriver::GuestMemory::Read(target.address + bytes - 1u, probe);
            Require(probe[0] == guestByte(bytes - 1u), "a guest read over an unwritten native depth surface did not return its guest bytes");
        }
        Require(!resident(), "a guest access left an unwritten native depth surface resident");
        RequireNoResidentDepthRead_nid_no_patch(target.address, bytes);
        RequireNoResidentDepthRead_nid_no_patch(target.htileAddress, metadataBytes);
        recorder.Sync();
        const auto rebound = DepthSurfaceView(context, target);
        Require(rebound != VK_NULL_HANDLE && resident(), "rebinding after a guest access did not restore a native depth surface");
        view = rebound;
    }
    NoteDepthSurfaceWrite(context, target);
    TargetRefuses([&] { AgcDriver::GuestMemory::Read(target.address, probe); }, PublicationRefusal(format));
    TargetRefuses([&] { RequireNoMultisampleTarget(target.htileAddress, 1u, "a DMA copy"); }, PublicationRefusal(format));
    Require(resident(), "a refused guest access retired a written native depth surface");
    for (std::size_t offset = 0u; offset < bytes; ++offset) Require(memory.Data()[offset] == guestByte(offset), "retiring an unwritten native depth surface changed its guest bytes");
    Require(std::all_of(metadata.Data(), metadata.Data() + metadataBytes, [](std::byte value) { return value == std::byte{0}; }), "retiring an unwritten native depth surface changed its HTILE");
    std::printf("Native depth format=%u samples=%u: guest access retired the unwritten surface and refused the written one\n", static_cast<std::uint32_t>(format), fragments);
}

PFN_vkGetDeviceProcAddr reclaimDeviceProc = nullptr;
std::atomic<std::uint64_t> reclaimFreedMemory{0};

VKAPI_ATTR void VKAPI_CALL countReclaimFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator) {
    if (memory != VK_NULL_HANDLE) ++reclaimFreedMemory;
    reinterpret_cast<PFN_vkFreeMemory>(reclaimDeviceProc(device, "vkFreeMemory"))(device, memory, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL reclaimDeviceResolver(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkFreeMemory") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&countReclaimFreeMemory);
    return reclaimDeviceProc(device, name);
}

void CheckIdleRetiredReclaim(bool recorderSync = false) {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "native depth idle reclaim requires Vulkan");
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    context.bufferPool.reset();
    reclaimDeviceProc = context.deviceProc;
    reclaimFreedMemory = 0;
    context.deviceProc = &reclaimDeviceResolver;
    constexpr VkExtent2D extent{5u, 3u};
    constexpr std::uint32_t samples = 4u;
    Require(DepthSupported(context, VK_FORMAT_D32_SFLOAT, samples), "idle reclaim requires sampled four-sample D32");
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    const auto metadataBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    TargetMemory depthA(bytes), depthB(bytes), depthC(bytes);
    TargetMemory htileA(65536u), htileB(65536u), htileC(65536u);
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); ClearDepthSurfaces(device.Device()); }
    } cleanup{*device};
    for (auto* metadata : {&htileA, &htileB, &htileC}) std::memset(metadata->Data(), 0, metadataBytes);
    const auto makeTarget = [&](TargetMemory& depth, TargetMemory& htile) {
        DepthTarget target{depth.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, htile.Address(), false, samples};
        target.depthSwizzle = 24u;
        target.iterate256 = target.htilePipeAligned = target.htileDepthOnly = true;
        return target;
    };
    auto active = makeTarget(depthA, htileA);
    auto rebindable = makeTarget(depthB, htileB);
    auto retired = makeTarget(depthC, htileC);
    const auto activeView = DepthSurfaceView(context, active, false);
    auto activePin = DepthSurfaceOrderTarget(context, active).keep;
    const auto rebindView = DepthSurfaceView(context, rebindable, false);
    auto rebindPin = DepthSurfaceOrderTarget(context, rebindable).keep;
    Require(DepthSurfaceView(context, retired, false) != VK_NULL_HANDLE, "retired reclaim fixture has no depth attachment");
    auto pin = DepthSurfaceOrderTarget(context, retired).keep;
    std::weak_ptr<void> activeWeak = activePin, rebindWeak = rebindPin, retiredWeak = pin;
    activePin.reset();
    rebindPin.reset();
    const auto words = DepthSampleWords(retired);
    auto texture = DepthSurfaceTexture(context, words, DecodeTextureResource(words), VkComponentMapping{});
    Require(texture != nullptr, "retired reclaim fixture has no sampled view");
    RetireDepthSurfaces(context.device, active.address, bytes);
    RequireNoResidentDepthRead_nid_no_patch(rebindable.htileAddress, metadataBytes);
    RequireNoResidentDepthRead_nid_no_patch(retired.htileAddress, metadataBytes);
    RetireDepthSurfaces(context.device, retired.address, bytes);
    const auto sync = [&] {
        if (recorderSync) {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            context.recorder->Sync();
        } else device->WaitIdle();
    };
    sync();
    const auto released = DepthSurfacesReleased();
    Require(!retiredWeak.expired(), "a CPU-pinned retired depth surface was reclaimed");
    pin.reset();
    sync();
    Require(!retiredWeak.expired(), "a retired depth surface with a sampled view was reclaimed");
    texture.reset();
    const auto freed = reclaimFreedMemory.load();
    bool nestedCompletion = false;
    bool aliveDuringNestedSync = false;
    if (recorderSync) {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        context.recorder->Commands();
        context.recorder->OnComplete([&] {
            context.recorder->Sync();
            nestedCompletion = true;
            aliveDuringNestedSync = !retiredWeak.expired();
        });
        context.recorder->Sync();
    } else sync();
    if (recorderSync) Require(nestedCompletion && aliveDuringNestedSync, "a nested recorder sync reclaimed a depth surface while an outer batch was finishing");
    Require(retiredWeak.expired(), "an unpinned idle retired depth surface was not reclaimed");
    Require(DepthSurfacesReleased() == released + 1u, "reclaim did not invalidate the depth framebuffer generation");
    Require(reclaimFreedMemory.load() > freed, "reclaim did not free native depth Vulkan memory");
    Require(!activeWeak.expired() && !rebindWeak.expired(), "reclaim removed an active or rebindable depth surface");
    Require(DepthSurfaceView(context, active, false) == activeView, "reclaim changed an active retired depth binding");
    Require(DepthSurfaceView(context, rebindable, false) == rebindView, "reclaim changed a rebindable depth binding");
    Require(!DepthSurfaceAt(retired.address), "reclaim exposed a retired depth surface");
}

void WaitForPresentTrace(std::size_t expected = 1u) {
    const auto* path = std::getenv("APS5_CAPTURE_TRACE_PATH");
    if (std::getenv("APS5_CAPTURE_TRACE") == nullptr || path == nullptr || *path == '\0') return;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        std::ifstream input(path);
        const std::string trace{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        std::size_t count = 0;
        for (auto position = trace.find(" present dump="); position != std::string::npos; position = trace.find(" present dump=", position + 1u)) if (++count >= expected) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (std::chrono::steady_clock::now() < deadline);
    Require(false, "present capture trace did not flush before the deadline");
}

bool CheckPresentBorrowRefresh() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "present alias fixture requires Vulkan");
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    Require(context.recorder != nullptr && context.detiler != nullptr, "present alias fixture requires recorder and detiler");
    constexpr std::uint32_t side = 128u;
    constexpr std::size_t sliceBytes = 65536u;
    constexpr std::size_t bytes = 2u * sliceBytes;
    constexpr std::size_t importBytes = 0x100000u;
    TargetMemory memory(importBytes, true);
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); }
    } cleanup{*device};
    std::memset(memory.Data(), 0, bytes);
    std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
    if (HostImportFor(context, memory.Address(), importBytes) == nullptr) {
        std::puts("skipped, present alias fixture needs a host import");
        return false;
    }
    if (!AgcDriver::GuestMemory::Watched(memory.Address(), bytes)) {
        std::puts("skipped, present alias fixture needs write-watched guest memory after import");
        return false;
    }
    if (AgcDriver::GuestMemory::CollectWritesUncached(memory.Address(), bytes) == 0) {
        std::puts("skipped, present alias fixture needs tracked guest generations");
        return false;
    }
    GuestTextureResource first{};
    first.baseAddress = memory.Address();
    first.width = first.height = side;
    first.depthOrLastArray = 1u;
    first.mipCount = 1u;
    first.tileMode = TextureTileMode::kR64KBX;
    first.dimension = TextureDimension::k2DArray;
    first.format = 56u;
    first.dstSelX = 4u; first.dstSelY = 5u; first.dstSelZ = 6u; first.dstSelW = 7u;
    auto second = first;
    second.format = 60u;
    const auto geometry = DescribeSurface(first);
    Require(geometry.guestBytes == bytes && geometry.layerBytes == sliceBytes && geometry.imageLayers == 2u, "present alias fixture does not have two tracked 64 KiB layers");
    auto a = CachedStorageLayers(context, first, 1u, 1u);
    auto b = CachedStorageLayers(context, second, 0u, 1u);
    Require(a != nullptr && b != nullptr && a != b && a->Cached() && b->Cached(), "present alias fixture did not prewarm two cache entries");
    const auto readPixel = [&](const std::shared_ptr<StorageTexture>& image, std::uint32_t layer) {
        Buffer output(context, 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = context.recorder->Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, layer, 1u};
        copy.imageExtent = {1u, 1u, 1u};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, output.Handle(), 1u, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        context.recorder->Sync();
        output.Invalidate();
        std::array<std::uint8_t, 4> pixel{};
        std::memcpy(pixel.data(), output.Bytes().data(), pixel.size());
        return pixel;
    };
    const auto clear = [&](const std::shared_ptr<StorageTexture>& image, std::uint32_t layer, const VkClearColorValue& value) {
        const auto commands = context.recorder->Commands();
        context.recorder->Keep(image);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, layer, 1u};
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &value, 1u, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        image->MarkSliceDirty(layer);
    };
    const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
    VkClearColorValue green{};
    green.uint32[0] = 0u; green.uint32[1] = 255u; green.uint32[2] = 0u; green.uint32[3] = 255u;
    clear(a, 1u, red);
    Require(CachedStorageLayers(context, second, 0u, 1u) == b, "present alias fixture replaced the prewarmed second image");
    Require(readPixel(b, 1u) == std::array<std::uint8_t, 4>{255u, 0u, 0u, 255u}, "present alias fixture did not borrow the pending first image layer");
    clear(b, 0u, green);
    Require(StorageTexture::FindPendingSlice(memory.Address(), sliceBytes) == a, "present alias fixture did not select the older pending image");
    Require(PendingStorageOverlaps(memory.Address(), sliceBytes, a.get()), "present alias fixture has no foreign pending image");
    Require(AgcDriver::GuestMemory::UnchangedSince(memory.Address(), sliceBytes, a->GenerationFor(memory.Address(), sliceBytes)), "present alias fixture guest memory changed");
    Require(readPixel(a, 0u) == std::array<std::uint8_t, 4>{0u, 0u, 0u, 0u}, "the older image's first layer was not initially stale");
    const AgcDriver::DisplayBuffer display{memory.Address(), 0x8000000022000000ull, side, side};
    Require(AgcDriver::DisplayBufferSize(display) == sliceBytes, "present alias fixture display extent differs from its first layer");
    std::string refusal;
    try { device->PresentDisplayBuffer(display); }
    catch (const std::exception& error) { refusal = error.what(); }
    Require(refusal == "Vulkan presentation: device has no swapchain", "present alias fixture did not reach the headless presentation: " + refusal);
    Require(readPixel(a, 0u) == std::array<std::uint8_t, 4>{0u, 255u, 0u, 255u}, "present selected stale first-layer pixels over a newer pending alias");
    Require(readPixel(a, 1u) == std::array<std::uint8_t, 4>{255u, 0u, 0u, 255u}, "present refresh lost the older image's independently pending second layer");
    {
        TargetMemory control(importBytes, true);
        std::memset(control.Data(), 0, bytes);
        Require(HostImportFor(context, control.Address(), importBytes) != nullptr && AgcDriver::GuestMemory::Watched(control.Address(), bytes), "control needs a watched host import");
        auto surface = first;
        surface.baseAddress = control.Address();
        auto image = CachedStorageLayers(context, surface, 0u, 1u);
        clear(image, 0u, red);
        const AgcDriver::DisplayBuffer alone{control.Address(), 0x8000000022000000ull, side, side};
        Require(StorageTexture::FindPendingSlice(control.Address(), sliceBytes) == image && !PendingStorageOverlaps(control.Address(), sliceBytes, image.get()), "control present has a foreign pending image");
        Require(AgcDriver::GuestMemory::UnchangedSince(control.Address(), sliceBytes, image->GenerationFor(control.Address(), sliceBytes)), "control present guest memory changed");
        std::string controlRefusal;
        try { device->PresentDisplayBuffer(alone); }
        catch (const std::exception& error) { controlRefusal = error.what(); }
        Require(controlRefusal == "Vulkan presentation: device has no swapchain", "control did not reach headless presentation: " + controlRefusal);
        Require(readPixel(image, 0u) == std::array<std::uint8_t, 4>{255u, 0u, 0u, 255u}, "control present changed its resident pixels");
        device->WaitIdle();
        ClearCachedTextures(device->Device());
    }
    WaitForPresentTrace(2u);
    return true;
}

bool CheckPresentCopiedWriter() {
    const auto device = OpenVulkanTestDevice();
    Require(device != nullptr, "copied present fixture requires Vulkan");
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
    Require(context.recorder != nullptr && context.detiler != nullptr, "copied present fixture requires recorder and detiler");
    constexpr std::uint32_t side = 128u;
    constexpr std::size_t sliceBytes = 65536u;
    constexpr std::size_t bytes = 2u * sliceBytes;
    constexpr std::size_t recordOffset = 0x1004u;
    constexpr std::size_t writtenOffset = recordOffset + 8u;
    constexpr std::size_t importBytes = 1024u * 1024u;
    TargetMemory memory(importBytes, true);
    struct Cleanup {
        AgcDriver::VulkanDevice& device;
        ~Cleanup() { device.WaitIdle(); ClearCachedTextures(device.Device()); }
    } cleanup{*device};
    std::memset(memory.Data(), 0, bytes);
    std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
    if (HostImportFor(context, memory.Address(), importBytes) == nullptr || !AgcDriver::GuestMemory::Watched(memory.Address(), bytes) || context.limits.minStorageBufferOffsetAlignment <= 4u) {
        std::puts("skipped, copied present fixture requires watched host imports and storage alignment above 4");
        return false;
    }
    GuestTextureResource resource{};
    resource.baseAddress = memory.Address();
    resource.width = resource.height = side;
    resource.depthOrLastArray = 1u;
    resource.mipCount = 1u;
    resource.tileMode = TextureTileMode::kR64KBX;
    resource.dimension = TextureDimension::k2DArray;
    resource.format = 56u;
    resource.dstSelX = 4u; resource.dstSelY = 5u; resource.dstSelZ = 6u; resource.dstSelW = 7u;
    const auto geometry = DescribeSurface(resource);
    Require(geometry.guestBytes == bytes && geometry.layerBytes == sliceBytes && geometry.imageLayers == 2u, "copied present fixture requires two 64 KiB layers");
    auto image = CachedStorageLayers(context, resource, 1u, 1u);
    Require(image != nullptr, "copied present fixture has no resident image");
    const auto commands = context.recorder->Commands();
    context.recorder->Keep(image);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    const VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
    const VkImageSubresourceRange second{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 1u, 1u};
    context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image->Image(), VK_IMAGE_LAYOUT_GENERAL, &red, 1u, &second);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    image->MarkSliceDirty(1u);
    context.recorder->Sync();
    Require(Recorder::PendingWriteBackCompletions() == 0u, "copied present fixture started with a pending writeback");
    const AgcDriver::DisplayBuffer display{memory.Address(), 0x8000000022000000ull, side, side};
    Require(AgcDriver::DisplayBufferSize(display) == sliceBytes, "copied present fixture display size differs from its layer");
    std::uint32_t pixelX = side, pixelY = side;
    for (std::uint32_t y = 0; y < side; ++y) for (std::uint32_t x = 0; x < side; ++x) {
        if (AgcDriver::DisplayBufferOffset(display, x, y) == writtenOffset) { pixelX = x; pixelY = y; }
    }
    Require(pixelX < side && pixelY < side, "copied present fixture cannot map the written byte to a display texel");
    Require(StorageTexture::FindPendingSlice(memory.Address(), sliceBytes) == image, "copied present fixture has no pending resident array");
    Require(AgcDriver::GuestMemory::UnchangedSince(memory.Address(), sliceBytes, image->GenerationFor(memory.Address(), sliceBytes)), "copied present fixture source was already stale");
    alignas(256) constexpr std::array<std::uint32_t, 6> code{0x7e000281u, 0x7e020288u, 0xe0701000u, 0x80000001u, 0xbf810000u, 0xbf810000u};
    const auto record = memory.Address() + recordOffset;
    std::vector<std::uint32_t> userData(16, 0u);
    const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(record), static_cast<std::uint32_t>((record >> 32u) & 0xffffu), 16u, 0x31016facu};
    std::copy(descriptor.begin(), descriptor.end(), userData.begin());
    const std::span<const std::uint32_t> instructions(code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> shaderMemory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(instructions)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{32u, 1u, 1u}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{{ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), instructions, 0, {}}, {32, 0, userData, compute, std::nullopt, std::nullopt, shaderMemory}, device->Target(), {0, 0, 0, 128}};
    request.useCache = false;
    const auto program = ShaderRecompiler::Recompile(request);
    device->Dispatch(program, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    if (Recorder::PendingWriteBackCompletions() == 0u) {
        std::puts("skipped, dispatch did not leave a copied-buffer writeback completion");
        return false;
    }
    Require(Recorder::PendingWriteBackCompletions() == 1u, "copied present fixture has an unexpected number of pending writebacks");
    Require(StorageTexture::FindPendingSlice(memory.Address(), sliceBytes) == image, "copied dispatch removed the intended pending resident image");
    Require(Recorder::PendingWriteBackCompletions() == 1u, "copied writer completed before presentation");
    Require(context.recorder->PendingWriteOverlaps(record, 16u), "copied writer has no pending write note for its buffer");
    Require(!PendingStorageOverlaps(memory.Address(), sliceBytes, image.get()), "copied writer fixture has a foreign pending storage image");
    Require(AgcDriver::GuestMemory::UnchangedSince(memory.Address(), sliceBytes, image->GenerationFor(memory.Address(), sliceBytes)), "copied writer fixture already invalidated the resident generation");
    Require(AgcDriver::GuestMemory::GpuMutex().DepthOnThisThread() == 1u, "copied present fixture requires an outermost GPU hold");
    std::string refusal;
    try { device->PresentDisplayBuffer(display); } catch (const std::exception& error) { refusal = error.what(); }
    Require(refusal == "Vulkan presentation: device has no swapchain", "copied present fixture did not reach headless presentation: " + refusal);
    const auto readPixel = [&](std::uint32_t layer, std::uint32_t x, std::uint32_t y) {
        Buffer output(context, 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto read = context.recorder->Commands();
        RecordMemoryBarrier(context, read, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, layer, 1u};
        copy.imageOffset = {static_cast<std::int32_t>(x), static_cast<std::int32_t>(y), 0};
        copy.imageExtent = {1u, 1u, 1u};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(read, image->Image(), VK_IMAGE_LAYOUT_GENERAL, output.Handle(), 1u, &copy);
        RecordMemoryBarrier(context, read, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        context.recorder->Sync();
        output.Invalidate();
        std::array<std::uint8_t, 4> pixel{};
        std::memcpy(pixel.data(), output.Bytes().data(), pixel.size());
        return pixel;
    };
    Require(readPixel(0u, pixelX, pixelY) == std::array<std::uint8_t, 4>{1u, 0u, 0u, 0u}, "present missed the copied writer's layer 0 pixel");
    Require(readPixel(1u, 0u, 0u) == std::array<std::uint8_t, 4>{255u, 0u, 0u, 255u}, "present lost the independently pending layer 1 pixel");
    WaitForPresentTrace();
    return true;
}

void CheckHtileRouting(Context context, VkFormat format, std::uint32_t fragments) {
    const VkExtent2D extent{5u, 3u};
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, format == VK_FORMAT_D16_UNORM ? 2u : 4u, fragments));
    TargetMemory memory(bytes), metadata(65536u), single(65536u);
    const auto metadataBytes = HtileMetadataBytes(extent);
    std::memset(metadata.Data(), 0, metadataBytes);
    const std::vector<std::byte> cleared(metadata.Data(), metadata.Data() + metadataBytes);
    const auto untouched = [&] { return std::equal(cleared.begin(), cleared.end(), metadata.Data()); };
    DepthTarget target{memory.Address(), 0u, extent, format, 0.0f, 0u, metadata.Address(), false, fragments};
    target.depthSwizzle = 24u;
    target.iterate256 = target.htilePipeAligned = true;
    Recorder recorder(context);
    recorder.Activate();
    struct Cleanup {
        Recorder& recorder;
        Context context;
        ~Cleanup() { recorder.Sync(); ClearCachedTextures(context.device); ClearDepthSurfaces(context.device); }
    } cleanup{recorder, context};
    const auto view = DepthSurfaceView(context, target);
    Require(HasResidentDepthGuards_nid_no_patch(), "the native depth owner installed no guest guard");
    Require(untouched(), "the first bind of a native depth target stored expanded HTILE words back");
    Require(DepthSurfaceView(context, target) == view && untouched(), "rebinding a native-owned depth target went through the generic HTILE store-back");
    auto alias = target;
    alias.samples = 1u;
    alias.address = single.Address();
    Require(DepthSurfaceView(context, alias) != VK_NULL_HANDLE, "a single-sample bind could not retire an unwritten native HTILE owner");
    Require(untouched(), "a single-sample bind over a native owner's HTILE stored expanded words into it");
    std::printf("Native depth format=%u samples=%u: binds over the owner's HTILE left its words alone\n", static_cast<std::uint32_t>(format), fragments);
}

void CheckEligibility(Context context) {
    constexpr VkExtent2D extent{129u, 67u};
    constexpr std::uint32_t samples = 4u;
    const auto bytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, 4u, samples));
    TargetMemory memory(bytes), metadata(65536u), stencil(bytes);
    std::memset(metadata.Data(), 0, 65536u);
    DepthTarget target{memory.Address(), 0u, extent, VK_FORMAT_D32_SFLOAT, 0.0f, 0u, metadata.Address(), false, samples};
    target.depthSwizzle = 24u;
    target.iterate256 = target.htilePipeAligned = true;
    Recorder recorder(context);
    recorder.Activate();
    const auto lookup = [&](const DepthTarget& use) {
        const auto words = DepthSampleWords(use);
        return DepthSurfaceTexture(context, words, DecodeTextureResource(words), VkComponentMapping{});
    };
    const auto clear = [&] {
        recorder.Sync();
        ClearCachedTextures(context.device);
        ClearDepthSurfaces(context.device);
    };
    struct Cleanup {
        decltype(clear)& action;
        ~Cleanup() { action(); }
    } cleanup{clear};
    for (std::uint32_t field = 0u; field < 19u; ++field) {
        auto invalid = target;
        std::string_view reason;
        switch (field) {
            case 0u: invalid.depthSwizzle = 0u; reason = "non-Z_X"; break;
            case 1u: invalid.firstSlice = 1u; reason = "arrays"; break;
            case 2u: invalid.lastSlice = 1u; reason = "arrays"; break;
            case 3u: invalid.pipeBankXor = 0x10u; reason = "XOR"; break;
            case 4u: invalid.htileAddress = 0u; reason = "HTILE"; break;
            case 5u: invalid.htileAddress += 16u; reason = "HTILE"; break;
            case 6u: invalid.htilePipeAligned = false; reason = "HTILE"; break;
            case 7u: invalid.htileVrsEncoding = 1u; reason = "VRS"; break;
            case 8u: invalid.clearDepth = 0.5f; reason = "clear value"; break;
            case 9u: invalid.htileAddress = invalid.address; reason = "overlap"; break;
            case 10u: metadata.Data()[HtileMetadataBytes(extent) - 1u] = std::byte{0x40}; reason = "clear value"; break;
            case 11u: invalid.format = VK_FORMAT_D32_SFLOAT_S8_UINT; invalid.stencilAddress = stencil.Address(); reason = "stencil"; break;
            case 12u: invalid.htileDepthOnly = true; metadata.Data()[0] = std::byte{0x30}; reason = "clear value"; break;
            case 13u: invalid.clearDepth = 1.0f; { const std::uint32_t word = 0xfffffff0u; for (std::size_t offset = 0u; offset < HtileMetadataBytes(extent); offset += 4u) std::memcpy(metadata.Data() + offset, &word, 4u); } reason = "clear value"; break;
            case 14u: metadata.Data()[0] = std::byte{0xf}; reason = "clear value"; break;
            case 15u: invalid.clearDepth = 1.0f; reason = "clear value"; break;
            case 16u: invalid.clearDepth = 1.0f; invalid.htileDepthOnly = true; reason = "clear value"; break;
            case 17u: { const std::uint32_t word = 0xfffc0000u; for (std::size_t offset = 0u; offset < HtileMetadataBytes(extent); offset += 4u) std::memcpy(metadata.Data() + offset, &word, 4u); } reason = "clear value"; break;
            case 18u: invalid.htileDepthOnly = true; { const std::uint32_t word = 0xfffffff0u; for (std::size_t offset = 0u; offset < HtileMetadataBytes(extent); offset += 4u) std::memcpy(metadata.Data() + offset, &word, 4u); } reason = "clear value"; break;
        }
        Require(DepthSurfaceView(context, invalid) != VK_NULL_HANDLE, "sampling ineligibility must preserve attachment rendering");
        TargetRefuses([&] { lookup(invalid); }, reason);
        if (field == 11u) {
            GuestTextureResource alias{};
            alias.baseAddress = invalid.stencilAddress;
            alias.width = extent.width;
            alias.height = extent.height;
            alias.dimension = TextureDimension::k2D;
            alias.samples = 1u;
            alias.format = 36u;
            Require(DepthSurfaceTexture(context, {}, alias, VkComponentMapping{}) == nullptr, "single-sample color at a native stencil address must not be claimed as a depth reader");
            alias.format = 5u;
            TargetRefuses([&] { DepthSurfaceTexture(context, {}, alias, VkComponentMapping{}); }, "stencil");
        }
        if (field == 10u) {
            std::memset(metadata.Data(), 0, 65536u);
            TargetRefuses([&] { lookup(invalid); }, reason);
        }
        clear();
        std::memset(metadata.Data(), 0, 65536u);
    }
    {
        TargetMemory unwatched(bytes);
        AgcDriver::GuestMemory::Unwatch(unwatched.Address(), bytes);
        auto invalid = target;
        invalid.address = unwatched.Address();
        Require(DepthSurfaceView(context, invalid) != VK_NULL_HANDLE, "unwatched sampling memory must preserve attachment rendering");
        TargetRefuses([&] { lookup(invalid); }, "watched accessible guest memory");
        NoteDepthMetadataFill(invalid.htileAddress, HtileMetadataBytes(extent), 0u);
        TargetRefuses([&] { lookup(invalid); }, "watched accessible guest memory");
        clear();
    }
    DepthSurfaceView(context, target);
    metadata.Data()[HtileMetadataBytes(extent) - 1u] = std::byte{0x40};
    TargetRefuses([&] { lookup(target); }, "guest writes");
    clear();
    std::memset(metadata.Data(), 0, 65536u);
    DepthSurfaceView(context, target);
    QueueTargetByteWrite(context, recorder, target.address + bytes - 1u, std::byte{0x5a});
    TargetRefuses([&] { lookup(target); }, "external GPU writes");
    clear();
    DepthSurfaceView(context, target);
    NoteDepthMetadataFill(target.htileAddress, HtileMetadataBytes(extent), 0u);
    TargetRefuses([&] { lookup(target); }, "metadata fill transition");
    clear();
    const auto initialView = DepthSurfaceView(context, target);
    auto changedClearRegister = target;
    changedClearRegister.clearDepth = 1.0f;
    Require(DepthSurfaceView(context, changedClearRegister) == initialView, "changing only DB_DEPTH_CLEAR must reuse the native attachment without recertifying its initial contents");
    auto texture = lookup(target);
    recorder.Sync();
    ClearDepthSurfaces(context.device);
    Require(texture != nullptr && texture->View() != VK_NULL_HANDLE, "a held depth view lost its native image owner");
    Buffer output(context, (extent.width * extent.height * samples + 2u) * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
    {
        const Fetch fetch(context, *texture, output, FetchShader(extent.width, extent.height, samples, false, true));
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), extent.width * extent.height * samples);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        output.Invalidate();
        for (std::uint32_t i = 0u; i < extent.width * extent.height * samples; ++i) {
            std::array<float, 4> value{};
            std::memcpy(value.data(), output.Bytes().data() + (i + 1u) * 16u, 16u);
            Require(value == std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}, "a held native depth view lost its image contents when the surface cache was cleared");
        }
    }
    texture.reset();
    std::puts("Native depth eligibility, guest writes, queued writes, metadata transitions and held views checked");
}

void CheckRetirement(Context context, VkFormat format, std::uint32_t samples, std::uint32_t mode) {
    constexpr VkExtent2D extent{5u, 3u};
    const auto depthBytes = static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(extent, format == VK_FORMAT_D16_UNORM ? 2u : 4u, samples));
    const auto metadataBytes = static_cast<std::size_t>(HtileMetadataBytes(extent));
    const auto allocationBytes = depthBytes + 65536u;
    TargetMemory memory(allocationBytes), colorMemory(65536u);
    const auto release = [](const void*, std::size_t, const void*, bool) {};
    const auto add = [&](TargetMemory& allocation, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Add(allocation.Data(), bytes, true, true, true);
    };
    const auto remove = [&](TargetMemory& allocation, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        if (mutation.Overlaps(allocation.Data(), bytes)) mutation.Unmap(allocation.Data(), bytes, release);
    };
    Recorder recorder(context);
    recorder.Activate();
    const auto cleanup = [&] {
        recorder.Sync();
        ClearCachedTextures(context.device);
        ClearDepthSurfaces(context.device);
        ClearMultisampleTargets(context.device);
        remove(memory, allocationBytes);
        remove(colorMemory, 65536u);
    };
    struct Cleanup {
        decltype(cleanup)& action;
        ~Cleanup() { action(); }
    } cleanupScope{cleanup};
    add(memory, allocationBytes);
    add(colorMemory, 65536u);
    std::memset(memory.Data() + depthBytes, 0, metadataBytes);
    DepthTarget target{memory.Address(), 0u, extent, format, 0.0f, 0u, memory.Address() + depthBytes, false, samples};
    target.depthSwizzle = 24u;
    target.iterate256 = target.htilePipeAligned = target.htileDepthOnly = true;
    const auto words = DepthSampleWords(target);
    const auto resource = DecodeTextureResource(words);
    const auto firstView = DepthSurfaceView(context, target);
    NoteDepthSurfaceWrite(context, target);
    auto firstTexture = DepthSurfaceTexture(context, words, resource, VkComponentMapping{});
    Require(firstTexture != nullptr, "retirement fixture has no native sampled view");
    ColorTarget color{colorMemory.Address(), extent, VK_FORMAT_R8G8B8A8_UNORM, extent.width * extent.height * 4u, 0u};
    color.tileMode = ColorTileMode::RenderTarget;
    color.samples = 4u;
    color.surfaceAddress = color.address;
    Require(MultisampleTargetView(context, color) != VK_NULL_HANDLE, "retirement fixture has no color attachment");
    recorder.Sync();
    const auto guarded = [&] {
        TargetRefuses([&] { RequireNoResidentDepthRead_nid_no_patch(target.address, 1u); }, PublicationRefusal(format));
    };
    {
        GuestAllocations::Mutation mutation;
        mutation.Protect(memory.Data(), allocationBytes, true, true, true, [] {});
    }
    guarded();
    {
        GuestAllocations::Mutation mutation;
        TargetRefuses([&] { mutation.Unmap(memory.Data(), allocationBytes, [](const void*, std::size_t, const void*, bool) { throw std::runtime_error("retirement fixture failed host release"); }); }, "failed host release");
        Require(mutation.Covers(memory.Data(), allocationBytes), "failed host release changed registry coverage");
    }
    guarded();
    if ((mode & 1u) == 0u) ClearMultisampleTargets(context.device);
    const auto unmap = [&](std::size_t offset, std::size_t bytes) {
        GuestAllocations::Mutation mutation;
        mutation.Unmap(memory.Data() + offset, bytes, release);
    };
    if (mode == 0u) {
        GuestAllocations::Mutation mutation;
        mutation.Remove(memory.Data());
    } else if (mode == 1u) unmap(0u, allocationBytes);
    else if (mode == 2u) {
        unmap(0u, depthBytes / 2u);
        guarded();
        unmap(depthBytes / 2u, depthBytes / 2u);
        guarded();
        unmap(depthBytes, metadataBytes / 2u);
        guarded();
        unmap(depthBytes + metadataBytes / 2u, metadataBytes / 2u);
    } else {
        unmap(depthBytes / 2u, depthBytes / 2u + metadataBytes / 2u);
        guarded();
        unmap(0u, depthBytes / 2u);
        guarded();
        unmap(depthBytes + metadataBytes / 2u, metadataBytes / 2u);
    }
    RequireNoResidentDepthRead_nid_no_patch(target.address, 1u);
    RequireNoResidentDepthRead_nid_no_patch(target.htileAddress, metadataBytes);
    Require(!DepthSurfaceHolds(context, resource, firstTexture.get()) && !DepthSurfaceAt(target.address), "a fully released depth owner remains discoverable");
    if ((mode & 1u) != 0u) {
        ClearDepthSurfaces(context.device);
        TargetRefuses([&] { RequireNoMultisampleTarget(color.address, 1u, "a CPU read"); }, "multisampled");
        remove(colorMemory, 65536u);
        RequireNoMultisampleTarget(color.address, 1u, "a CPU read");
    }
    remove(memory, allocationBytes);
    add(memory, allocationBytes);
    target.clearDepth = 1.0f;
    const std::uint32_t one = 0xfffffff0u;
    for (std::size_t offset = 0u; offset < metadataBytes; offset += 4u) std::memcpy(memory.Data() + depthBytes + offset, &one, 4u);
    const auto remappedView = DepthSurfaceView(context, target);
    Require(remappedView != VK_NULL_HANDLE && ((mode & 1u) != 0u || remappedView != firstView), "a remapped allocation reused its old native depth attachment");
    auto fresh = DepthSurfaceTexture(context, words, resource, VkComponentMapping{});
    Require(fresh != nullptr && fresh->View() != firstTexture->View(), "a remapped allocation reused its old sampled view");
    recorder.Sync();
    const auto count = extent.width * extent.height * samples;
    Buffer output(context, (count + 2u) * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (const auto& [texture, expected] : {std::pair{firstTexture, 0.0f}, std::pair{fresh, 1.0f}}) {
        std::memset(output.Bytes().data(), 0xa5, output.Bytes().size());
        const Fetch fetch(context, *texture, output, FetchShader(extent.width, extent.height, samples, false, true));
        CommandBatch batch(context);
        fetch.Run(batch.Handle(), count);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        output.Invalidate();
        for (std::uint32_t i = 0u; i < count; ++i) {
            std::array<float, 4> value{};
            std::memcpy(value.data(), output.Bytes().data() + (i + 1u) * 16u, 16u);
            if (value != std::array<float, 4>{expected, 0.0f, 0.0f, 1.0f}) std::printf("Retirement sample format=%d samples=%u mode=%u index=%u expected=%g actual=(%g,%g,%g,%g)\n", format, samples, mode, i, expected, value[0], value[1], value[2], value[3]);
            Require(value == std::array<float, 4>{expected, 0.0f, 0.0f, 1.0f}, "retired or remapped depth image lost its exact contents");
        }
        Require(std::all_of(output.Bytes().begin(), output.Bytes().begin() + 16u, [](std::byte value) { return value == std::byte{0xa5}; }) && std::all_of(output.Bytes().end() - 16u, output.Bytes().end(), [](std::byte value) { return value == std::byte{0xa5}; }), "retirement fetch wrote beyond its output");
    }
}

}

int main(int argc, char** argv) {
    try {
#ifdef __linux__
        const SharedBackingScope sharedBacking;
#endif
        if (argc == 2 && std::string_view(argv[1]) == "--sync-retired-reclaim") {
            CheckIdleRetiredReclaim(true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--present-copied-writer") {
            return CheckPresentCopiedWriter() ? 0 : VulkanTestSkipped;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--present-borrow-refresh") {
            return CheckPresentBorrowRefresh() ? 0 : VulkanTestSkipped;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--idle-retired-reclaim") {
            CheckIdleRetiredReclaim();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--custom-location-htile-read") {
            CheckCustomLocationHtileRead();
            return 0;
        }
        std::unique_ptr<Device> device;
        try { device = std::make_unique<Device>(); }
        catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        CheckDrawFeedback();
        const auto context = device->GetContext();
        if (argc == 1) {
            CheckDepthAttachmentAlias();
            for (const auto samples : {2u, 4u, 8u}) {
                if (!DepthSupported(context, VK_FORMAT_D32_SFLOAT, samples)) continue;
                CheckWrittenPublication(samples, {5u, 3u}, false, 0u);
                CheckWrittenPublication(samples, {70u, 67u}, true, 0x8c00u);
            }
            CheckStorageAliasRetirement();
        }
        std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
        Require(DepthSupported(context, VK_FORMAT_D32_SFLOAT, 4u), "the native depth regression requires sampled D32 with four samples");
        if (argc == 2 && std::string_view(argv[1]) == "--clear-refusals") {
            for (const auto& [fault, expected] : std::array{
                     std::pair{ClearFault::PartialStore, "no complete tracked HTILE store"},
                     std::pair{ClearFault::CpuWrite, "CPU writes raced"},
                     std::pair{ClearFault::DepthWrite, "guest writes"},
                     std::pair{ClearFault::BadMetadata, "does not prove its clear value"},
                     std::pair{ClearFault::PossibleStore, "no complete tracked HTILE store"},
                     std::pair{ClearFault::UnprovenDispatch, "incomplete or unproven"},
                     std::pair{ClearFault::PendingWrite, "external GPU writes"},
                     std::pair{ClearFault::Overlap, "overlapping native multisample depth"}}) {
                TargetRefuses([&] { CheckDepthTarget(context, VK_FORMAT_D32_SFLOAT, 4u, false, true, false, fault, lock); }, expected);
                std::printf("Native depth clear fault=%u refused: %s\n", static_cast<std::uint32_t>(fault), expected);
            }
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--clears") {
            CheckClearDispatch();
            std::uint32_t checked = 0u, unavailable = 0u;
            for (const auto format : {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT}) {
                for (const auto samples : {2u, 4u, 8u}) {
                    if (!DepthSupported(context, format, samples)) { unavailable += 4u; continue; }
                    for (const bool depthOnly : {false, true}) {
                        for (const bool one : {false, true}) { CheckDepthTarget(context, format, samples, one, true, depthOnly, ClearFault::None, lock); ++checked; }
                    }
                }
            }
            CheckDepthTarget(context, VK_FORMAT_D32_SFLOAT, 4u, false, true, false, ClearFault::CompletedPossibleStore, lock);
            ++checked;
            CheckDepthTarget(context, VK_FORMAT_D32_SFLOAT, 4u, false, true, false, ClearFault::ImportWindowWrite, lock);
            ++checked;
            CheckDepthTarget(context, VK_FORMAT_D32_SFLOAT, 4u, false, true, false, ClearFault::DriverDispatch, lock);
            ++checked;
            std::printf("Native depth completed clear cases checked=%u unavailable=%u\n", checked, unavailable);
            return 0;
        }
        Require(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--retirement"), "invalid native depth fixture arguments");
        if (argc == 2) {
            std::uint32_t checked = 0u, unavailable = 0u;
            for (const auto format : {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT}) {
                for (const auto samples : {2u, 4u, 8u}) {
                    if (!DepthSupported(context, format, samples)) { unavailable += 4u; continue; }
                    for (std::uint32_t mode = 0u; mode < 4u; ++mode) { CheckRetirement(context, format, samples, mode); ++checked; }
                }
            }
            std::printf("Native depth retirement cases checked=%u unavailable=%u\n", checked, unavailable);
            return 0;
        }
        std::uint32_t checked = 0u, unavailable = 0u;
        for (const auto format : {VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT}) {
            for (const auto samples : {2u, 4u, 8u}) {
                if (!DepthSupported(context, format, samples)) { unavailable += 4u; continue; }
                for (const bool one : {false, true}) {
                    for (const bool iterate256 : {true, false}) { CheckDepthTarget(context, format, samples, one, false, std::nullopt, ClearFault::None, lock, iterate256); ++checked; }
                }
                CheckHtileRouting(context, format, samples);
                CheckUnwrittenGuestAccess(context, format, samples);
            }
        }
        CheckEligibility(context);
        std::printf("Native multisampled depth cases checked=%u unavailable=%u\n", checked, unavailable);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
