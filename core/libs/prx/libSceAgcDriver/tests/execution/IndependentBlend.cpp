#define SDL_MAIN_HANDLED
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Cover_vert_spv.h"
#include "IndependentBlend_spv.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace AgcDriver::Graphics;
constexpr int VulkanTestSkipped = 77;
constexpr std::uint32_t Width = 8;
constexpr std::uint32_t Height = 8;
constexpr std::size_t Unit = 65536;
constexpr std::array<float, 4> Initial{0.125f, 0.25f, 0.125f, 0.25f};
alignas(4096) std::array<std::byte, 3 * Unit> Storage{};

class Window {
public:
    Window() {
        Require(SDL_Init(SDL_INIT_VIDEO) == 0, std::string("SDL_Init: ") + SDL_GetError());
        window = SDL_CreateWindow("AnyPS5 independent blend regression", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 32, 32, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        Require(window != nullptr, std::string("SDL_CreateWindow: ") + SDL_GetError());
        unsigned count = 0;
        Require(SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr), "cannot count SDL Vulkan extensions");
        extensions.resize(count);
        Require(SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data()), "cannot get SDL Vulkan extensions");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        descriptor = {this, extensions, [](void* opaque, VkInstance instance) {
            auto& owner = *static_cast<Window*>(opaque);
            owner.instance = instance;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            Require(SDL_Vulkan_CreateSurface(owner.window, instance, &surface), "cannot create the hidden test surface");
            return surface;
        }, [](void* opaque, std::uint32_t* width, std::uint32_t* height) {
            int x = 0, y = 0;
            SDL_Vulkan_GetDrawableSize(static_cast<Window*>(opaque)->window, &x, &y);
            *width = static_cast<std::uint32_t>(x);
            *height = static_cast<std::uint32_t>(y);
        }, 32, 32, {}};
    }

    ~Window() {
        if (window != nullptr) SDL_DestroyWindow(window);
        SDL_Quit();
    }

    PFN_vkVoidFunction Function(const char* name) const {
        const auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
        Require(get != nullptr, "cannot find vkGetInstanceProcAddr");
        const auto function = get(instance, name);
        Require(function != nullptr, std::string("missing instance function: ") + name);
        return function;
    }

    void RequireValidationLayer() const {
        const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(Function("vkEnumerateInstanceLayerProperties"));
        std::uint32_t count = 0;
        Check(enumerate(&count, nullptr), "vkEnumerateInstanceLayerProperties count");
        std::vector<VkLayerProperties> layers(count);
        Check(enumerate(&count, layers.data()), "vkEnumerateInstanceLayerProperties");
        Require(std::ranges::any_of(layers, [](const VkLayerProperties& layer) { return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0; }), "VK_LAYER_KHRONOS_validation is unavailable");
        const auto* configured = std::getenv("VK_INSTANCE_LAYERS");
        const std::string_view names = configured != nullptr ? configured : "";
        bool enabled = false;
        for (std::size_t first = 0; first < names.size();) {
            const auto last = names.find_first_of(":;", first);
            enabled = enabled || names.substr(first, last == std::string_view::npos ? last : last - first) == "VK_LAYER_KHRONOS_validation";
            if (last == std::string_view::npos) break;
            first = last + 1;
        }
        Require(enabled, "VK_INSTANCE_LAYERS must enable VK_LAYER_KHRONOS_validation");
    }

    AgcDriver::PresentationWindow descriptor{};
    VkInstance instance = VK_NULL_HANDLE;
private:
    SDL_Window* window = nullptr;
    std::vector<const char*> extensions;
};

class Validation {
public:
    explicit Validation(const Window& window) : window(window) {
        VkDebugUtilsMessengerCreateInfoEXT create{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        create.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create.pUserData = &errors;
        create.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void* opaque) -> VkBool32 {
            static_cast<std::atomic<unsigned>*>(opaque)->fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "independent-blend validation: %s\n", data->pMessage);
            return VK_FALSE;
        };
        Check(reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(window.Function("vkCreateDebugUtilsMessengerEXT"))(window.instance, &create, nullptr, &messenger), "vkCreateDebugUtilsMessengerEXT");
    }
    ~Validation() {
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(window.Function("vkDestroyDebugUtilsMessengerEXT"))(window.instance, messenger, nullptr);
    }
    void CheckErrors() const { Require(errors.load() == 0, "independent-blend draw produced Vulkan validation errors"); }
private:
    const Window& window;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    std::atomic<unsigned> errors{0};
};

State MakeState() {
    const auto base = (reinterpret_cast<std::uintptr_t>(Storage.data()) + Unit - 1) & ~std::uintptr_t{Unit - 1};
    const ColorTargetLayout layout(Width, Height, ColorTileMode::Linear, 16);
    State state{};
    for (unsigned index = 0; index < 2; ++index) {
        ColorTarget color{};
        color.address = base + index * Unit;
        color.extent = {Width, Height};
        color.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        color.bytes = layout.Bytes();
        color.componentMapping = 0xe4u;
        color.tileMode = ColorTileMode::Linear;
        color.elementBytes = 16;
        color.surfaceAddress = color.address;
        color.surfaceExtent = color.extent;
        color.exportIndex = index;
        state.colors.push_back(color);
        auto& blend = state.blends.emplace_back();
        blend.colorWriteMask = index == 0 ? VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_B_BIT : 15u;
        blend.blendEnable = index == 1;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        std::vector<std::byte> initial(layout.Bytes());
        for (std::size_t at = 0; at < initial.size(); at += sizeof(Initial)) std::memcpy(initial.data() + at, Initial.data(), sizeof(Initial));
        AgcDriver::GuestMemory::Write(color.address, initial);
    }
    state.color = state.colors.front();
    state.blend = state.blends.front();
    state.hasColorTarget = true;
    state.stages = {ShaderPath::Vertex, 0u, 64u, 64u, std::nullopt, std::nullopt};
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 0, static_cast<float>(Width), static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    return state;
}

template<class TContext>
void Unsupported(TContext context, const State& state, std::span<const CompiledShader> shaders) {
    if constexpr (requires { context.independentBlend; }) {
        context.independentBlend = false;
        VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        Check(context.template Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &create, nullptr, &layout), "vkCreateDescriptorSetLayout");
        try {
            const auto reject = [&](const State& candidate) {
                std::string failure;
                try { const Pipeline pipeline(context, candidate, VertexInputLayout{}, PipelineSetLayouts{layout}, shaders); }
                catch (const std::runtime_error& error) { failure = error.what(); }
                Require(failure.find("independentBlend") != std::string::npos, "distinct attachment states were not refused without the device feature");
            };
            reject(state);
            State alternate = state;
            for (auto& color : alternate.colors) { color.format = VK_FORMAT_R8G8_UNORM; color.elementBytes = 2; }
            alternate.colors.front().componentMapping = 0xecu;
            alternate.color = alternate.colors.front();
            alternate.blends.assign(2, state.blends.back());
            for (auto& blend : alternate.blends) {
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            }
            alternate.blend = alternate.blends.front();
            reject(alternate);
            State uniform = state;
            uniform.blends.assign(2, state.blends.front());
            const Pipeline allowed(context, uniform, VertexInputLayout{}, PipelineSetLayouts{layout}, shaders);
        } catch (...) {
            context.template Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, layout, nullptr);
            throw;
        }
        context.template Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, layout, nullptr);
    } else Require(false, "the Context cannot represent independentBlend support");
}
}

int main() {
    try {
        std::unique_ptr<Window> window;
        std::unique_ptr<AgcDriver::VulkanDevice> device;
        std::unique_ptr<Validation> validation;
        try {
            window = std::make_unique<Window>();
            window->RequireValidationLayer();
            device = std::make_unique<AgcDriver::VulkanDevice>(&window->descriptor);
            validation = std::make_unique<Validation>(*window);
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan validation device: %s\n", error.what());
            return VulkanTestSkipped;
        }
        auto context = device->graphicsContext();
        Require(!context.graphicsPipelineLibrary, "this regression requires APS5_NO_GPL=1");
        VkPhysicalDeviceFeatures available{};
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(window->Function("vkGetPhysicalDeviceFeatures"))(context.physical, &available);
        if (available.independentBlend != VK_TRUE) {
            Require(std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr, "the pixel regression needs hardware independentBlend support");
            std::puts("skipped, independentBlend is unavailable");
            return VulkanTestSkipped;
        }
        std::printf("independent blend Vulkan device: %s; legacy pipeline; VK_LAYER_KHRONOS_validation enabled\n", device->DeviceName().c_str());
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(Storage.data(), Storage.size(), true, true, true);
        }
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto state = MakeState();
            ShaderRecompiler::RecompileResult vertex, fragment;
            vertex.spirv = std::vector<std::uint32_t>(std::begin(COVER_VERT_SPV), std::end(COVER_VERT_SPV));
            fragment.spirv = std::vector<std::uint32_t>(std::begin(INDEPENDENT_BLEND_SPV), std::end(INDEPENDENT_BLEND_SPV));
            const std::array<CompiledShader, 2> shaders{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}}};
            device->Draw(state, {0, 3, 0, 1, 0, false}, shaders);
            device->WaitIdle();
            const std::array<std::array<float, 4>, 2> expected{{{0.25f, 0.25f, 0.75f, 0.25f}, {0.375f, 0.75f, 0.875f, 0.75f}}};
            for (std::size_t index = 0; index < state.colors.size(); ++index) {
                const auto& color = state.colors[index];
                StorageTexture::FlushPending(color.address, color.bytes, nullptr, "independent blend readback");
                device->WaitIdle();
                const ColorTargetLayout layout(Width, Height, color.tileMode, color.elementBytes);
                for (std::uint32_t y = 0; y < Height; ++y) for (std::uint32_t x = 0; x < Width; ++x) {
                    std::array<float, 4> pixel{};
                    std::memcpy(pixel.data(), reinterpret_cast<const std::byte*>(color.address) + layout.Offset(x, y), sizeof(pixel));
                    Require(pixel == expected[index], "independent blending or attachment write masks produced wrong pixels");
                }
            }
            validation->CheckErrors();
            Unsupported(context, state, shaders);
            validation->CheckErrors();
        }
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(Storage.data());
        }
        std::puts("independent blend pixels, validation and unsupported-feature controls passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
