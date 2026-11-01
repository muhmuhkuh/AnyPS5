#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
#include <thread>
#include <list>
#include <map>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>

namespace AgcDriver::Graphics {

namespace {

constexpr std::array<VkDynamicState, 15> dynamicStates{
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR,
    VK_DYNAMIC_STATE_DEPTH_BIAS,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
    VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE_EXT,
    VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
    VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
    VK_DYNAMIC_STATE_STENCIL_OP_EXT,
};

using Key = std::vector<std::byte>;

struct LibraryLiveCounters {
    std::array<std::atomic<std::uint64_t>, 4> live{};
};

struct LibraryRegistry {
    std::mutex mutex;
    bool destroy = true;
    std::vector<std::weak_ptr<struct LibraryOwner>> owners;
    std::vector<std::weak_ptr<OptimizedPipeline>> optimized;
    std::shared_ptr<LibraryLiveCounters> live = std::make_shared<LibraryLiveCounters>();
};

struct PendingLibrary {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkDestroyShaderModule destroyModule = nullptr;
    VkPipeline handle = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules;

    ~PendingLibrary() {
        if (handle != VK_NULL_HANDLE) destroyPipeline(device, handle, nullptr);
        if (destroyModule != nullptr) for (const auto module : modules) if (module != VK_NULL_HANDLE) destroyModule(device, module, nullptr);
    }
};

struct LibraryOwner {
    LibraryOwner(VkDevice device, PFN_vkDestroyPipeline destroyPipeline, PFN_vkDestroyShaderModule destroyModule, VkPipeline handle, std::vector<VkShaderModule> modules, std::shared_ptr<LibraryRegistry> registry, std::shared_ptr<LibraryLiveCounters> live, std::size_t part)
        : device(device), destroyPipeline(destroyPipeline), destroyModule(destroyModule), handle(handle), modules(std::move(modules)), registry(std::move(registry)), live(std::move(live)), part(part) {
        this->live->live[this->part].fetch_add(1, std::memory_order_relaxed);
    }

    ~LibraryOwner() {
        std::lock_guard lock(registry->mutex);
        if (registry->destroy) DestroyNow();
        else {
            handle = VK_NULL_HANDLE;
            modules.clear();
            if (counted) {
                live->live[part].fetch_sub(1, std::memory_order_relaxed);
                counted = false;
            }
        }
    }

    void Abandon() noexcept {
        handle = VK_NULL_HANDLE;
        modules.clear();
        if (counted) {
            live->live[part].fetch_sub(1, std::memory_order_relaxed);
            counted = false;
        }
    }

    void DestroyNow() noexcept {
        if (handle != VK_NULL_HANDLE) destroyPipeline(device, handle, nullptr);
        if (destroyModule != nullptr) for (const auto module : modules) if (module != VK_NULL_HANDLE) destroyModule(device, module, nullptr);
        handle = VK_NULL_HANDLE;
        modules.clear();
        if (counted) {
            live->live[part].fetch_sub(1, std::memory_order_relaxed);
            counted = false;
        }
    }

    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkDestroyShaderModule destroyModule = nullptr;
    VkPipeline handle = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules;
    std::shared_ptr<LibraryRegistry> registry;
    std::shared_ptr<LibraryLiveCounters> live;
    std::size_t part = 0;
    bool counted = true;

};

struct LibraryCacheEntry {
    std::shared_ptr<LibraryOwner> owner;
    std::list<Key>::iterator lru;
};

struct DeviceLibraries {
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkDestroyPipelineLayout destroyLayout = nullptr;
    PFN_vkDestroyShaderModule destroyModule = nullptr;
    std::map<Key, VkPipelineLayout> layouts;
    std::array<std::map<Key, LibraryCacheEntry>, 4> parts;
    std::array<std::map<Key, std::weak_ptr<LibraryOwner>>, 4> retired;
    std::array<std::list<Key>, 4> lru;
    std::shared_ptr<LibraryRegistry> registry = std::make_shared<LibraryRegistry>();
    std::shared_ptr<LibraryLiveCounters> live = registry->live;
    std::shared_ptr<PipelineLibraryCounters> counters = std::make_shared<PipelineLibraryCounters>();
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct LibraryStore {
    std::mutex mutex;
    std::map<VkDevice, DeviceLibraries> devices;
    std::map<VkDevice, std::shared_ptr<LibraryRegistry>> registries;
    std::condition_variable idle;
    std::condition_variable work;
    std::deque<std::pair<VkDevice, std::function<void()>>> jobs;
    std::map<VkDevice, std::size_t> pending;
    std::map<VkDevice, std::size_t> activeBuilds;
    std::thread worker;
    bool stopping = false;
};

std::atomic<unsigned> runningWorkers{0};

LibraryStore& Libraries() {
    static auto* store = new LibraryStore();
    return *store;
}

void runJobs(LibraryStore& store) {
    std::unique_lock lock(store.mutex);
    for (;;) {
        store.work.wait(lock, [&] { return !store.jobs.empty() || store.stopping; });
        if (store.stopping) break;
        auto [device, job] = std::move(store.jobs.front());
        store.jobs.pop_front();
        lock.unlock();
        job();
        lock.lock();
        if (--store.pending[device] == 0) store.idle.notify_all();
    }
    runningWorkers.fetch_sub(1, std::memory_order_acq_rel);
}

void enqueue(LibraryStore& store, VkDevice device, std::function<void()> job) {
    if (store.stopping) return;
    if (!store.worker.joinable()) {
        store.worker = std::thread(runJobs, std::ref(store));
        runningWorkers.fetch_add(1, std::memory_order_acq_rel);
    }
    auto& pending = store.pending[device];
    store.jobs.emplace_back(device, std::move(job));
    ++pending;
    store.work.notify_one();
}

Key Combined(const Key& part, const Key& context) {
    Key key;
    const auto append = [&](const Key& value) {
        const auto size = value.size();
        const auto bytes = std::as_bytes(std::span(&size, 1));
        key.insert(key.end(), bytes.begin(), bytes.end());
        key.insert(key.end(), value.begin(), value.end());
    };
    append(part);
    append(context);
    return key;
}

}

std::span<const VkDynamicState> PipelineLibraryDynamicStates() {
    return dynamicStates;
}

VkPipeline LinkPipelineFromLibraries(const Context& context, const VkGraphicsPipelineCreateInfo& info, const VkPipelineRenderingCreateInfoKHR& rendering, const VkPipelineLayoutCreateInfo& layout, const PipelineLibraryKeys& keys, std::shared_ptr<OptimizedPipeline>* optimized, bool partsOnly, std::span<VkShaderModule> modules, std::vector<std::shared_ptr<void>>* libraryLeases) {
    Require(info.renderPass == VK_NULL_HANDLE && info.pDynamicState != nullptr, "pipeline library parts need dynamic rendering and the library dynamic states");
    const bool split = (layout.flags & VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT) != 0;
    Require(!split || (layout.setLayoutCount == 2u && layout.pSetLayouts != nullptr && ((keys.preRasterizationSets | keys.fragmentSets) & ~3u) == 0u), "split pipeline library layout requires two descriptor set slots and a valid mask");
    auto& store = Libraries();
    std::unique_lock lock(store.mutex);
    ++store.activeBuilds[context.device];
    struct ActiveBuild {
        LibraryStore& store;
        VkDevice device;
        std::unique_lock<std::mutex>& lock;
        ~ActiveBuild() {
            if (!lock.owns_lock()) lock.lock();
            if (--store.activeBuilds[device] == 0) store.idle.notify_all();
        }
    } activeBuild{store, context.device, lock};
    auto& device = store.devices[context.device];
    if (const auto registry = store.registries.find(context.device); registry != store.registries.end()) {
        device.registry = registry->second;
        device.live = device.registry->live;
    } else {
        store.registries.emplace(context.device, device.registry);
    }
    if (device.destroyPipeline == nullptr) {
        device.destroyPipeline = context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline");
        device.destroyLayout = context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
        device.destroyModule = context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    }
    static const bool profileLibraries = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool cacheStats = std::getenv("APS5_CACHE_STATS") != nullptr;
    const auto reportAt = std::chrono::steady_clock::now();
    if ((profileLibraries || cacheStats) && reportAt - device.lastReport >= std::chrono::seconds(10)) {
        device.lastReport = reportAt;
        AgcDriver::ProfilePrint_nid_no_patch("[pipeline-library-cache] vertex-input cached %zu live %llu evicted %llu; pre-raster cached %zu live %llu evicted %llu; fragment-shader cached %zu live %llu evicted %llu; fragment-output cached %zu live %llu evicted %llu; built %llu/%llu/%llu/%llu linked %llu optimized %llu\n", device.parts[0].size(), static_cast<unsigned long long>(device.live->live[0].load(std::memory_order_relaxed)), static_cast<unsigned long long>(device.counters->evicted[0]), device.parts[1].size(), static_cast<unsigned long long>(device.live->live[1].load(std::memory_order_relaxed)), static_cast<unsigned long long>(device.counters->evicted[1]), device.parts[2].size(), static_cast<unsigned long long>(device.live->live[2].load(std::memory_order_relaxed)), static_cast<unsigned long long>(device.counters->evicted[2]), device.parts[3].size(), static_cast<unsigned long long>(device.live->live[3].load(std::memory_order_relaxed)), static_cast<unsigned long long>(device.counters->evicted[3]), static_cast<unsigned long long>(device.counters->built[0]), static_cast<unsigned long long>(device.counters->built[1]), static_cast<unsigned long long>(device.counters->built[2]), static_cast<unsigned long long>(device.counters->built[3]), static_cast<unsigned long long>(device.counters->linked), static_cast<unsigned long long>(device.counters->optimized));
    }
    const auto layoutFor = [&](const Key& key, std::optional<std::uint32_t> mask) {
        auto found = device.layouts.find(key);
        if (found != device.layouts.end()) return found->second;
        auto info = layout;
        std::array<VkDescriptorSetLayout, 2> sets{VK_NULL_HANDLE, VK_NULL_HANDLE};
        if (mask.has_value()) {
            for (std::uint32_t set = 0; set < sets.size(); ++set) {
                if ((*mask & (1u << set)) != 0u) sets[set] = layout.pSetLayouts[set];
            }
            info.pSetLayouts = sets.data();
        }
        VkPipelineLayout handle = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &info, nullptr, &handle), "vkCreatePipelineLayout library");
        return device.layouts.emplace(key, handle).first->second;
    };
    const auto fullLayout = layoutFor(keys.layout, std::nullopt);
    const std::array<VkPipelineLayout, 4> partLayouts{VK_NULL_HANDLE, split ? layoutFor(keys.preRasterizationLayout, keys.preRasterizationSets) : fullLayout, split ? layoutFor(keys.fragmentLayout, keys.fragmentSets) : fullLayout, VK_NULL_HANDLE};
    std::vector<VkPipelineShaderStageCreateInfo> preRasterization;
    const VkPipelineShaderStageCreateInfo* fragment = nullptr;
    bool mesh = false;
    for (std::uint32_t index = 0; index < info.stageCount; ++index) {
        const auto& stage = info.pStages[index];
        if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) fragment = &stage;
        else preRasterization.push_back(stage);
        mesh |= stage.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    Require(fragment != nullptr && !preRasterization.empty(), "a pipeline built from libraries needs a fragment and a pre-rasterization stage");
    constexpr std::array<VkGraphicsPipelineLibraryFlagsEXT, 4> flags{
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
    };
    const std::array<const Key*, 4> partKeys{&keys.vertexInput, &keys.preRasterization, &keys.fragmentShader, &keys.fragmentOutput};
    const Key none;
    VkPipelineRenderingCreateInfoKHR shaderRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    shaderRendering.viewMask = rendering.viewMask;
    shaderRendering.depthAttachmentFormat = rendering.depthAttachmentFormat;
    shaderRendering.stencilAttachmentFormat = rendering.stencilAttachmentFormat;
    const std::array<std::uint32_t, 3> attachments{shaderRendering.viewMask, static_cast<std::uint32_t>(shaderRendering.depthAttachmentFormat), static_cast<std::uint32_t>(shaderRendering.stencilAttachmentFormat)};
    const auto attachmentBytes = std::as_bytes(std::span(attachments));
    const Key attachmentKey(attachmentBytes.begin(), attachmentBytes.end());
    const auto preRasterizationContext = Combined(keys.preRasterizationLayout, attachmentKey);
    const auto fragmentContext = Combined(keys.fragmentLayout, attachmentKey);
    const std::array<const Key*, 4> contextKeys{&none, &preRasterizationContext, &fragmentContext, &keys.renderPass};
    VkPipelineDepthStencilStateCreateInfo dynamicDepth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    static const bool backgroundLinking = std::getenv("APS5_NO_PIPELINE_LTO") == nullptr;
    const bool optimize = backgroundLinking && context.pipelineLinkTimeOptimization;
    std::vector<VkPipeline> libraries;
    std::vector<std::shared_ptr<LibraryOwner>> owners;
    const auto cacheLimit = [] {
        static const auto configured = PipelineSwitch("APS5_PIPELINE_LIBRARY_ENTRIES", 0u);
        return configured > std::numeric_limits<std::size_t>::max() ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(configured);
    }();
    Require(partsOnly || cacheLimit == 0 || libraryLeases != nullptr, "bounded pipeline library caching requires the linked pipeline to retain its library leases");
    for (std::size_t part = 0; part < flags.size(); ++part) {
        if (part == 0 && mesh) continue;
        auto key = Combined(*partKeys[part], *contextKeys[part]);
        key.push_back(static_cast<std::byte>(optimize));
        auto& cache = device.parts[part];
        auto& retired = device.retired[part];
        const auto findRetired = [&]() -> std::shared_ptr<LibraryOwner> {
            const auto found = retired.find(key);
            if (found == retired.end()) return {};
            auto owner = found->second.lock();
            if (!owner) retired.erase(found);
            return owner;
        };
        if (const auto found = cache.find(key); found != cache.end()) {
            device.lru[part].splice(device.lru[part].end(), device.lru[part], found->second.lru);
            owners.push_back(found->second.owner);
            libraries.push_back(found->second.owner->handle);
            continue;
        }
        if (auto owner = findRetired()) {
            libraries.push_back(owner->handle);
            owners.push_back(std::move(owner));
            continue;
        }
        PerformanceTimer timing("Vulkan.GraphicsPipelineLibrary");
        VkGraphicsPipelineLibraryCreateInfoEXT library{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, flags[part] == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT ? &rendering : &shaderRendering};
        library.flags = flags[part];
        VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &library};
        create.flags = optimize ? VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        create.pDynamicState = info.pDynamicState;
        create.layout = partLayouts[part];
        switch (flags[part]) {
        case VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT:
            create.pVertexInputState = info.pVertexInputState;
            create.pInputAssemblyState = info.pInputAssemblyState;
            create.layout = VK_NULL_HANDLE;
            break;
        case VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT:
            create.stageCount = static_cast<std::uint32_t>(preRasterization.size());
            create.pStages = preRasterization.data();
            create.pViewportState = info.pViewportState;
            create.pRasterizationState = info.pRasterizationState;
            create.pTessellationState = info.pTessellationState;
            break;
        case VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT:
            create.stageCount = 1;
            create.pStages = fragment;
            create.pMultisampleState = info.pMultisampleState;
            create.pDepthStencilState = info.pDepthStencilState != nullptr ? info.pDepthStencilState : &dynamicDepth;
            break;
        default:
            create.pMultisampleState = info.pMultisampleState;
            create.pColorBlendState = info.pColorBlendState;
            create.layout = VK_NULL_HANDLE;
            break;
        }
        VkPipeline handle = VK_NULL_HANDLE;
        // Built without the store's mutex: a draw whose parts are resident is not held behind
        // another thread's build, and builds from several threads (the registration prebuild, the
        // journal replay) run at once; a part another thread built meanwhile is kept, this one dropped.
        lock.unlock();
        VkResult created;
        {
            auto pipelineCacheLock = context.LockPipelineCacheForUse();
            created = context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, context.pipelineCache, 1, &create, nullptr, &handle);
        }
        lock.lock();
        Check(created, "vkCreateGraphicsPipelines library");
        if (const auto raced = cache.find(key); raced != cache.end()) {
            device.destroyPipeline(context.device, handle, nullptr);
            device.lru[part].splice(device.lru[part].end(), device.lru[part], raced->second.lru);
            owners.push_back(raced->second.owner);
            libraries.push_back(raced->second.owner->handle);
            continue;
        }
        if (auto owner = findRetired()) {
            device.destroyPipeline(context.device, handle, nullptr);
            libraries.push_back(owner->handle);
            owners.push_back(std::move(owner));
            continue;
        }
        PendingLibrary pending{context.device, device.destroyPipeline, device.destroyModule, handle, {}};
        if (optimize) {
            for (std::uint32_t stage = 0; stage < create.stageCount; ++stage) {
                for (auto& module : modules) {
                    if (module == VK_NULL_HANDLE || module != create.pStages[stage].module) continue;
                    pending.modules.push_back(module);
                    module = VK_NULL_HANDLE;
                }
            }
        }
        auto owner = std::make_shared<LibraryOwner>(context.device, device.destroyPipeline, device.destroyModule, pending.handle, pending.modules, device.registry, device.live, part);
        pending.handle = VK_NULL_HANDLE;
        pending.modules.clear();
        {
            std::lock_guard registryLock(device.registry->mutex);
            std::erase_if(device.registry->owners, [](const auto& weak) { return weak.expired(); });
            device.registry->owners.push_back(owner);
        }
        device.lru[part].push_back(key);
        const auto lru = std::prev(device.lru[part].end());
        try {
            cache.emplace(key, LibraryCacheEntry{owner, lru});
        } catch (...) {
            device.lru[part].pop_back();
            throw;
        }
        owners.push_back(owner);
        ++device.counters->built[part];
        libraries.push_back(handle);
        while (cacheLimit != 0 && cache.size() > cacheLimit) {
            const auto victimKey = device.lru[part].front();
            const auto victim = cache.find(victimKey);
            if (victim->second.owner.use_count() > 1) retired.insert_or_assign(victimKey, victim->second.owner);
            device.lru[part].pop_front();
            cache.erase(victim);
            ++device.counters->evicted[part];
            if (device.counters->evicted[part] % 128u == 0u)
                std::erase_if(retired, [](const auto& entry) { return entry.second.expired(); });
        }
    }
    if (partsOnly) return VK_NULL_HANDLE;
    if (libraryLeases != nullptr) {
        libraryLeases->clear();
        libraryLeases->reserve(owners.size());
        for (const auto& owner : owners) libraryLeases->push_back(owner);
    }
    PerformanceTimer timing("Vulkan.GraphicsPipelineLink");
    std::optional<std::pair<std::vector<VkSampleCountFlagBits>, VkSampleCountFlagBits>> sampleCounts;
    for (auto* next = static_cast<const VkBaseInStructure*>(rendering.pNext); next != nullptr; next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_NV) continue;
        const auto& counts = *reinterpret_cast<const VkAttachmentSampleCountInfoNV*>(next);
        sampleCounts.emplace(std::vector<VkSampleCountFlagBits>(counts.pColorAttachmentSamples, counts.pColorAttachmentSamples + counts.colorAttachmentCount), counts.depthStencilAttachmentSamples);
    }
    const auto attachmentSamples = [](const std::pair<std::vector<VkSampleCountFlagBits>, VkSampleCountFlagBits>& counts) {
        return VkAttachmentSampleCountInfoNV{VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_NV, nullptr, static_cast<std::uint32_t>(counts.first.size()), counts.first.empty() ? nullptr : counts.first.data(), counts.second};
    };
    auto linkedSamples = sampleCounts ? attachmentSamples(*sampleCounts) : VkAttachmentSampleCountInfoNV{};
    VkPipelineLibraryCreateInfoKHR linked{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
    linked.libraryCount = static_cast<std::uint32_t>(libraries.size());
    linked.pLibraries = libraries.data();
    VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &linked};
    if (sampleCounts) {
        linkedSamples.pNext = create.pNext;
        create.pNext = &linkedSamples;
    }
    create.layout = info.layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    lock.unlock();
    {
        auto pipelineCacheLock = context.LockPipelineCacheForUse();
        Check(context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, context.pipelineCache, 1, &create, nullptr, &pipeline), "vkCreateGraphicsPipelines link");
    }
    lock.lock();
    ++device.counters->linked;
    if (optimize && optimized != nullptr) {
        auto slot = std::make_shared<OptimizedPipeline>();
        slot->device = context.device;
        slot->destroy = device.destroyPipeline;
        for (const auto& owner : owners) slot->libraryLeases.push_back(owner);
        {
            std::lock_guard registryLock(device.registry->mutex);
            std::erase_if(device.registry->optimized, [](const auto& weak) { return weak.expired(); });
            device.registry->optimized.push_back(slot);
        }
        *optimized = slot;
        const auto createPipelines = context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
        const auto cache = context.pipelineCache;
        const auto cacheSynchronization = context.pipelineCacheSynchronization;
        const auto libraryLayout = fullLayout;
        const auto registry = device.registry;
        std::function<void()> relink = [slot, createPipelines, cache, cacheSynchronization, libraryLayout, libraries, owners, registry, counters = device.counters, sampleCounts, attachmentSamples, &store] {
            if (slot->released.load()) return;
            auto linkedSamples = sampleCounts ? attachmentSamples(*sampleCounts) : VkAttachmentSampleCountInfoNV{};
            VkPipelineLibraryCreateInfoKHR linked{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
            linked.libraryCount = static_cast<std::uint32_t>(libraries.size());
            linked.pLibraries = libraries.data();
            VkGraphicsPipelineCreateInfo create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &linked};
            if (sampleCounts) {
                linkedSamples.pNext = create.pNext;
                create.pNext = &linkedSamples;
            }
            create.flags = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
            create.layout = libraryLayout;
            VkPipeline result = VK_NULL_HANDLE;
            {
                std::lock_guard registryLock(registry->mutex);
                if (!registry->destroy || slot->released.load(std::memory_order_acquire) || slot->abandoned.load(std::memory_order_acquire)) return;
                auto pipelineCacheLock = cacheSynchronization != nullptr ? cacheSynchronization->LockForPipelineUse() : PipelineCacheSynchronization::PipelineUseLock{};
                if (createPipelines(slot->device, cache, 1, &create, nullptr, &result) != VK_SUCCESS) return;
                std::lock_guard slotLock(slot->relinkMutex);
                if (slot->released.load(std::memory_order_acquire) || slot->abandoned.load(std::memory_order_acquire)) {
                    slot->destroy(slot->device, result, nullptr);
                } else {
                    slot->handle.store(result);
                }
            }
            std::lock_guard lock(store.mutex);
            ++counters->optimized;
        };
        if (PipelineJournalReplaying()) {
            slot->relink = std::move(relink);
            slot->deferred.store(true, std::memory_order_release);
        } else {
            enqueue(store, context.device, std::move(relink));
        }
    }
    return pipeline;
}

void OptimizedPipeline::Release() noexcept {
    released.store(true, std::memory_order_release);
    std::function<void()> cancelled;
    WithPipelineLibraryDomain(libraryLeases, [&](bool alive) {
        std::lock_guard lock(relinkMutex);
        if (const auto pipeline = handle.exchange(VK_NULL_HANDLE)) {
            if (alive) destroy(device, pipeline, nullptr);
        }
        deferred.store(false, std::memory_order_release);
        cancelled = std::move(relink);
    });
    libraryLeases.clear();
}

void OptimizedPipeline::Abandon() noexcept {
    AbandonPipelineLibraryLeases(libraryLeases);
    AbandonLocal();
}

void OptimizedPipeline::AbandonLocal() noexcept {
    abandoned.store(true, std::memory_order_release);
    released.store(true, std::memory_order_release);
    std::function<void()> cancelled;
    {
        std::lock_guard lock(relinkMutex);
        handle.store(VK_NULL_HANDLE, std::memory_order_release);
        deferred.store(false, std::memory_order_release);
        cancelled = std::move(relink);
    }
    libraryLeases.clear();
}

void OptimizedPipeline::CancelDeferred() noexcept {
    std::function<void()> cancelled;
    {
        std::lock_guard lock(relinkMutex);
        deferred.store(false, std::memory_order_release);
        cancelled = std::move(relink);
    }
}

void OptimizedPipeline::Schedule() {
    auto& store = Libraries();
    std::function<void()> cancelled;
    std::lock_guard lock(store.mutex);
    {
        std::lock_guard relinkLock(relinkMutex);
        if (!deferred.exchange(false, std::memory_order_acq_rel)) return;
        if (released.load(std::memory_order_acquire)) cancelled = std::move(relink);
        else if (relink) enqueue(store, device, std::move(relink));
    }
}

void WaitForOptimizedPipelines(VkDevice device) {
    auto& store = Libraries();
    std::unique_lock lock(store.mutex);
    store.idle.wait(lock, [&] { return store.pending[device] == 0; });
}

void ClearPipelineLibraries(VkDevice device) {
    auto& store = Libraries();
    std::unique_lock lock(store.mutex);
    store.idle.wait(lock, [&] { return store.pending[device] == 0 && store.activeBuilds[device] == 0; });
    const auto found = store.devices.find(device);
    if (found == store.devices.end()) return;
    auto& libraries = found->second;
    std::vector<std::shared_ptr<OptimizedPipeline>> optimized;
    {
        std::lock_guard registryLock(libraries.registry->mutex);
        for (auto it = libraries.registry->optimized.begin(); it != libraries.registry->optimized.end();) {
            if (auto slot = it->lock()) {
                optimized.push_back(std::move(slot));
                ++it;
            } else {
                it = libraries.registry->optimized.erase(it);
            }
        }
        libraries.registry->optimized.clear();
    }
    for (const auto& slot : optimized) slot->CancelDeferred();
    for (auto& part : libraries.parts) part.clear();
    for (auto& order : libraries.lru) order.clear();
    for (const auto& [key, handle] : libraries.layouts) libraries.destroyLayout(device, handle, nullptr);
    store.devices.erase(found);
}

void AbandonPipelineLibraries(VkDevice device) {
    ClearPipelineLibraries(device);
    auto& store = Libraries();
    std::unique_lock lock(store.mutex);
    store.idle.wait(lock, [&] { return store.pending[device] == 0 && store.activeBuilds[device] == 0; });
    const auto found = store.registries.find(device);
    if (found == store.registries.end()) return;
    auto registry = found->second;
    std::vector<std::shared_ptr<LibraryOwner>> owners;
    {
        std::lock_guard registryLock(registry->mutex);
        for (const auto& weak : registry->owners) if (auto owner = weak.lock()) owners.push_back(std::move(owner));
        Require(std::all_of(owners.begin(), owners.end(), [](const auto& owner) { return owner.use_count() == 1; }), "pipeline library owners remain leased during device teardown");
        for (const auto& owner : owners) owner->DestroyNow();
        registry->owners.clear();
        registry->destroy = false;
    }
    store.registries.erase(found);
}

void AbandonPipelineLibraryLeases(std::span<const std::shared_ptr<void>> leases) {
    if (leases.empty()) return;
    auto owner = std::static_pointer_cast<LibraryOwner>(leases.front());
    auto registry = owner->registry;
    auto& store = Libraries();
    std::unique_lock lock(store.mutex);
    store.idle.wait(lock, [&] { return store.pending[owner->device] == 0 && store.activeBuilds[owner->device] == 0; });
    std::vector<std::shared_ptr<OptimizedPipeline>> optimized;
    {
        std::lock_guard registryLock(registry->mutex);
        registry->destroy = false;
        for (auto& weak : registry->optimized) if (auto slot = weak.lock()) optimized.push_back(std::move(slot));
        registry->optimized.clear();
    }
    for (const auto& slot : optimized) slot->AbandonLocal();
    std::vector<std::shared_ptr<LibraryOwner>> heldOwners;
    {
        std::lock_guard registryLock(registry->mutex);
        for (const auto& weak : registry->owners) if (auto held = weak.lock()) heldOwners.push_back(std::move(held));
        for (const auto& held : heldOwners) held->Abandon();
        registry->owners.clear();
    }
    const auto current = store.registries.find(owner->device);
    if (current != store.registries.end() && current->second == registry) store.registries.erase(current);
    const auto device = store.devices.find(owner->device);
    if (device != store.devices.end() && device->second.registry == registry) {
        for (auto& part : device->second.parts) part.clear();
        for (auto& order : device->second.lru) order.clear();
        store.devices.erase(device);
    }
}

void WithPipelineLibraryDomain(std::span<const std::shared_ptr<void>> leases, const std::function<void(bool)>& action) {
    if (leases.empty()) {
        action(true);
        return;
    }
    auto owner = std::static_pointer_cast<LibraryOwner>(leases.front());
    std::lock_guard lock(owner->registry->mutex);
    action(owner->registry->destroy);
}

void StopPipelineLibraryWorker() {
    auto& store = Libraries();
    std::thread worker;
    {
        std::lock_guard lock(store.mutex);
        store.stopping = true;
        for (const auto& [device, job] : store.jobs) --store.pending[device];
        store.jobs.clear();
        worker = std::move(store.worker);
    }
    store.idle.notify_all();
    store.work.notify_all();
    if (!worker.joinable()) return;
    if (worker.get_id() == std::this_thread::get_id()) worker.detach();
    else worker.join();
}

unsigned PipelineLibraryWorkers() {
    return runningWorkers.load(std::memory_order_acquire);
}

PipelineLibraryCounters PipelineLibraryCountersOf(VkDevice device) {
    auto& store = Libraries();
    std::lock_guard lock(store.mutex);
    const auto found = store.devices.find(device);
    if (found == store.devices.end()) return {};
    auto counters = *found->second.counters;
    for (std::size_t part = 0; part < counters.cached.size(); ++part) {
        counters.cached[part] = found->second.parts[part].size();
        counters.live[part] = found->second.live->live[part].load(std::memory_order_relaxed);
    }
    return counters;
}

}
