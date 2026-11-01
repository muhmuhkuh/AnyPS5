#pragma once

#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace AgcDriver::CpuPhaseStats {

enum class Phase : std::uint8_t {
    ResourceCapture,
    DescriptorScopedRead,
    FenceWait,
    PresentFenceWait,
    BatchCompletions,
    KeptRetirement,
    BatchRelease,
    DeferredRelease,
    ShaderRegistryLookup,
    PipelineStoreLookupBuild,
    PipelineConstruction,
    ShaderResourcesComplete,
    ShaderResourcesBuildComplete,
    ShaderResourcesRevalidate,
    ShaderResourceImageBinding,
    ShaderResourcesResolveImageTables,
    ShaderResourcesImageTableCandidateCheck,
    ShaderResourcesImageTableEntryProcessing,
    ShaderResourcesImageTableDecodeValidation,
    ShaderResourcesImageTablePendingCpuWrites,
    ShaderResourcesImageTableBindPlace,
    ShaderResourcesUploadFinish,
    ShaderResourcesBdaResources,
    ShaderResourcesFinishImageTables,
    ShaderResourcesDescriptorUpdate,
    DrawPrepare,
    DrawRecord,
    Count,
};

struct Counter {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> sumNs{0};
    std::atomic<std::uint64_t> maxNs{0};
};

struct State {
    std::array<Counter, static_cast<std::size_t>(Phase::Count)> phases{};
    std::atomic<std::int64_t> nextReportNs{0};
};

inline bool Enabled() {
    static const bool enabled = std::getenv("APS5_CPU_PHASE_STATS") != nullptr;
    return enabled;
}

inline State& Totals() {
    static State state;
    return state;
}

inline const char* Name(Phase phase) {
    constexpr std::array<const char*, static_cast<std::size_t>(Phase::Count)> names{
        "resource_capture_inclusive",
        "descriptor_scoped_read_nested",
        "recorder_fence_wait",
        "present_fence_wait",
        "batch_completion_actions",
        "kept_object_retirement",
        "recorder_batch_release",
        "deferred_object_release",
        "shader_registry_lookup",
        "graphics_pipeline_store_lookup_build_inclusive",
        "graphics_pipeline_constructor_body_nested",
        "shader_resources_complete_inclusive",
        "shader_resources_build_complete_inclusive",
        "shader_resources_revalidate_inclusive",
        "shader_resource_image_binding_nested",
        "shader_resources_resolve_image_tables_nested",
        "shader_resources_image_table_candidate_check_nested",
        "shader_resources_image_table_entry_processing_nested",
        "shader_resources_image_table_decode_validation_nested",
        "shader_resources_image_table_pending_cpu_writes_nested",
        "shader_resources_image_table_bind_validation_and_place_nested",
        "shader_resources_guest_upload_finish_nested",
        "shader_resources_bda_resources_nested",
        "shader_resources_finish_image_tables_nested",
        "shader_resources_descriptor_info_and_vk_update_nested",
        "draw_prepare_inclusive",
        "draw_record_inclusive",
    };
    return names[static_cast<std::size_t>(phase)];
}

inline void Record(Phase phase, std::uint64_t ns) noexcept {
    auto& counter = Totals().phases[static_cast<std::size_t>(phase)];
    counter.calls.fetch_add(1, std::memory_order_relaxed);
    counter.sumNs.fetch_add(ns, std::memory_order_relaxed);
    auto old = counter.maxNs.load(std::memory_order_relaxed);
    while (old < ns && !counter.maxNs.compare_exchange_weak(old, ns, std::memory_order_relaxed)) {}
}

inline void ReportIfDue() noexcept {
    if (!Enabled()) return;
    try {
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto& state = Totals();
        auto due = state.nextReportNs.load(std::memory_order_relaxed);
        if ((due != 0 && now < due) || !state.nextReportNs.compare_exchange_strong(due, now + 10'000'000'000ll, std::memory_order_relaxed)) return;
        for (std::size_t i = 0; i < state.phases.size(); ++i) {
            auto& counter = state.phases[i];
            const auto calls = counter.calls.load(std::memory_order_relaxed);
            const auto sum = counter.sumNs.load(std::memory_order_relaxed);
            const auto maximum = counter.maxNs.load(std::memory_order_relaxed);
            if (calls == 0) continue;
            AgcDriver::ProfilePrint_nid_no_patch("[cpu-phase] %s: %llu calls, %.3f s cumulative wall, %.3f ms max\n", Name(static_cast<Phase>(i)), static_cast<unsigned long long>(calls), static_cast<double>(sum) / 1e9, static_cast<double>(maximum) / 1e6);
        }
    } catch (...) {
    }
}

class ScopeTimer {
public:
    explicit ScopeTimer(Phase phase, bool condition = true, bool report = false) noexcept : phase(phase), active(Enabled() && condition), report(report) {
        if (active) {
            try {
                started = std::chrono::steady_clock::now();
            } catch (...) {
                active = false;
            }
        }
    }

    ~ScopeTimer() noexcept {
        if (!active) return;
        try {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
            Record(phase, ns > 0 ? static_cast<std::uint64_t>(ns) : 0u);
            if (report) ReportIfDue();
        } catch (...) {
        }
    }

    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

private:
    Phase phase;
    bool active;
    bool report;
    std::chrono::steady_clock::time_point started{};
};

}
