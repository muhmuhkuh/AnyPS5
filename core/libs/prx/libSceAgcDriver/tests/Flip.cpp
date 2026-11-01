#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/PresentMode.hpp"
#include "prx/libSceAgcDriver/Execution/include/SwapchainImages.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define APS5_DUP _dup
#define APS5_DUP2 _dup2
#define APS5_CLOSE _close
#define APS5_FILENO _fileno
#else
#include <unistd.h>
#define APS5_DUP dup
#define APS5_DUP2 dup2
#define APS5_CLOSE close
#define APS5_FILENO fileno
#endif

namespace AgcDriver::DriverDetail {
struct DriverTestAccess {
    static void SetDevice(std::shared_ptr<VulkanDevice> device) { Driver::Get().device = std::move(device); }
    static void RememberReplaced(const std::shared_ptr<VulkanDevice>& device) { Driver::Get().replacedDevices.push_back(device); }
    static std::uint32_t OpenFlipWork() { return Driver::Get().flipWorkOpen.load(std::memory_order_acquire); }
};
}

namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}

namespace {

void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

template<typename TAction>
std::string expectFailure(TAction action) {
    try { action(); }
    catch (const std::runtime_error& error) { return error.what(); }
    throw std::runtime_error("expected an exception");
}

struct State {
    std::mutex mutex;
    std::condition_variable changed;
    bool block = false;
    bool entered = false;
    bool fail = false;
    bool checkSelfWait = false;
    bool gate = false;
    bool waiting = false;
    std::atomic<int> alive = 0;
    std::atomic<int> ready = 0;
    std::atomic<int> failed = 0;
    AgcDriver::FlipInfo last{};
};

class FlipRelease final {
public:
    explicit FlipRelease(std::shared_ptr<State> state) : state(std::move(state)) {}
    ~FlipRelease() { Release(); }
    void Release() {
        {
            std::lock_guard lock(state->mutex);
            state->block = false;
        }
        state->changed.notify_all();
    }
private:
    std::shared_ptr<State> state;
};

class Request final : public AgcDriver::IFlipRequest {
public:
    explicit Request(std::shared_ptr<State> value) : state(std::move(value)) { ++state->alive; }
    ~Request() override { --state->alive; }
    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override {
        std::unique_lock lock(state->mutex);
        if (state->checkSelfWait) {
            check(expectFailure([] { AgcDriverSuspendPoint_nid_postfix(); }).find("itself") != std::string::npos, "self suspend was not rejected");
            check(expectFailure([] { AgcDriverWaitIdle_nid_postfix(); }).find("itself") != std::string::npos, "self wait was not rejected");
        }
        state->entered = true;
        state->changed.notify_all();
        state->changed.wait(lock, [&] { return !state->block; });
        if (state->fail) throw std::runtime_error("intentional flip failure");
        ++state->ready;
    }
    void Fail(std::exception_ptr error) noexcept override {
        if (!error) std::terminate();
        ++state->failed;
    }
private:
    std::shared_ptr<State> state;
};

class RenderingGate final : public AgcDriver::IRenderingWait {
public:
    explicit RenderingGate(std::shared_ptr<State> value) : state(std::move(value)) {}
    void Wait() override {
        std::unique_lock lock(state->mutex);
        state->waiting = true;
        state->changed.notify_all();
        state->changed.wait(lock, [&] { return !state->gate; });
    }
private:
    std::shared_ptr<State> state;
};

class Output final : public AgcDriver::IVideoOutput {
public:
    void Fail(std::exception_ptr error) noexcept override { if (!error) std::terminate(); }
    std::shared_ptr<State> state = std::make_shared<State>();
    std::shared_ptr<AgcDriver::IRenderingWait> CaptureRenderingWait(std::uint32_t) override { return std::make_shared<RenderingGate>(state); }
    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& info) override {
        std::lock_guard lock(state->mutex);
        state->last = info;
        return std::make_shared<Request>(state);
    }
};

void submitFlip(std::uint32_t handle = 7) {
    std::array<std::uint32_t, 6> words{0xc004105c, handle, 0xfffffffeu, 1, 0x76543211u, 0xfedcba98u};
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "flip submission failed");
}

void testFlipAndBoundary() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    expectFailure([&] { AgcDriverRegisterVideoOutput_nid_postfix(7, output); });
    std::array<std::uint32_t, 6> words{0xc004105c, 7, 0xfffffffeu, 1, 0, 0};
    Packet packet{words.data(), 6, 0, {}};
    words[0] = 0xc004105d;
    expectFailure([&] { sceAgcDriverSubmitDcb(&packet); });
    words[0] = 0xc004105c;
    packet.dw_num = 5;
    expectFailure([&] { sceAgcDriverSubmitDcb(&packet); });
    expectFailure([] { submitFlip(8); });
    std::array<std::uint32_t, 12> rollback{0xc004105c, 7, 0, 1, 0, 0, 0xc004105c, 8, 0, 1, 0, 0};
    Packet rejected{rollback.data(), 12, 0, {}};
    expectFailure([&] { sceAgcDriverSubmitDcb(&rejected); });
    check(output->state->alive == 0 && output->state->ready == 0, "rejected submission retained or executed a reservation");
    alignas(4) static std::uint32_t condition = 1;
    const auto conditionAddress = reinterpret_cast<std::uintptr_t>(&condition);
    std::array<std::uint32_t, 11> guarded{0xc0032200, static_cast<std::uint32_t>(conditionAddress), static_cast<std::uint32_t>(conditionAddress >> 32u), 0, 6, 0xc004105c, 7, 0, 1, 0, 0};
    Packet conditionalFlip{guarded.data(), 11, 0, {}};
    check(expectFailure([&] { sceAgcDriverSubmitDcb(&conditionalFlip); }).find("a flip inside a conditional execution range") != std::string::npos, "a flip inside a COND_EXEC range was not rejected");
    check(output->state->alive == 0 && output->state->ready == 0, "a rejected conditional flip retained or executed a reservation");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->block = true;
        output->state->checkSelfWait = true;
    }
    std::future<void> boundary;
    FlipRelease release(output->state);
    submitFlip();
    {
        std::unique_lock lock(output->state->mutex);
        check(output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->entered; }), "worker did not reach flip");
        check(output->state->last.argument == -0x123456789abcdefLL && output->state->last.index == -2, "decoded flip arguments changed");
    }
    boundary = std::async(std::launch::async, [] { AgcDriverSuspendPoint_nid_postfix(); });
    check(boundary.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "suspend blocked on preceding work");
    boundary.get();
    check(output->state->ready == 0, "blocked flip completed before release");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
    auto replacement = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, replacement);
    auto next = std::async(std::launch::async, [] { submitFlip(); });
    const bool held = next.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout;
    release.Release();
    const bool advanced = next.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (advanced) next.get();
    check(held && advanced, "the next flip did not wait for the previous submission");
    AgcDriverWaitIdle_nid_postfix();
    check(output->state->ready == 1 && replacement->state->ready == 1, "registration lifetime or FIFO was lost");
    check(output->state->failed == 0, "successful request failed");
    std::vector<std::thread> producers;
    std::array<std::exception_ptr, 4> errors{};
    for (std::size_t i = 0; i < errors.size(); ++i) {
        producers.emplace_back([&, i] {
            try {
                for (int j = 0; j < 50; ++j) {
                    submitFlip();
                    if (j % 5 == 0) AgcDriverSuspendPoint_nid_postfix();
                }
            } catch (...) { errors[i] = std::current_exception(); }
        });
    }
    for (auto& producer : producers) producer.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    AgcDriverSuspendPoint_nid_postfix();
    AgcDriverWaitIdle_nid_postfix();
    check(replacement->state->ready == 201, "concurrent submissions were lost");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, replacement);
}

void testFlipHold() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = true;
    }
    struct GateRelease {
        std::shared_ptr<State> state;
        ~GateRelease() {
            {
                std::lock_guard lock(state->mutex);
                state->gate = false;
            }
            state->changed.notify_all();
        }
    } gateRelease{output->state};
    std::array<std::uint32_t, 10> gated{0xc0021018, 7, 0, 0, 0xc004105c, 7, 0xfffffffeu, 1, 0, 0};
    Packet gatedFrame{gated.data(), static_cast<std::uint32_t>(gated.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&gatedFrame) == 0, "gated flip submission failed");
    {
        std::unique_lock lock(output->state->mutex);
        check(output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->waiting; }), "worker did not reach the rendering wait");
    }
    auto held = std::async(std::launch::async, [] { submitFlip(); });
    check(held.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout, "a flip was accepted before the worker reached the previous one");
    check(output->state->ready == 0, "a flip passed a closed rendering wait");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = false;
    }
    output->state->changed.notify_all();
    check(held.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "a flip stayed held after the worker reached the previous one");
    held.get();
    AgcDriverWaitIdle_nid_postfix();
    check(output->state->ready == 2, "held flips were lost");
    alignas(4) static std::atomic<std::uint32_t> label = 0;
    struct LabelRelease {
        ~LabelRelease() { label.store(1, std::memory_order_release); }
    } labelRelease;
    const auto labelAddress = reinterpret_cast<std::uintptr_t>(&label);
    std::array<std::uint32_t, 13> waiting{0xc0053c00, 0x13, static_cast<std::uint32_t>(labelAddress), static_cast<std::uint32_t>(static_cast<std::uint64_t>(labelAddress) >> 32u), 1, 0xffffffffu, 0x19, 0xc004105c, 7, 0xfffffffeu, 1, 0, 0};
    Packet waitingFrame{waiting.data(), static_cast<std::uint32_t>(waiting.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&waitingFrame) == 0, "waiting flip submission failed");
    auto released = std::async(std::launch::async, [] { submitFlip(); });
    check(released.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "a flip was held while the worker waited for a label only the title stores");
    released.get();
    check(output->state->ready == 2, "a flip passed an unsatisfied wait");
    label.store(1, std::memory_order_release);
    AgcDriverWaitIdle_nid_postfix();
    check(output->state->ready == 4, "flips behind a title-stored label were lost");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = true;
        output->state->waiting = false;
    }
    alignas(4) static std::atomic<std::uint32_t> later = 0;
    struct LaterRelease {
        ~LaterRelease() { later.store(1, std::memory_order_release); }
    } laterRelease;
    const auto laterAddress = reinterpret_cast<std::uintptr_t>(&later);
    std::array<std::uint32_t, 17> gatedWait{0xc0021018, 7, 0, 0, 0xc0053c00, 0x13, static_cast<std::uint32_t>(laterAddress), static_cast<std::uint32_t>(static_cast<std::uint64_t>(laterAddress) >> 32u), 1, 0xffffffffu, 0x19, 0xc004105c, 7, 0xfffffffeu, 1, 0, 0};
    Packet gatedWaitFrame{gatedWait.data(), static_cast<std::uint32_t>(gatedWait.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&gatedWaitFrame) == 0, "gated waiting flip submission failed");
    {
        std::unique_lock lock(output->state->mutex);
        check(output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->waiting; }), "worker did not reach the second rendering wait");
    }
    auto progressed = std::async(std::launch::async, [] { submitFlip(); });
    check(progressed.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout, "a flip was accepted while the worker was held before a label wait");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = false;
    }
    output->state->changed.notify_all();
    check(progressed.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "a flip stayed held after the worker went on to wait for a label only the title stores");
    progressed.get();
    check(output->state->ready == 4, "a flip passed an unsatisfied wait after the rendering wait");
    later.store(1, std::memory_order_release);
    AgcDriverWaitIdle_nid_postfix();
    check(output->state->ready == 6, "flips behind a later title-stored label were lost");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

void testFlipHoldCoversTrailingLabel() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = true;
        output->state->waiting = false;
    }
    alignas(8) static std::atomic<std::uint32_t> label = 0;
    label.store(0, std::memory_order_release);
    const auto address = reinterpret_cast<std::uintptr_t>(&label);
    std::array<std::uint32_t, 15> words{0xc004105c, 7, 0xfffffffeu, 1, 0, 0, 0xc0021018, 7, 0, 0, 0xc0033700, 0x04100200, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(static_cast<std::uint64_t>(address) >> 32u), 1};
    Packet frame{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&frame) == 0, "flip submission with trailing label failed");
    bool waiting = false;
    {
        std::unique_lock lock(output->state->mutex);
        waiting = output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->waiting; });
    }
    auto next = std::async(std::launch::async, [] { submitFlip(); });
    const auto early = next.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    const auto earlyLabel = label.load(std::memory_order_acquire);
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = false;
    }
    output->state->changed.notify_all();
    const auto completed = next.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (completed) next.get();
    AgcDriverWaitIdle_nid_postfix();
    const auto finalLabel = label.load(std::memory_order_acquire);
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
    check(waiting, "worker did not reach the rendering wait after flip");
    check(!early && earlyLabel == 0, "next flip returned before the trailing label");
    check(completed && finalLabel == 1, "trailing label or next flip did not complete");
}

void testFlipWaitsForGpuCompletion() {
    using namespace AgcDriver;
    auto device = std::make_shared<VulkanDevice>();
    check(device->CanWaitUnlocked(), "GPU timeline support is required for the completion test");
    const auto context = VulkanDeviceTestAccess::ContextOf(*device);
    const auto createSemaphore = context.Function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    const auto destroySemaphore = context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    auto signalSemaphore = reinterpret_cast<PFN_vkSignalSemaphore>(context.deviceProc(context.device, "vkSignalSemaphoreKHR"));
    if (signalSemaphore == nullptr) signalSemaphore = context.Function<PFN_vkSignalSemaphore>("vkSignalSemaphore");
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    check(createSemaphore(context.device, &info, nullptr, &semaphore) == VK_SUCCESS, "cannot create GPU gate");
    struct Gate {
        VkDevice device;
        VkSemaphore semaphore;
        PFN_vkSignalSemaphore signal;
        PFN_vkDestroySemaphore destroy;
        VulkanDevice& gpuDevice;
        std::atomic<bool> released{false};
        void Release() {
            if (released.exchange(true)) return;
            VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            info.semaphore = semaphore;
            info.value = 1;
            if (signal(device, &info) != VK_SUCCESS) std::terminate();
        }
        ~Gate() {
            Release();
            try { std::lock_guard gpu(GuestMemory::GpuMutex()); gpuDevice.WaitIdle(); } catch (...) {}
            destroy(device, semaphore, nullptr);
        }
    } gate{context.device, semaphore, signalSemaphore, destroySemaphore, *device};
    DriverDetail::DriverTestAccess::SetDevice(device);
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    {
        std::lock_guard lock(output->state->mutex);
        output->state->block = true;
    }
    FlipRelease release(output->state);
    submitFlip();
    {
        std::unique_lock lock(output->state->mutex);
        check(output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->entered; }), "worker did not reach GPU test flip");
    }
    auto buffer = std::make_shared<Graphics::Buffer>(context, 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::weak_ptr<Graphics::Buffer> retained = buffer;
    std::atomic<std::uint32_t> callbackValue{0};
    Graphics::Recorder* recorder = nullptr;
    std::uint64_t expectedSerial = 0;
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        recorder = Graphics::Recorder::Active();
        check(recorder != nullptr, "active GPU recorder missing");
        const auto commands = recorder->Commands();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, buffer->Handle(), 0, 4, 0x12345678u);
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder->Keep(buffer);
        check(recorder->AfterRecordedWork([retained, &callbackValue] {
            if (const auto data = retained.lock()) {
                std::uint32_t value = 0;
                std::memcpy(&value, data->Bytes().data(), sizeof(value));
                callbackValue.store(value, std::memory_order_release);
            }
        }), "GPU completion callback was not recorded");
        recorder->HoldNextSubmit(semaphore, 1);
        expectedSerial = recorder->OpenBatchSerial();
    }
    buffer.reset();
    std::atomic<bool> watchdogFired{false};
    std::jthread watchdog([&](std::stop_token stop) {
        for (int i = 0; i < 300 && !stop.stop_requested(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!stop.stop_requested()) { watchdogFired.store(true); gate.Release(); }
    });
    auto next = std::async(std::launch::async, [] { submitFlip(); });
    release.Release();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (recorder->NewestSubmitted() < expectedSerial && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool submitted = recorder->NewestSubmitted() >= expectedSerial;
    auto cpuFlip = std::async(std::launch::async, [] { AgcDriverSubmitFlip_nid_postfix({7, -2, 1, 0}); });
    const bool cpuAdvanced = cpuFlip.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    const bool held = next.wait_for(std::chrono::milliseconds(250)) == std::future_status::timeout;
    const bool ownerHeld = !retained.expired();
    const bool callbackPending = callbackValue.load(std::memory_order_acquire) == 0;
    gate.Release();
    watchdog.request_stop();
    watchdog.join();
    cpuFlip.get();
    const bool advanced = next.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (advanced) next.get();
    AgcDriverWaitIdle_nid_postfix();
    const auto value = callbackValue.load(std::memory_order_acquire);
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
    DriverDetail::DriverTestAccess::SetDevice(nullptr);
    check(submitted && cpuAdvanced && held && ownerHeld && callbackPending && !watchdogFired.load(), "GPU or CPU flip ordering around the unsignaled fence failed");
    check(advanced && value == 0x12345678u, "flip returned before GPU write and completion callback");
}

void testFlipDeviceReplacement() {
    using namespace AgcDriver;
    auto oldDevice = std::make_shared<VulkanDevice>();
    check(oldDevice->CanWaitUnlocked(), "old GPU timeline support is required");
    const auto context = VulkanDeviceTestAccess::ContextOf(*oldDevice);
    const auto createSemaphore = context.Function<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    const auto destroySemaphore = context.Function<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    auto signalSemaphore = reinterpret_cast<PFN_vkSignalSemaphore>(context.deviceProc(context.device, "vkSignalSemaphoreKHR"));
    if (signalSemaphore == nullptr) signalSemaphore = context.Function<PFN_vkSignalSemaphore>("vkSignalSemaphore");
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    check(createSemaphore(context.device, &info, nullptr, &semaphore) == VK_SUCCESS, "cannot create replacement gate");
    DriverDetail::DriverTestAccess::SetDevice(oldDevice);
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    std::uint64_t oldSerial = 0;
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        auto* recorder = Graphics::Recorder::Active();
        check(recorder != nullptr, "old recorder missing");
        for (int i = 0; i < 48; ++i) {
            recorder->Commands();
            oldSerial = oldDevice->SubmitAndEpoch();
        }
        oldDevice->WaitIdle();
        recorder->Commands();
        recorder->HoldNextSubmit(semaphore, 1);
        oldSerial = recorder->OpenBatchSerial();
    }
    submitFlip();
    auto* oldRecorder = Graphics::Recorder::Active();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (oldRecorder->NewestSubmitted() < oldSerial && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool submitted = oldRecorder->NewestSubmitted() >= oldSerial;
    while (DriverDetail::DriverTestAccess::OpenFlipWork() != 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool published = DriverDetail::DriverTestAccess::OpenFlipWork() == 0;
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = semaphore;
    signal.value = 1;
    check(signalSemaphore(context.device, &signal) == VK_SUCCESS, "cannot release replacement gate");
    AgcDriverWaitIdle_nid_postfix();
    destroySemaphore(context.device, semaphore, nullptr);
    std::shared_ptr<VulkanDevice> newDevice;
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        oldDevice->PrepareForReplacement();
        DriverDetail::DriverTestAccess::RememberReplaced(oldDevice);
        newDevice = std::make_shared<VulkanDevice>();
        DriverDetail::DriverTestAccess::SetDevice(newDevice);
    }
    check(newDevice->CanWaitUnlocked(), "new GPU timeline support is required");
    auto* recorder = Graphics::Recorder::Active();
    check(recorder != nullptr, "new recorder missing");
    const auto newSerial = recorder->NewestSubmitted();
    auto next = std::async(std::launch::async, [] { submitFlip(); });
    const bool advanced = next.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    if (!advanced) {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        while (recorder->NewestSubmitted() < oldSerial) {
            recorder->Commands();
            newDevice->SubmitAndEpoch();
        }
    }
    const bool recovered = next.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (recovered) next.get();
    AgcDriverWaitIdle_nid_postfix();
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        DriverDetail::DriverTestAccess::SetDevice(nullptr);
        newDevice.reset();
        oldDevice.reset();
    }
    check(submitted && published && oldSerial > newSerial, "replacement fixture did not separate the device serials");
    check(advanced && recovered, "next flip waited on the old serial of the replacement device");
}

void testFlipShutdown() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = true;
    }
    std::array<std::uint32_t, 10> words{0xc004105c, 7, 0xfffffffeu, 1, 0, 0, 0xc0021018, 7, 0, 0};
    Packet frame{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&frame) == 0, "shutdown flip submission failed");
    bool waiting = false;
    {
        std::unique_lock lock(output->state->mutex);
        waiting = output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->waiting; });
    }
    auto next = std::async(std::launch::async, [] {
        try { submitFlip(); return std::string{}; }
        catch (const ProcessShutdown&) { return std::string("shutdown"); }
        catch (const std::exception& error) { return std::string(error.what()); }
    });
    const bool held = next.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout;
    auto shutdown = std::async(std::launch::async, [] { LibcRunShutdown_nid_postfix(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = false;
    }
    output->state->changed.notify_all();
    const bool nextDone = next.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    const bool shutdownDone = shutdown.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    const auto nextResult = nextDone ? next.get() : std::string("timeout");
    if (shutdownDone) shutdown.get();
    check(waiting && held && nextDone && shutdownDone, "shutdown did not release held flip work");
    check(nextResult == "shutdown" || nextResult.find("stopping") != std::string::npos, "shutdown did not cancel the held flip");
}

void testComputeRenderingWait() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    std::array<std::uint32_t, 4> wait{0xc0021018, 8, 0, 0};
    Packet packet{wait.data(), 4, 0, {}};
    check(expectFailure([&] { sceAgcDriverSubmitAcb(0x20, &packet); }).find("unregistered video output") != std::string::npos, "a compute rendering wait on an unregistered output was accepted");
    wait = {0xc0021018, 7, 0, 1};
    check(expectFailure([&] { sceAgcDriverSubmitAcb(0x20, &packet); }).find("rendering wait mode") != std::string::npos, "a compute rendering wait with a mode was accepted");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = true;
    }
    struct GateRelease {
        std::shared_ptr<State> state;
        ~GateRelease() {
            {
                std::lock_guard lock(state->mutex);
                state->gate = false;
            }
            state->changed.notify_all();
        }
    } gateRelease{output->state};
    wait = {0xc0021018, 7, 0, 0};
    check(sceAgcDriverSubmitAcb(0x20, &packet) == 0, "compute rendering wait submission failed");
    {
        std::unique_lock lock(output->state->mutex);
        check(output->state->changed.wait_for(lock, std::chrono::seconds(5), [&] { return output->state->waiting; }), "compute worker did not reach the rendering wait");
    }
    auto idle = std::async(std::launch::async, [] { AgcDriverWaitIdle_nid_postfix(); });
    check(idle.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout, "the driver went idle past a closed compute rendering wait");
    {
        std::lock_guard lock(output->state->mutex);
        output->state->gate = false;
    }
    output->state->changed.notify_all();
    check(idle.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "a compute rendering wait stayed blocked after release");
    idle.get();
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

void testComputeFlip() {
    auto output = std::make_shared<Output>();
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    std::array<std::uint32_t, 6> words{0xc004105c, 7, 0xfffffffeu, 1, 0x76543211u, 0xfedcba98u};
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitAcb(0x20, &packet) == 0, "compute flip submission failed");
    AgcDriverWaitIdle_nid_postfix();
    check(output->state->ready == 1 && output->state->failed == 0, "a flip submitted on a compute queue was not handed to its output");
    check(output->state->last.argument == -0x123456789abcdefLL && output->state->last.index == -2 && output->state->last.mode == 1, "decoded compute flip arguments changed");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

void testFailure() {
    auto output = std::make_shared<Output>();
    output->state->fail = true;
    AgcDriverRegisterVideoOutput_nid_postfix(7, output);
    submitFlip();
    std::array<std::string, 4> messages;
    std::vector<std::thread> waiters;
    for (auto& message : messages) waiters.emplace_back([&message] { message = expectFailure([] { AgcDriverWaitIdle_nid_postfix(); }); });
    for (auto& waiter : waiters) waiter.join();
    for (auto& message : messages) check(message == "intentional flip failure", "asynchronous failure was lost");
    check(expectFailure([] { AgcDriverWaitIdle_nid_postfix(); }) == messages[0], "idle lost flip failure");
    check(expectFailure([] { AgcDriverSuspendPoint_nid_postfix(); }) == messages[0], "suspend lost flip failure");
    check(expectFailure([] { submitFlip(); }) == messages[0], "submit lost flip failure");
    check(output->state->ready == 0 && output->state->failed == 1, "failed flip was completed successfully");
    AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
}

void testFailureLog() {
    const auto path = std::filesystem::temp_directory_path() / ("aps5_worker_failure_log_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    std::fflush(stderr);
    const int saved = APS5_DUP(APS5_FILENO(stderr));
    check(saved >= 0, "stderr could not be saved");
#ifdef _WIN32
    check(_wfreopen(path.wstring().c_str(), L"wb", stderr) != nullptr, "stderr could not be redirected");
#else
    check(std::freopen(path.c_str(), "wb", stderr) != nullptr, "stderr could not be redirected");
#endif
    std::string message;
    std::string error;
    try {
        auto output = std::make_shared<Output>();
        output->state->fail = true;
        AgcDriverRegisterVideoOutput_nid_postfix(7, output);
        submitFlip();
        message = expectFailure([] { AgcDriverWaitIdle_nid_postfix(); });
        check(expectFailure([] { submitFlip(); }) == message, "the stored failure was not rethrown on the next driver call");
        AgcDriverUnregisterVideoOutput_nid_postfix(7, output);
    } catch (const std::exception& caught) {
        error = caught.what();
    }
    std::fflush(stderr);
    APS5_DUP2(saved, APS5_FILENO(stderr));
    APS5_CLOSE(saved);
    std::string captured;
    {
        std::ifstream file(path, std::ios::binary);
        captured.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(path);
    if (!error.empty()) throw std::runtime_error(error);
    check(message == "intentional flip failure", "the worker failure was not rethrown");
    check(captured.find("[gpu] driver failure: intentional flip failure\n") != std::string::npos, "the worker failure was not printed");
    check(captured.find("driver failure") == captured.rfind("driver failure"), "the worker failure was printed more than once");
}

void testReset(bool compute) {
    std::array<std::uint32_t, 4> registers{0xc0027600, 0x20c, 1, 0};
    Packet packet{registers.data(), 4, 0, {}};
    if (compute) sceAgcDriverSubmitAcb(0x20, &packet);
    else sceAgcDriverSubmitDcb(&packet);
    AgcDriverSuspendPoint_nid_postfix();
    std::array<std::uint32_t, 5> dispatch{0xc0031500, 1, 1, 1, 0x41};
    packet = Packet{dispatch.data(), 5, 0, {}};
    if (compute) sceAgcDriverSubmitAcb(0x20, &packet);
    else sceAgcDriverSubmitDcb(&packet);
    const auto message = expectFailure([] { AgcDriverWaitIdle_nid_postfix(); });
    check(message.find(compute ? "not readable" : "required shader register") != std::string::npos, "suspend reset wrong queue state");
}

}

void testPresentMode() {
    using AgcDriver::SelectPresentMode;
    VkSurfaceCapabilitiesKHR surface{};
    surface.minImageCount = 2;
    surface.maxImageCount = 8;
    check(AgcDriver::SwapchainImageCount(surface) == 3, "the swapchain has no image beyond the surface minimum");
    surface.minImageCount = 3;
    check(AgcDriver::SwapchainImageCount(surface) == 3, "a three-image surface minimum gained an image");
    surface.minImageCount = 5;
    check(AgcDriver::SwapchainImageCount(surface) == 5, "the swapchain does not follow a larger surface minimum");
    surface.minImageCount = 2;
    surface.maxImageCount = 2;
    check(AgcDriver::SwapchainImageCount(surface) == 2, "the swapchain exceeds the surface maximum");
    surface.maxImageCount = 0;
    check(AgcDriver::SwapchainImageCount(surface) == 3, "an unlimited surface maximum capped the swapchain");
    const std::array wayland{VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
    const std::array all{VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_FIFO_KHR};
    check(SelectPresentMode(nullptr, wayland) == VK_PRESENT_MODE_FIFO_KHR, "the default present mode is not FIFO");
    check(SelectPresentMode("fifo", wayland) == VK_PRESENT_MODE_FIFO_KHR, "fifo was not selected");
    check(SelectPresentMode("immediate", wayland) == VK_PRESENT_MODE_IMMEDIATE_KHR, "immediate was not selected");
    check(SelectPresentMode("mailbox", all) == VK_PRESENT_MODE_MAILBOX_KHR, "mailbox was not selected");
    const std::array latest{VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_LATEST_READY_KHR};
    check(SelectPresentMode("fifo_latest_ready", latest) == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR, "fifo_latest_ready was not selected");
    check(SelectPresentMode(nullptr, latest) == VK_PRESENT_MODE_FIFO_KHR, "the default changed on a surface with fifo_latest_ready");
    check(expectFailure([&] { SelectPresentMode("fifo_latest_ready", wayland); }).find("does not support") != std::string::npos, "fifo_latest_ready was accepted on a surface without it");
    check(expectFailure([&] { SelectPresentMode("mailbox", wayland); }).find("does not support") != std::string::npos, "an unsupported present mode was accepted");
    check(expectFailure([&] { SelectPresentMode("Immediate", all); }).find("must be fifo, fifo_latest_ready, mailbox or immediate") != std::string::npos, "an unknown present mode name was accepted");
    check(expectFailure([&] { SelectPresentMode("", all); }).find("must be fifo, fifo_latest_ready, mailbox or immediate") != std::string::npos, "an empty present mode name was accepted");
    const std::array<VkPresentModeKHR, 0> none{};
    check(expectFailure([&] { SelectPresentMode(nullptr, none); }).find("does not support") != std::string::npos, "FIFO was selected on a surface without it");
    using AgcDriver::SelectPresentQueue;
    check(SelectPresentQueue(16, 2) == 2u, "the presentation queue does not follow the main and settle queues");
    check(SelectPresentQueue(16, 1) == 1u, "the presentation queue does not follow the main queue alone");
    check(!SelectPresentQueue(2, 2).has_value(), "a presentation queue was taken from a family without a free queue");
    check(!SelectPresentQueue(1, 1).has_value(), "a presentation queue was taken from a family with one queue");
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "hold-shutdown") {
            testFlipShutdown();
            std::puts("AGC flip shutdown test passed");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "gpu-replacement") {
            testFlipDeviceReplacement();
            LibcRunShutdown_nid_postfix();
            std::puts("AGC flip GPU replacement test passed");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "gpu-hold") {
            testFlipWaitsForGpuCompletion();
            LibcRunShutdown_nid_postfix();
            std::puts("AGC flip GPU completion test passed");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "present-mode") {
            testPresentMode();
            std::puts("present mode selection tests passed");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "failure-log") {
            testFailureLog();
            const auto shutdown = expectFailure([] { LibcRunShutdown_nid_postfix(); });
            check(shutdown == "intentional flip failure", "shutdown lost worker failure");
            std::puts("AGC worker failure log test passed");
            return 0;
        }
        if (argc == 2) testReset(std::string(argv[1]) == "compute");
        else {
            std::fputs("stage boundary\n", stderr); testFlipAndBoundary();
            std::fputs("stage hold\n", stderr); testFlipHold();
            std::fputs("stage trailing\n", stderr); testFlipHoldCoversTrailingLabel();
            std::fputs("stage compute wait\n", stderr); testComputeRenderingWait();
            std::fputs("stage compute flip\n", stderr); testComputeFlip();
            std::fputs("stage failure\n", stderr); testFailure();
        }
        const auto shutdown = expectFailure([] { LibcRunShutdown_nid_postfix(); });
        check(shutdown.find(argc == 2 ? (std::string(argv[1]) == "compute" ? "not readable" : "required shader register") : "intentional flip failure") != std::string::npos, "shutdown lost worker failure");
        std::puts("AGC flip and suspend tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { LibcRunShutdown_nid_postfix(); }
        catch (const std::exception& shutdown) { std::fprintf(stderr, "shutdown: %s\n", shutdown.what()); }
        return 1;
    }
}
