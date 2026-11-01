#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "VulkanTestDevice.hpp"
#include "GuestImageAcquirePacket.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include "prx/libc/include/GuestWriteWatch.hpp"
#endif

namespace AgcDriver::DriverDetail {
struct DriverTestAccess {
    static std::shared_ptr<VulkanDevice> GetDevice() { return Driver::Get().device.Load(); }
};
}
namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}
namespace {
using namespace AgcDriver::Graphics;
constexpr std::size_t HeapBytes = 1024u * 1024u;
constexpr std::uint32_t Width = 256, Height = 128;
constexpr std::uint32_t Initial = 0x55555555u;
alignas(256) constexpr std::array<std::uint32_t, 1> Warmup{0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 15> Writer{
    0x7e080218u, 0x343c0885u, 0x4a3c3d00u, 0x7e3e0280u, 0x7e140d1eu, 0x4c163cffu, 0x000000ffu, 0x7e160d0bu,
    0x101414ffu, 0x3b808081u, 0x101616ffu, 0x3b808081u, 0xf0201108u, 0x00010a1eu, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 12> Reader{
    0xf4040100u, 0xfa000000u, 0xf4080201u, 0xfa000000u, 0xbf8cc07fu, 0xf4000182u, 0xfa000000u,
    0xbf8cc07fu, 0x7e000206u, 0xe0700000u, 0x80020000u, 0xbf810000u};
alignas(256) std::array<std::uint32_t, 4> Output{};
alignas(256) std::array<std::uint32_t, 4> OutputDescriptor{};
alignas(256) std::array<std::uint32_t, 2> PointerTable{};

std::uint64_t Address(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }
void Registers(std::vector<std::uint32_t>& words, std::uint32_t first, std::span<const std::uint32_t> values) {
    words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
    words.push_back(first);
    words.insert(words.end(), values.begin(), values.end());
}
std::vector<std::uint32_t> DispatchWords(std::span<const std::uint32_t> code, std::span<const std::uint32_t> data, bool acquire, std::uint32_t threads) {
    std::vector<std::uint32_t> words;
    if (acquire) {
        constexpr auto packet = ImageBdaAcquireFullCache();
        words.insert(words.end(), packet.begin(), packet.end());
    }
    const std::array<std::uint32_t, 3> shape{threads, 1u, 1u};
    Registers(words, 0x207u, shape);
    const auto address = Address(code.data());
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u)};
    Registers(words, 0x20cu, program);
    const std::array<std::uint32_t, 1> resource{static_cast<std::uint32_t>(data.size()) << 1u};
    Registers(words, 0x213u, resource);
    if (!data.empty()) Registers(words, 0x240u, data);
    words.insert(words.end(), {0xc0031500u, 1u, 1u, 1u, 0x8041u});
    return words;
}
void Submit(std::span<const std::uint32_t> code, std::span<const std::uint32_t> data, bool acquire = false, std::uint32_t threads = 32u) {
    auto words = DispatchWords(code, data, acquire, threads);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    AgcDriverWaitIdle_nid_postfix();
}
class Heap {
public:
    Heap() {
#ifdef _WIN32
        ptr = GuestArena::GuestArenaAllocate_nid_postfix(HeapBytes, 65536u);
        GuestArena::GuestArenaCommit_nid_postfix(ptr, HeapBytes, PAGE_READWRITE, HeapBytes);
#else
        void* raw = mmap(nullptr, HeapBytes + 65536u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "guest BDA fixture: mmap failed");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw), aligned = (begin + 65535u) & ~std::uintptr_t{65535u};
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + HeapBytes != begin + HeapBytes + 65536u) munmap(reinterpret_cast<void*>(aligned + HeapBytes), begin + HeapBytes + 65536u - aligned);
        ptr = reinterpret_cast<void*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(ptr, HeapBytes);
#endif
        Require(ptr != nullptr, "guest BDA fixture: allocation failed");
        std::memset(ptr, 0x55, HeapBytes);
        GuestAllocations::Mutation().Add(ptr, HeapBytes, true, true, true);
    }
    ~Heap() {
        GuestAllocations::Mutation().Remove(ptr);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(ptr, HeapBytes);
        GuestArena::GuestArenaRelease_nid_postfix(ptr, HeapBytes);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(ptr, HeapBytes);
        munmap(ptr, HeapBytes);
#endif
    }
    void* ptr = nullptr;
};
std::array<std::uint32_t, 8> ImageDescriptor(const void* ptr) {
    const auto address = Address(ptr);
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (((Width - 1u) & 3u) << 30u),
        ((Width - 1u) >> 2u) | ((Height - 1u) << 14u), 0x22cu | (27u << 20u) | (9u << 28u), 0u, 0u, 0u, 0u};
}
struct DriverShutdown { ~DriverShutdown() { try { AgcDriverShutdown_nid_postfix(); } catch (...) {} } };
}
int main() {
    try {
        { const auto probe = OpenVulkanTestDevice(); if (!probe) return VulkanTestSkipped; }
        Heap heap;
        DriverShutdown shutdown;
        Submit(Warmup, {});
        const auto device = AgcDriver::DriverDetail::DriverTestAccess::GetDevice();
        Require(device != nullptr, "guest BDA fixture: Driver device missing after warmup");
        auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(*device);
        if (!context.bufferDeviceAddress || !context.hostImportAlignment) return VulkanTestSkipped;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            const auto* import = HostImportFor(context, Address(heap.ptr), HeapBytes);
            if (import == nullptr || import->buffer == VK_NULL_HANDLE || import->address == 0 || !AgcDriver::GuestMemory::Watched(Address(heap.ptr), HeapBytes)) return VulkanTestSkipped;
        }
        const auto descriptor = ImageDescriptor(heap.ptr);
        const auto decoded = DecodeTextureResource(descriptor);
        const auto surface = DescribeSurface(decoded);
        Require(decoded.tileMode == TextureTileMode::kR64KBX && decoded.pipeBankXor == 0 && surface.guestBytes > 65536u && surface.guestBytes <= HeapBytes, "guest BDA fixture: expected R64KBX surface over 64 KiB");
        std::array<std::uint32_t, 24> writerData{};
        std::copy(descriptor.begin(), descriptor.end(), writerData.begin() + 4);
        const auto input = Address(heap.ptr);
        PointerTable = {static_cast<std::uint32_t>(input), static_cast<std::uint32_t>(input >> 32u)};
        Output.fill(0xaaaaaaaau);
        const auto output = Address(Output.data());
        OutputDescriptor = {static_cast<std::uint32_t>(output), static_cast<std::uint32_t>((output >> 32u) & 0xffffu), static_cast<std::uint32_t>(sizeof(Output)), 0x31016facu};
        const auto table = Address(PointerTable.data()), outDesc = Address(OutputDescriptor.data());
        const std::array<std::uint32_t, 4> readerData{static_cast<std::uint32_t>(table), static_cast<std::uint32_t>(table >> 32u), static_cast<std::uint32_t>(outDesc), static_cast<std::uint32_t>(outDesc >> 32u)};
        Submit(Reader, readerData, false, 1u);
        AgcDriver::GuestMemory::FlushGpuWrites(Address(Output.data()), sizeof(Output));
        std::printf("guest reader baseline=0x%08x expected=0x%08x\n", Output[0], Initial);
        Require(Output[0] == Initial, "guest reader did not load the initial value");
        Output.fill(0xaaaaaaaau);
        Submit(Writer, writerData);
        std::shared_ptr<StorageTexture> writerImage;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            writerImage = StorageTexture::FindPending(Address(heap.ptr), surface.guestBytes);
            if (writerImage == nullptr) {
                std::puts("INCONCLUSIVE: Guest writer/WaitIdle did not retain a pending storage image");
                return VulkanTestSkipped;
            }
        }
        std::uint32_t beforeReader = 0;
        AgcDriver::GuestMemory::ReadCommittedUnsynced(Address(heap.ptr), std::as_writable_bytes(std::span(&beforeReader, 1)));
        std::printf("guest diagnostic CPU peek before reader=0x%08x initial=0x%08x\n", beforeReader, Initial);
        const bool acquire = std::getenv("APS5_TEST_IMAGE_BDA_ACQUIRE") != nullptr;
        Submit(Reader, readerData, acquire, 1u);
        AgcDriver::GuestMemory::FlushGpuWrites(Address(Output.data()), sizeof(Output));
        const auto observed = Output[0];
        bool stillPending = false;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            stillPending = StorageTexture::FindPending(Address(heap.ptr), surface.guestBytes) == writerImage;
            if (writerImage->Dirty()) writerImage->WriteBack();
            PublishShadow(Address(heap.ptr), static_cast<std::size_t>(surface.guestBytes), PublishScope::Whole, PublishReason::Validate);
        }
        AgcDriverWaitIdle_nid_postfix();
        std::uint32_t expected = 0;
        AgcDriver::GuestMemory::ReadCommittedUnsynced(Address(heap.ptr), std::as_writable_bytes(std::span(&expected, 1)));
        Require(expected != Initial, "guest BDA fixture: target texel did not change; writer mapping or encoding is inconclusive");
        std::printf("guest image read: acquire=%d pending_after_reader=%d reader=0x%08x materialized=0x%08x initial=0x%08x; actual BDA/import identity unproven\n", acquire, stillPending, observed, expected, Initial);
        if (!stillPending) std::puts("OBSERVED: image no longer pending after reader; publish site not attributed");
        if (acquire || std::getenv("APS5_TEST_IMAGE_BDA_REQUIRE_MATCH") != nullptr) Require(observed == expected, "guest reader missed the pending image store");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
