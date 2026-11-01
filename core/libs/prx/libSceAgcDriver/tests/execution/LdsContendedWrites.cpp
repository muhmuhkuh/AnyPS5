#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <span>
#include <vector>

namespace {
using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;
constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t LdsDwords = 1;
alignas(256) std::array<std::uint32_t, Threads> Output{};

enum class StoreValue { SameSeven, LaneId };

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t elementCount) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u),
            elementCount, 0x11016facu};
}

std::vector<std::uint32_t> Kernel(StoreValue value, bool laneZeroInactive) {
    std::vector<std::uint32_t> code;
    if (value == StoreValue::SameSeven) {
        code.insert(code.end(), {0x7e0802ffu, 7u});
    } else {
        code.push_back(0x7e080300u);
    }
    code.push_back(0x7e040280u);

    if (laneZeroInactive) {
        code.push_back(0xbe88037eu);
        code.insert(code.end(), {0xbefe03ffu, 0xfffffffeu});
    }

    code.insert(code.end(), {0xd8340000u, 0x00000402u});

    if (laneZeroInactive) code.push_back(0xbefe0308u);
    code.push_back(0xbf8cc07fu);
    code.push_back(0xbf8a0000u);
    code.push_back(0xbf8cc07fu);
    code.insert(code.end(), {0xd8d80000u, 0x14000002u});
    code.push_back(0xbf8cc07fu);
    code.insert(code.end(), {0xe0702000u, 0x80001400u});
    code.push_back(0xbf810000u);
    return code;
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device,
                                           std::span<const std::uint32_t> code) {
    std::vector<std::uint32_t> userData(4, 0u);
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size()));
    std::copy(output.begin(), output.end(), userData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{
        {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{
        {Threads, 1, 1}, LdsDwords, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.ComputeTarget(32), {0, 0, 0, 128}};
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void Run(AgcDriver::VulkanDevice& device, StoreValue value, bool laneZeroInactive) {
    const auto code = Kernel(value, laneZeroInactive);
    const auto result = Compile(device, code);
    Output.fill(0xdeadbeefu);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    const auto sharedValue = Output[0];
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        Require(Output[lane] == sharedValue,
                "lds contended writes: lanes did not read the same post-barrier LDS value");
        if (value == StoreValue::SameSeven) {
            Require(Output[lane] == 7u,
                    "lds contended writes: equal-value contended stores did not read back 7");
        } else {
            const std::uint32_t firstPossible = laneZeroInactive ? 1u : 0u;
            Require(Output[lane] >= firstPossible && Output[lane] < Threads,
                    "lds contended writes: final value was not written by an active lane");
        }
    }
}
} // namespace

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device, StoreValue::SameSeven, false);
        Run(*device, StoreValue::LaneId, false);
        Run(*device, StoreValue::LaneId, true);
        std::puts("lds contended writes tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}


