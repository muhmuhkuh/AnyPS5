#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Words = 32;
constexpr std::uint32_t Size = 16;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Type3D = 10;
constexpr std::uint32_t Depth = 4;
constexpr std::uint32_t TableWidth = 64;
constexpr std::uint32_t TableHeight = 4;
constexpr std::uint32_t TableRecords = 12;
constexpr std::uint32_t TableStrideDwords = 12;
constexpr std::uint32_t SampleResult = 8;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(256) std::array<std::uint8_t, 65536> Plane{};
alignas(256) std::array<std::uint8_t, 65536> Volume{};
alignas(4096) std::array<std::uint8_t, 65536> TableVolumeA{};
alignas(4096) std::array<std::uint8_t, 65536> TableVolumeB{};

alignas(256) constexpr std::array<std::uint32_t, 36> VolumeSampleCode{
    0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xe0301010, 0x80000601, 0xe0301014, 0x80000701, 0xbf8c3f70, 0xf09c0f10, 0x00610c02,
    0xf0001f10, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
    0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038,
    0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 36> PlaneSampleCode{
    0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xe0301010, 0x80000601, 0xe0301014, 0x80000701, 0xbf8c3f70, 0xf09c0f08, 0x00610c02,
    0xf0001f10, 0x00011005, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028,
    0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030, 0x80001001, 0xe0701034, 0x80001101, 0xe0701038,
    0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 16> ExplicitLodVolumeCode{
    0x34020087, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xe0301008, 0x80000401, 0xe030100c,
    0x80000501, 0xbf8c3f70, 0xf0900f08, 0x00610c02, 0xbf8c3f70, 0xe0701020, 0x80000c01, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 28> SelectedTableSampleLzCode{
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u, 0xf4080700u, 0xfa000020u, 0x8f108202u,
    0xf4200444u, 0x20000000u, 0x9312b011u, 0xf4300802u, 0x24000000u, 0x360200bfu, 0x7e020d01u, 0x060202f0u,
    0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c0f08u, 0x01090401u, 0x7e120202u, 0x34101286u,
    0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 22> DirectTableSampleLzCode{
    0xf4080700u, 0xfa000020u, 0xf4080800u, 0xfa000030u, 0xf4080900u, 0xfa000040u, 0x360200bfu, 0x7e020d01u,
    0x060202f0u, 0x100202ffu, 0x3c800000u, 0x7e0402ffu, 0x3e000000u, 0xf09c0f08u, 0x01090401u, 0x7e120202u,
    0x34101286u, 0x4a101100u, 0x34101084u, 0xe0781000u, 0x80070408u, 0xbf810000u,
};

constexpr std::array<float, 8> PlaneR{0.0f, 0.5f, 1.0f, 1.5f, -0.25f, 3.0f, 0.25f, 0.75f};
constexpr std::array<std::uint32_t, 8> PlaneZ{0u, 1u, 0u, 2u, 0xffffffffu, 4u, 0u, 3u};
alignas(256) std::array<std::uint32_t, TableRecords * TableStrideDwords> Table{};
alignas(256) std::array<std::uint32_t, 64> Keys{};
alignas(256) std::array<std::uint32_t, 32> Srt{};

struct Lane {
    std::uint32_t x;
    std::uint32_t y;
};

Lane LaneOf(std::uint32_t tid) {
    return {(tid * 7u + 3u) % Size, (tid * 5u + 1u) % Size};
}

std::uint32_t Bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

void FillInput(bool volume) {
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto lane = LaneOf(tid);
        const auto u = Bits((static_cast<float>(lane.x) + 0.5f) / static_cast<float>(Size));
        const auto v = Bits((static_cast<float>(lane.y) + 0.5f) / static_cast<float>(Size));
        const auto r = volume ? Bits((static_cast<float>(tid % Depth) + 0.5f) / static_cast<float>(Depth)) : Bits(PlaneR[tid % PlaneR.size()]);
        const auto z = volume ? tid % Depth : PlaneZ[tid % PlaneZ.size()];
        const std::array<std::uint32_t, 6> words{u, v, r, lane.x, lane.y, z};
        std::copy(words.begin(), words.end(), &Buffer[tid * Words]);
    }
}

std::uint8_t SliceMarker(std::uint32_t z) {
    return static_cast<std::uint8_t>(z * 53u + 7u);
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* texels, std::uint32_t depth) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (((Size - 1u) & 3u) << 30u),
        ((Size - 1u) >> 2u) | ((Size - 1u) << 14u),
        0xfacu | ((depth == 1u ? Type2D : Type3D) << 28u),
        depth - 1u,
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 8> TableTextureDescriptor(const void* texels, std::uint32_t depth) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texels));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888UNorm << 20u) | (((TableWidth - 1u) & 3u) << 30u),
        ((TableWidth - 1u) >> 2u) | ((TableHeight - 1u) << 14u) | (1u << 31u),
        0xfacu | ((depth == 1u ? Type2D : Type3D) << 28u),
        depth - 1u,
        0u,
        0u,
        0u,
    };
}

void FillTexture(std::span<std::uint8_t> storage, const std::array<std::uint32_t, 8>& descriptor, std::uint32_t slices) {
    const auto geometry = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor));
    Require(geometry.guestBytes <= storage.size(), "image plane address on volume: the texture does not fit its texel storage");
    const auto& mip = geometry.mips.at(0);
    std::fill(storage.begin(), storage.end(), 0xeeu);
    for (std::uint32_t z = 0; z < slices; ++z) {
        for (std::uint32_t y = 0; y < Size; ++y) {
            for (std::uint32_t x = 0; x < Size; ++x) {
                auto* texel = &storage[geometry.GuestLayerOffset(z) + mip.tiledOffset + static_cast<std::uint64_t>(y) * mip.pitchBytes + x * 4u];
                texel[0] = SliceMarker(z);
                texel[1] = static_cast<std::uint8_t>(x);
                texel[2] = static_cast<std::uint8_t>(y);
                texel[3] = 255u;
            }
        }
    }
}

void FillTableVolume(std::span<std::uint8_t> storage, std::uint32_t slices, std::uint32_t markerBias) {
    const auto descriptor = TableTextureDescriptor(storage.data(), slices);
    const auto geometry = AgcDriver::Graphics::DescribeSurface(AgcDriver::Graphics::DecodeTextureResource(descriptor));
    Require(geometry.guestBytes <= storage.size(), "image plane address on volume: the table texture does not fit its texel storage");
    std::fill(storage.begin(), storage.end(), 0xeeu);
    for (std::uint32_t z = 0; z < slices; ++z) {
        for (std::uint32_t y = 0; y < TableHeight; ++y) {
            for (std::uint32_t x = 0; x < TableWidth; ++x) {
                auto* texel = &storage[geometry.GuestLayerOffset(z) + geometry.mips.at(0).tiledOffset + static_cast<std::uint64_t>(y) * geometry.mips.at(0).pitchBytes + x * 4u];
                texel[0] = static_cast<std::uint8_t>(SliceMarker(z) + markerBias);
                texel[1] = static_cast<std::uint8_t>(x);
                texel[2] = static_cast<std::uint8_t>(y);
                texel[3] = 255u;
            }
        }
    }
}

void SetTableRecord(std::uint32_t record, const std::array<std::uint32_t, 4>& sampler, const std::array<std::uint32_t, 8>& texture) {
    auto* words = Table.data() + record * TableStrideDwords;
    std::copy(sampler.begin(), sampler.end(), words);
    std::copy(texture.begin(), texture.end(), words + 4);
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 4> SamplerDescriptor() {
    return {0u, 0xfffu << 12u, 1u << 26u, 0u};
}

template <std::size_t N>
void RunDirect(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, N>& code, const std::array<std::uint32_t, 8>& texture) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(Buffer.size() * 4u));
    const auto sampler = SamplerDescriptor();
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    const std::span<const std::uint32_t> span(code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(span.data()), span, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(span.data()));
    device.WaitIdle();
}

void RunSelectedTable(AgcDriver::VulkanDevice& device, std::uint32_t selectedRecord, const std::array<std::uint32_t, 8>& planarTexture, const std::array<std::uint32_t, 8>& volumeTexture) {
    Table.fill(0u);
    Keys.fill(0u);
    Srt.fill(0u);
    const auto sampler = SamplerDescriptor();
    SetTableRecord(0u, sampler, planarTexture);
    SetTableRecord(1u, sampler, volumeTexture);
    Keys[0] = selectedRecord;
    const auto table = BufferDescriptor(Table.data(), static_cast<std::uint32_t>(sizeof(Table)));
    const auto keys = BufferDescriptor(Keys.data(), static_cast<std::uint32_t>(sizeof(Keys)));
    const auto output = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(sizeof(Buffer)));
    std::copy(table.begin(), table.end(), Srt.begin());
    std::copy(keys.begin(), keys.end(), Srt.begin() + 4);
    std::copy(output.begin(), output.end(), Srt.begin() + 8);
    std::copy(sampler.begin(), sampler.end(), Srt.begin() + 12);
    const std::span<const std::uint32_t> code(SelectedTableSampleLzCode);
    const std::array<ShaderRecompiler::MemoryRegion, 4> memory{{
        {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)},
        {reinterpret_cast<std::uintptr_t>(Srt.data()), std::as_bytes(std::span(Srt))},
        {reinterpret_cast<std::uintptr_t>(Table.data()), std::as_bytes(std::span(Table))},
        {reinterpret_cast<std::uintptr_t>(Keys.data()), std::as_bytes(std::span(Keys))},
    }};
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Srt.data())), static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Srt.data()) >> 32u)};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto& expectedDescriptor = selectedRecord == 0u ? planarTexture : volumeTexture;
    const auto expectedShape = selectedRecord == 0u ? ShaderRecompiler::DescriptorImageShape::Image2D : ShaderRecompiler::DescriptorImageShape::Image3D;
    bool hasSelectedTableEntry = false;
    std::string tableBindingSummary;
    for (const auto& binding : result.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::ImageTable) continue;
        tableBindingSummary += " {kind=" + std::to_string(static_cast<int>(binding.kind)) + " count=" + std::to_string(binding.count) + " words=" + std::to_string(binding.guestDescriptor.size()) + " shapes=";
        for (const auto shape : binding.imageShapes) tableBindingSummary += std::to_string(static_cast<int>(shape)) + ",";
        tableBindingSummary += "}";
        if (binding.kind != ShaderRecompiler::DescriptorKind::SampledImage || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u || binding.imageShapes.size() != binding.count) continue;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            if (binding.imageShapes[element] != expectedShape) continue;
            const auto descriptorOffset = static_cast<std::size_t>(element) * 8u;
            if (std::equal(expectedDescriptor.begin(), expectedDescriptor.end(), binding.guestDescriptor.begin() + descriptorOffset)) hasSelectedTableEntry = true;
        }
    }
    Require(hasSelectedTableEntry, "image plane address on volume: no ImageTable binding retained the selected T# with the expected descriptor layout for record " + std::to_string(selectedRecord) + tableBindingSummary);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void RunDirectTable(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 8>& texture) {
    Table.fill(0u);
    Keys.fill(0u);
    Srt.fill(0u);
    const auto sampler = SamplerDescriptor();
    const auto table = BufferDescriptor(Table.data(), static_cast<std::uint32_t>(sizeof(Table)));
    const auto keys = BufferDescriptor(Keys.data(), static_cast<std::uint32_t>(sizeof(Keys)));
    const auto output = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(sizeof(Buffer)));
    std::copy(table.begin(), table.end(), Srt.begin());
    std::copy(keys.begin(), keys.end(), Srt.begin() + 4);
    std::copy(output.begin(), output.end(), Srt.begin() + 8);
    std::copy(sampler.begin(), sampler.end(), Srt.begin() + 12);
    std::copy(texture.begin(), texture.end(), Srt.begin() + 16);
    const std::span<const std::uint32_t> code(DirectTableSampleLzCode);
    const std::array<ShaderRecompiler::MemoryRegion, 4> memory{{
        {reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)},
        {reinterpret_cast<std::uintptr_t>(Srt.data()), std::as_bytes(std::span(Srt))},
        {reinterpret_cast<std::uintptr_t>(Table.data()), std::as_bytes(std::span(Table))},
        {reinterpret_cast<std::uintptr_t>(Keys.data()), std::as_bytes(std::span(Keys))},
    }};
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Srt.data())), static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Srt.data()) >> 32u)};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Expect(std::uint32_t tid, const std::array<std::uint32_t, 4>& expected, const std::string& name) {
    for (std::uint32_t component = 0; component < 4u; ++component) {
        const float value = std::bit_cast<float>(Buffer[tid * Words + SampleResult + component]) * 255.0f;
        Require(std::lround(value) == static_cast<long>(expected[component]), name + ": thread " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected[component]));
    }
}

void CheckPlaneLz() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto lane = LaneOf(tid);
        Expect(tid, {SliceMarker(0), lane.x, lane.y, 255u}, "2D image_sample_lz on a 2D T# with varying R");
    }
}

void CheckVolumeLz() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const auto lane = LaneOf(tid);
        Expect(tid, {SliceMarker(tid % Depth), lane.x, lane.y, 255u}, "3D image_sample_lz on a 3D T#");
    }
}

std::array<std::uint32_t, 64 * 4> CopyTableOutput() {
    std::array<std::uint32_t, 64 * 4> copy{};
    std::copy_n(Buffer.begin(), copy.size(), copy.begin());
    return copy;
}

void CheckTableSampleLz(const std::array<std::uint32_t, 64 * 4>* planar, std::uint32_t markerBias, const std::string& name) {
    for (std::uint32_t tid = 0; tid < 64u; ++tid) {
        const auto offset = tid * 4u;
        const std::array<std::uint32_t, 4> expected{static_cast<std::uint8_t>(SliceMarker(0) + markerBias), tid % TableWidth, 0u, 255u};
        for (std::uint32_t component = 0; component < 4u; ++component) {
            const auto actualBits = Buffer[offset + component];
            const auto actual = std::bit_cast<float>(actualBits) * 255.0f;
            Require(std::isfinite(actual) && std::lround(actual) == static_cast<long>(expected[component]),
                name + " lane " + std::to_string(tid) + " component " + std::to_string(component) + " is " + std::to_string(actual) + ", expected slice-zero RGBA channel " + std::to_string(expected[component]));
            if (planar != nullptr) Require(actualBits == (*planar)[offset + component], name + " differs from its direct 2D control at lane " + std::to_string(tid) + " component " + std::to_string(component));
        }
    }
}

void RequireExplicitLodRejected(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 8>& texture) {
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(Buffer.size() * 4u));
    const auto sampler = SamplerDescriptor();
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    const std::span<const std::uint32_t> code(ExplicitLodVolumeCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    std::string refusal;
    try {
        static_cast<void>(ShaderRecompiler::Recompile(request));
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    Require(refusal.find("incompatible with the static runtime image interface") != std::string::npos,
        "image plane address on volume: explicit-LOD 3D sample was not rejected: " + refusal);
}

void RunDirectTests(AgcDriver::VulkanDevice& device) {
    const auto planeDescriptor = TextureDescriptor(Plane.data(), 1u);
    FillTexture(Plane, planeDescriptor, 1u);
    FillInput(false);
    RunDirect(device, PlaneSampleCode, planeDescriptor);
    CheckPlaneLz();

    const auto volumeDescriptor = TextureDescriptor(Volume.data(), Depth);
    FillTexture(Volume, volumeDescriptor, Depth);
    FillInput(true);
    RunDirect(device, VolumeSampleCode, volumeDescriptor);
    CheckVolumeLz();

    FillInput(false);
    RunDirect(device, PlaneSampleCode, volumeDescriptor);
    CheckPlaneLz();
    RequireExplicitLodRejected(device, volumeDescriptor);
}

void RunTableTests(AgcDriver::VulkanDevice& device) {
    constexpr std::uint32_t SecondMarkerBias = 79u;
    FillTableVolume(TableVolumeA, Depth, 0u);
    FillTableVolume(TableVolumeB, Depth, SecondMarkerBias);
    const auto firstPlaneDescriptor = TableTextureDescriptor(TableVolumeA.data(), 1u);
    const auto secondPlaneDescriptor = TableTextureDescriptor(TableVolumeB.data(), 1u);
    const auto secondVolumeDescriptor = TableTextureDescriptor(TableVolumeB.data(), Depth);
    FillInput(false);
    RunDirectTable(device, firstPlaneDescriptor);
    CheckTableSampleLz(nullptr, 0u, "direct 2D table fixture control for record 0");
    const auto firstPlaneControl = CopyTableOutput();
    RunSelectedTable(device, 0u, firstPlaneDescriptor, secondVolumeDescriptor);
    CheckTableSampleLz(&firstPlaneControl, 0u, "GPU-selected 2D T# table record");

    RunDirectTable(device, secondPlaneDescriptor);
    CheckTableSampleLz(nullptr, SecondMarkerBias, "direct 2D table fixture control for record 1");
    const auto secondPlaneControl = CopyTableOutput();
    RunSelectedTable(device, 1u, firstPlaneDescriptor, secondVolumeDescriptor);
    CheckTableSampleLz(&secondPlaneControl, SecondMarkerBias, "GPU-selected 3D T# table record");
}

}

int main(int argc, char** argv) {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const std::string_view mode = argc == 2 ? argv[1] : "";
        Require(argc <= 2 && (mode.empty() || mode == "--direct-only" || mode == "--table-only"), "usage: agc_driver_image_plane_address_on_volume_tests [--direct-only|--table-only]");
        if (mode.empty() || mode == "--direct-only") RunDirectTests(*device);
        if (mode.empty() || mode == "--table-only") RunTableTests(*device);

        std::puts("image plane address on volume tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
