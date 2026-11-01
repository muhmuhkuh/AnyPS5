#include "prx/libSceAgcDriver/Execution/include/FileBackedProofArena.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

using AgcDriver::GuestMemory::FileBackedProofArena;

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void fill(FileBackedProofArena::Block& block, std::byte value) {
    auto bytes = block.WritableBytes();
    require(bytes.size() == block.Size(), "new block must be writable");
    require(block.Bytes().empty(), "unsealed data must not be published");
    std::fill(bytes.begin(), bytes.end(), value);
    block.Seal();
    require(block.WritableBytes().empty(), "sealed block must not expose writable bytes");
    require(std::all_of(block.Bytes().begin(), block.Bytes().end(), [value](auto byte) { return byte == value; }), "sealed bytes changed");
}

}

int main() {
    try {
        require(!FileBackedProofArena::Create(0), "zero capacity accepted");
        require(!FileBackedProofArena::Create(FileBackedProofArena::MaxFileBytes + 1), "oversized file accepted");
        require(!FileBackedProofArena::Create(1, 0), "zero metadata slots accepted");
        auto arena = FileBackedProofArena::Create(128ull << 20u, 8);
        require(arena != nullptr, "file mapping creation failed");
        require(!arena->Allocate(0) && !arena->Allocate(129u << 20u), "invalid allocation accepted");
        auto a = arena->Allocate(32u << 20u);
        auto b = arena->Allocate(32u << 20u);
        auto c = arena->Allocate(32u << 20u);
        require(a && b && c, "bounded allocations failed");
        fill(*a, std::byte{0x31});
        fill(*b, std::byte{0x52});
        fill(*c, std::byte{0x73});
        MEMORY_BASIC_INFORMATION mapping{};
        require(VirtualQuery(a->Bytes().data(), &mapping, sizeof(mapping)) == sizeof(mapping) && mapping.Type == MEM_MAPPED && mapping.State == MEM_COMMIT, "proof payload is not file mapped");
        a.reset();
        b.reset();
        auto joined = arena->Allocate(64u << 20u);
        require(joined.has_value(), "adjacent released capacity was not reclaimed");
        fill(*joined, std::byte{0xA4});
        require(std::all_of(c->Bytes().begin(), c->Bytes().end(), [](auto byte) { return byte == std::byte{0x73}; }), "reclaimed allocation overwrote a live block");
        auto tail = arena->Allocate(32u << 20u);
        require(tail && !arena->Allocate(1), "capacity exhaustion was not reported");
        joined.reset();
        c.reset();
        tail.reset();
        require(arena->LiveBlocks() == 0, "released blocks remain live");
        auto whole = arena->Allocate(128u << 20u);
        require(whole.has_value(), "full capacity was not recovered");
        whole->WritableBytes().front() = std::byte{0xD2};
        whole->Seal();
        auto retained = std::move(*whole);
        require(!static_cast<bool>(*whole) && whole->Bytes().empty(), "moved block retained ownership");
        whole.reset();
        arena.reset();
        require(retained.Bytes().front() == std::byte{0xD2}, "block did not retain its mapping owner");
        auto bounded = FileBackedProofArena::Create(16ull << 20u, 1);
        require(bounded != nullptr, "metadata-bound arena creation failed");
        auto only = bounded->Allocate(1);
        require(only && !bounded->Allocate(1), "metadata slot limit was exceeded");
        only.reset();
        require(bounded->Allocate(16u << 20u).has_value(), "metadata slot was not reclaimed");
        require(bounded->MetadataBytes() != 0, "allocator metadata is unaccounted");
        std::cout << "file-backed proof ownership, bytes, reclaim and bounds passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
