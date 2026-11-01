#pragma once
#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace AgcDriver::GuestMemory {

class FileBackedProofArena final : public std::enable_shared_from_this<FileBackedProofArena> {
public:
    class Block final {
    public:
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        Block(Block&& other) noexcept { moveFrom(std::move(other)); }
        Block& operator=(Block&& other) noexcept {
            if (this != &other) {
                release();
                moveFrom(std::move(other));
            }
            return *this;
        }
        ~Block() { release(); }

        std::span<std::byte> WritableBytes() noexcept {
            if (!owner_ || sealed_) return {};
            return {owner_->base_ + offset_, bytes_};
        }
        std::span<const std::byte> Bytes() const noexcept {
            if (!owner_ || !sealed_) return {};
            return {owner_->base_ + offset_, bytes_};
        }
        std::uint64_t Offset() const noexcept { return offset_; }
        std::size_t Size() const noexcept { return bytes_; }
        void Seal() noexcept { sealed_ = true; }
        explicit operator bool() const noexcept { return owner_ != nullptr; }

    private:
        friend class FileBackedProofArena;
        Block(std::shared_ptr<FileBackedProofArena> owner, std::size_t slot, std::uint64_t generation,
            std::uint64_t offset, std::size_t bytes) noexcept
            : owner_(std::move(owner)), slot_(slot), generation_(generation), offset_(offset), bytes_(bytes) {}
        void release() noexcept {
            if (owner_) owner_->release(slot_, generation_);
            owner_.reset();
            offset_ = 0;
            bytes_ = 0;
            sealed_ = false;
        }
        void moveFrom(Block&& other) noexcept {
            owner_ = std::move(other.owner_);
            slot_ = std::exchange(other.slot_, 0);
            generation_ = std::exchange(other.generation_, 0);
            offset_ = std::exchange(other.offset_, 0);
            bytes_ = std::exchange(other.bytes_, 0);
            sealed_ = std::exchange(other.sealed_, false);
        }
        std::shared_ptr<FileBackedProofArena> owner_;
        std::size_t slot_ = 0;
        std::uint64_t generation_ = 0;
        std::uint64_t offset_ = 0;
        std::size_t bytes_ = 0;
        bool sealed_ = false;
    };

    static constexpr std::uint64_t MaxFileBytes = std::uint64_t{8192} << 20u;
    static constexpr std::size_t MaxEntries = 65536;

    static std::shared_ptr<FileBackedProofArena> Create(std::uint64_t fileBytes, std::size_t maxEntries = MaxEntries) noexcept {
        if (fileBytes == 0 || fileBytes > MaxFileBytes || fileBytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) || maxEntries == 0 || maxEntries > MaxEntries) return {};
        try {
            auto arena = std::unique_ptr<FileBackedProofArena>(new FileBackedProofArena(fileBytes, maxEntries));
            if (!arena->initialize()) return {};
            return std::shared_ptr<FileBackedProofArena>(arena.release());
        } catch (...) {
            return {};
        }
    }

    std::optional<Block> Allocate(std::size_t bytes) noexcept {
        if (bytes == 0) return std::nullopt;
        try {
            std::lock_guard lock(mutex_);
            if (freeSlotCount_ == 0) return std::nullopt;
            std::size_t rangeIndex = freeRangeCount_;
            for (std::size_t i = 0; i < freeRangeCount_; ++i) {
                if (freeRanges_[i].bytes >= bytes) {
                    rangeIndex = i;
                    break;
                }
            }
            if (rangeIndex == freeRangeCount_) return std::nullopt;
            const auto offset = freeRanges_[rangeIndex].offset;
            if (freeRanges_[rangeIndex].bytes == bytes) {
                for (auto i = rangeIndex + 1; i < freeRangeCount_; ++i) freeRanges_[i - 1] = freeRanges_[i];
                --freeRangeCount_;
            } else {
                freeRanges_[rangeIndex].offset += bytes;
                freeRanges_[rangeIndex].bytes -= bytes;
            }
            const auto slot = freeSlots_[--freeSlotCount_];
            auto generation = nextGeneration_++;
            if (generation == 0) generation = nextGeneration_++;
            slots_[slot] = Slot{offset, bytes, generation, true};
            return Block(shared_from_this(), slot, generation, offset, bytes);
        } catch (...) {
            return std::nullopt;
        }
    }

    std::uint64_t Capacity() const noexcept { return capacity_; }
    std::size_t MetadataBytes() const noexcept {
        return sizeof(*this) + (maxEntries_ + 1) * sizeof(Range) + maxEntries_ * sizeof(Slot) +
            maxEntries_ * sizeof(std::size_t) + 4 * sizeof(void*);
    }
    std::size_t LiveBlocks() noexcept {
        std::lock_guard lock(mutex_);
        return maxEntries_ - freeSlotCount_;
    }

    FileBackedProofArena(const FileBackedProofArena&) = delete;
    FileBackedProofArena& operator=(const FileBackedProofArena&) = delete;

    ~FileBackedProofArena() {
        if (base_ != nullptr) UnmapViewOfFile(base_);
        if (mapping_ != nullptr) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

private:
    struct Range {
        std::uint64_t offset;
        std::uint64_t bytes;
    };
    struct Slot {
        std::uint64_t offset = 0;
        std::size_t bytes = 0;
        std::uint64_t generation = 0;
        bool active = false;
    };

    FileBackedProofArena(std::uint64_t capacity, std::size_t maxEntries)
        : capacity_(capacity), maxEntries_(maxEntries), freeRanges_(new Range[maxEntries + 1]), slots_(new Slot[maxEntries]), freeSlots_(new std::size_t[maxEntries]) {
        freeRanges_[0] = {0, capacity_};
        freeRangeCount_ = 1;
        freeSlotCount_ = maxEntries_;
        for (std::size_t i = 0; i < maxEntries_; ++i) freeSlots_[i] = maxEntries_ - i - 1;
    }

    bool initialize() {
        std::wstring directory(32768, L'\0');
        const auto length = GetTempPathW(static_cast<DWORD>(directory.size()), directory.data());
        if (length == 0 || length >= directory.size()) return false;
        directory.resize(length);
        if (directory.back() != L'\\' && directory.back() != L'/') directory.push_back(L'\\');
        for (std::uint32_t attempt = 0; attempt < 128; ++attempt) {
            const auto sequence = tempSequence_.fetch_add(1, std::memory_order_relaxed);
            const auto name = L"AnyPS5-proof-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(sequence) + L".tmp";
            const auto path = directory + name;
            const auto file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW,
                FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (file == INVALID_HANDLE_VALUE) {
                const auto error = GetLastError();
                if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) continue;
                return false;
            }
            file_ = file;
            FILE_ALLOCATION_INFO allocation{};
            allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(capacity_);
            if (!SetFileInformationByHandle(file_, FileAllocationInfo, &allocation, sizeof(allocation))) return false;
            LARGE_INTEGER end{};
            end.QuadPart = static_cast<LONGLONG>(capacity_);
            if (!SetFilePointerEx(file_, end, nullptr, FILE_BEGIN) || !SetEndOfFile(file_)) return false;
            mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READWRITE, static_cast<DWORD>(capacity_ >> 32u), static_cast<DWORD>(capacity_), nullptr);
            if (mapping_ == nullptr) return false;
            base_ = static_cast<std::byte*>(MapViewOfFile(mapping_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, static_cast<SIZE_T>(capacity_)));
            return base_ != nullptr;
        }
        return false;
    }

    void release(std::size_t slot, std::uint64_t generation) noexcept {
        std::lock_guard lock(mutex_);
        if (slot >= maxEntries_ || !slots_[slot].active || slots_[slot].generation != generation) return;
        const auto offset = slots_[slot].offset;
        const auto bytes = static_cast<std::uint64_t>(slots_[slot].bytes);
        slots_[slot].active = false;
        if (freeSlotCount_ >= maxEntries_ || offset > capacity_ || bytes > capacity_ - offset) std::terminate();
        std::size_t at = 0;
        while (at < freeRangeCount_ && freeRanges_[at].offset < offset) ++at;
        const bool joinsLeft = at != 0 && freeRanges_[at - 1].offset + freeRanges_[at - 1].bytes == offset;
        const bool joinsRight = at < freeRangeCount_ && offset + bytes == freeRanges_[at].offset;
        if (joinsLeft && joinsRight) {
            freeRanges_[at - 1].bytes += bytes + freeRanges_[at].bytes;
            for (auto i = at + 1; i < freeRangeCount_; ++i) freeRanges_[i - 1] = freeRanges_[i];
            --freeRangeCount_;
        } else if (joinsLeft) {
            freeRanges_[at - 1].bytes += bytes;
        } else if (joinsRight) {
            freeRanges_[at].offset = offset;
            freeRanges_[at].bytes += bytes;
        } else {
            if (freeRangeCount_ >= maxEntries_ + 1) std::terminate();
            for (auto i = freeRangeCount_; i > at; --i) freeRanges_[i] = freeRanges_[i - 1];
            freeRanges_[at] = {offset, bytes};
            ++freeRangeCount_;
        }
        freeSlots_[freeSlotCount_++] = slot;
    }

    std::uint64_t capacity_;
    std::size_t maxEntries_;
    std::unique_ptr<Range[]> freeRanges_;
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<std::size_t[]> freeSlots_;
    std::size_t freeRangeCount_ = 0;
    std::size_t freeSlotCount_ = 0;
    std::uint64_t nextGeneration_ = 1;
    std::mutex mutex_;
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    std::byte* base_ = nullptr;
    inline static std::atomic<std::uint64_t> tempSequence_{0};
};

}
#endif
