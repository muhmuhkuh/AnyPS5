#pragma once

#include <cstddef>
#include <cstdint>

namespace TextureSnapshotDiagnostics {

template <typename Node, std::size_t BucketCount, std::size_t MaxEntries = 65536>
class SnapshotDuplicateIndex {
    static_assert(BucketCount != 0 && (BucketCount & (BucketCount - 1)) == 0);

public:
    struct VisitResult {
        bool found;
        bool truncated;
        std::size_t visited;
    };

    void SetBuckets(Node** buckets) noexcept {
        if (entries_ == 0) buckets_ = buckets;
    }

    bool Insert(Node& node) noexcept {
        if (buckets_ == nullptr || node.duplicateAuditPrevious != nullptr || entries_ >= MaxEntries) return false;
        auto** link = buckets_ + (node.diagnosticPayloadHash & (BucketCount - 1));
        node.duplicateAuditNext = *link;
        node.duplicateAuditPrevious = link;
        if (node.duplicateAuditNext != nullptr) node.duplicateAuditNext->duplicateAuditPrevious = &node.duplicateAuditNext;
        *link = &node;
        ++entries_;
        return true;
    }

    bool Erase(Node& node) noexcept {
        if (node.duplicateAuditPrevious == nullptr || entries_ == 0) return false;
        *node.duplicateAuditPrevious = node.duplicateAuditNext;
        if (node.duplicateAuditNext != nullptr) node.duplicateAuditNext->duplicateAuditPrevious = node.duplicateAuditPrevious;
        node.duplicateAuditNext = nullptr;
        node.duplicateAuditPrevious = nullptr;
        --entries_;
        return true;
    }

    template <typename Visitor>
    VisitResult Visit(std::uint64_t hash, std::size_t maxVisited, Visitor&& visitor) const {
        if (buckets_ == nullptr) return {false, false, 0};
        auto* node = buckets_[hash & (BucketCount - 1)];
        std::size_t visited = 0;
        while (node != nullptr && visited < maxVisited) {
            auto* next = node->duplicateAuditNext;
            ++visited;
            if (node->diagnosticPayloadHash == hash && visitor(*node)) return {true, false, visited};
            node = next;
        }
        return {false, node != nullptr, visited};
    }

    void Clear() noexcept {
        if (buckets_ != nullptr) {
            for (std::size_t index = 0; index < BucketCount; ++index) {
                auto* node = buckets_[index];
                buckets_[index] = nullptr;
                while (node != nullptr) {
                    auto* next = node->duplicateAuditNext;
                    node->duplicateAuditNext = nullptr;
                    node->duplicateAuditPrevious = nullptr;
                    node = next;
                }
            }
        }
        entries_ = 0;
    }

    std::size_t Size() const noexcept { return entries_; }

private:
    Node** buckets_ = nullptr;
    std::size_t entries_ = 0;
};

}
