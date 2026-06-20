#pragma once
// CFR advisor cache loader — C++ side of the Python/C++ cache contract.
//
// File format (little-endian, packed):
//
//   uint32  magic   = 0x43464341 ('CFCA')
//   uint32  version = 1
//   uint32  n_entries
//   uint32  prob_dim   (currently always 6)
//   uint32  ev_dim     (currently always 6)
//   float32 ev_norm    (= 2 * starting_stack)
//   uint8   reserved[12]
//   --- entries (sorted by key ascending) ---
//   for i in [0..n_entries):
//       uint64  key
//       float32 probs[prob_dim]
//       float32 evs[ev_dim]
//
// Total entry size = 8 + 4*6 + 4*6 = 56 bytes.
// 2000 entries → 112 KB. Fits in L2 cache.
//
// Lookup: binary search over the sorted keys[] array.
//
// The corresponding Python exporter is
//   src/deep_cfr/cfr_cache.py:CFRCache.save_binary(path)
// which mirrors this layout byte-for-byte. A parity test compares
// (key, probs, evs) round-trip between the .npz and .bin formats.

#include <cstdint>
#include <string>
#include <vector>

namespace cfr {

struct CacheEntry {
    uint64_t key;
    float    probs[6];
    float    evs[6];
};

class CFRCacheLoader {
public:
    static constexpr uint32_t MAGIC   = 0x43464341u;
    static constexpr uint32_t VERSION = 1u;

    // Empty / unloaded.
    CFRCacheLoader() = default;

    // Load a binary cache. Returns true on success. False if file missing,
    // magic mismatch, or version unsupported. On failure ``loaded()`` stays
    // false and lookup() returns null.
    bool load(const std::string& path);

    bool loaded() const { return loaded_; }
    size_t size() const { return entries_.size(); }
    float ev_norm() const { return ev_norm_; }

    // Binary-search lookup. Returns pointer to entry (probs+evs valid for
    // the lifetime of *this) or nullptr on miss. Thread-safe (read-only).
    const CacheEntry* lookup(uint64_t key) const;

private:
    bool loaded_ = false;
    float ev_norm_ = 100.0f;
    std::vector<CacheEntry> entries_;   // sorted by key
};

} // namespace cfr
