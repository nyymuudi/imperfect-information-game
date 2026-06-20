// CFR advisor cache loader — implementation.
// See cfr_cache_loader.hpp for file-format documentation.
//
// Build: add to CMakeLists.txt sources, then rebuild cfr_engine.so.
// Integration (separate PR):
//   1. NLHEStateEncoder holds a static CFRCacheLoader instance.
//   2. NLHEStateEncoder::set_cache_path(string) loads it.
//   3. NLHEStateEncoder::encode() computes a key from state, looks up,
//      writes probs/evs to slots [37:49] (BASE_STATE_SIZE..STATE_SIZE).
//   4. Pybind11 binding exposes set_cache_path() to Python so
//      cpp_backend.NLHECppBackend.__init__() can wire it.

#include "cfr_cache_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace cfr {

bool CFRCacheLoader::load(const std::string& path) {
    loaded_ = false;
    entries_.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[cache] cannot open %s\n", path.c_str());
        return false;
    }

    uint32_t magic = 0, version = 0, n_entries = 0, prob_dim = 0, ev_dim = 0;
    float ev_norm = 0.0f;
    uint8_t reserved[12] = {0};

    f.read(reinterpret_cast<char*>(&magic),     sizeof(magic));
    f.read(reinterpret_cast<char*>(&version),   sizeof(version));
    f.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));
    f.read(reinterpret_cast<char*>(&prob_dim),  sizeof(prob_dim));
    f.read(reinterpret_cast<char*>(&ev_dim),    sizeof(ev_dim));
    f.read(reinterpret_cast<char*>(&ev_norm),   sizeof(ev_norm));
    f.read(reinterpret_cast<char*>(reserved),   sizeof(reserved));
    if (!f) {
        std::fprintf(stderr, "[cache] truncated header in %s\n", path.c_str());
        return false;
    }
    if (magic != MAGIC) {
        std::fprintf(stderr, "[cache] bad magic 0x%08x (expected 0x%08x) in %s\n",
                     magic, MAGIC, path.c_str());
        return false;
    }
    if (version != VERSION) {
        std::fprintf(stderr, "[cache] unsupported version %u in %s\n",
                     version, path.c_str());
        return false;
    }
    if (prob_dim != 6 || ev_dim != 6) {
        std::fprintf(stderr, "[cache] unexpected dims prob=%u ev=%u in %s\n",
                     prob_dim, ev_dim, path.c_str());
        return false;
    }

    ev_norm_ = ev_norm;
    entries_.resize(n_entries);
    for (uint32_t i = 0; i < n_entries; ++i) {
        CacheEntry& e = entries_[i];
        f.read(reinterpret_cast<char*>(&e.key),  sizeof(e.key));
        f.read(reinterpret_cast<char*>(e.probs), sizeof(e.probs));
        f.read(reinterpret_cast<char*>(e.evs),   sizeof(e.evs));
    }
    if (!f) {
        std::fprintf(stderr, "[cache] truncated entries (%u expected) in %s\n",
                     n_entries, path.c_str());
        entries_.clear();
        return false;
    }

    // Sanity: ensure sorted (Python exporter sorts; verify anyway).
    for (uint32_t i = 1; i < n_entries; ++i) {
        if (entries_[i].key < entries_[i - 1].key) {
            std::fprintf(stderr, "[cache] entries not sorted at i=%u in %s\n",
                         i, path.c_str());
            entries_.clear();
            return false;
        }
    }

    loaded_ = true;
    return true;
}

const CacheEntry* CFRCacheLoader::lookup(uint64_t key) const {
    if (!loaded_ || entries_.empty()) return nullptr;
    auto it = std::lower_bound(
        entries_.begin(), entries_.end(), key,
        [](const CacheEntry& e, uint64_t k) { return e.key < k; });
    if (it == entries_.end() || it->key != key) return nullptr;
    return &(*it);
}

} // namespace cfr
