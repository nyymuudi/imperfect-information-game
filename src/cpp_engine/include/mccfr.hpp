#pragma once
#include "leduc_game.hpp"
#include <functional>
#include <vector>
#include <random>
#include <unordered_map>
#include <cstring>

namespace cfr {

// ── Sample structs (SoA-friendly, CUDA-ready) ────────────────────────────────

// One regret sample: (info_set_key, action, regret_value, iteration)
struct RegretSample {
    char    info_set[32];   // fixed-size for SoA batching
    int8_t  action;
    float   regret;
    int32_t iteration;
};

// One strategy sample: (info_set_key, action, reach_probability)
struct StrategySample {
    char    info_set[32];
    int8_t  action;
    float   probability;
    int32_t iteration;
};

// ── Reservoir Buffer ──────────────────────────────────────────────────────────
// Vitter (1985) Algorithm R. Fixed-capacity, O(1) amortised insert.
// Memory is contiguous → memcpy to GPU in one call.

template <typename T>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(size_t capacity)
        : capacity_(capacity), size_(0), total_seen_(0) {
        data_.reserve(capacity);
    }

    // Insert sample. Returns true if stored (not rejected by reservoir).
    bool insert(const T& sample);

    size_t size()        const { return size_; }
    size_t total_seen()  const { return total_seen_; }
    bool   full()        const { return size_ == capacity_; }

    const T* data() const { return data_.data(); }
    T*       data()       { return data_.data(); }

    void clear() { data_.clear(); size_ = 0; total_seen_ = 0; }

    // Random mini-batch (uniform sampling without replacement when possible).
    std::vector<T> sample_batch(size_t n, std::mt19937& rng) const;

private:
    size_t        capacity_;
    size_t        size_;
    size_t        total_seen_;
    std::vector<T> data_;
    std::mt19937  rng_{42};
};

// ── Strategy query callback ───────────────────────────────────────────────────
// During traversal, we need to evaluate the current strategy network.
// This is provided as a callback from Python (wraps the PyTorch model).
// Signature: (info_set_key, legal_actions) -> action_probabilities

using StrategyFn = std::function<std::vector<float>(
    const std::string& info_set,
    const std::vector<Action>& legal_actions
)>;

// ── External Sampling MCCFR ───────────────────────────────────────────────────

struct TraversalConfig {
    int    n_traversals     = 1000;
    int    iteration        = 0;
    size_t regret_capacity  = 1 << 20;   // 1M samples
    size_t strategy_capacity= 1 << 20;
    bool   collect_strategy = true;      // also fill strategy buffer?
    uint64_t seed           = 42;
};

class MCCFREngine {
public:
    explicit MCCFREngine(const TraversalConfig& config);

    // Run n_traversals external-sampling traversals.
    // Requires a strategy callback (wraps the current regret network).
    // Fills regret_buffer_ and strategy_buffer_.
    void run_traversals(int traversing_player, const StrategyFn& strategy_fn);

    // Tabular (no network) mode: use uniform random strategy.
    void run_traversals_uniform(int traversing_player);

    // Access buffers (hand off to Python → train networks).
    const ReservoirBuffer<RegretSample>&   regret_buffer()   const { return regret_buf_; }
    const ReservoirBuffer<StrategySample>& strategy_buffer() const { return strategy_buf_; }

    ReservoirBuffer<RegretSample>&   regret_buffer()   { return regret_buf_; }
    ReservoirBuffer<StrategySample>& strategy_buffer() { return strategy_buf_; }

    void   clear_buffers()       { regret_buf_.clear(); strategy_buf_.clear(); }
    void   set_iteration(int it) { config_.iteration = it; }
    size_t regret_buffer_size()   const { return regret_buf_.size(); }
    size_t strategy_buffer_size() const { return strategy_buf_.size(); }

    // Convert buffers to Python-compatible numpy arrays.
    // Returns (info_set_indices, actions, values) as flat vectors.
    struct BufferExport {
        std::vector<std::string> info_sets;
        std::vector<int8_t>      actions;
        std::vector<float>       values;
        std::vector<int32_t>     iterations;
    };
    BufferExport export_regret_buffer()   const;
    BufferExport export_strategy_buffer() const;

private:
    // Core external-sampling traversal (recursive, returns node utility).
    float traverse(
        const LeducState& state,
        int traversing_player,
        float reach_prob_traverser,
        float reach_prob_opponent,
        const StrategyFn& strategy_fn
    );

    // Compute current strategy via regret matching.
    std::vector<float> regret_match(
        const std::string& info_set,
        const std::vector<Action>& legal_actions,
        const StrategyFn& strategy_fn
    );

    TraversalConfig              config_;
    ReservoirBuffer<RegretSample>   regret_buf_;
    ReservoirBuffer<StrategySample> strategy_buf_;
    std::mt19937                 rng_;
    std::vector<LeducDeal>       deals_;
};

} // namespace cfr