#pragma once
#include "leduc_game.hpp"
#include <functional>
#include <vector>
#include <random>
#include <unordered_map>
#include <array>
#include <cstring>

namespace cfr {

// ── Sample structs (SoA-friendly, CUDA-ready) ────────────────────────────────

struct RegretSample {
    char    info_set[32];
    int8_t  action;
    float   regret;
    int32_t iteration;
};

struct StrategySample {
    char    info_set[32];
    int8_t  action;
    float   probability;
    int32_t iteration;
};

// ── Reservoir Buffer ──────────────────────────────────────────────────────────

template <typename T>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(size_t capacity)
        : capacity_(capacity), size_(0), total_seen_(0) {
        data_.reserve(capacity);
    }
    bool insert(const T& sample);
    size_t size()        const { return size_; }
    size_t total_seen()  const { return total_seen_; }
    bool   full()        const { return size_ == capacity_; }
    const T* data() const { return data_.data(); }
    T*       data()       { return data_.data(); }
    void clear() { data_.clear(); size_ = 0; total_seen_ = 0; }
    std::vector<T> sample_batch(size_t n, std::mt19937& rng) const;
private:
    size_t        capacity_;
    size_t        size_;
    size_t        total_seen_;
    std::vector<T> data_;
    std::mt19937  rng_{42};
};

using StrategyFn = std::function<std::vector<float>(
    const std::string& info_set,
    const std::vector<Action>& legal_actions
)>;

// ── Regret target mode ────────────────────────────────────────────────────────
// CFRPLUS : R(I) <- max(R(I) + r^t, 0) accumulated across iterations, emitted
//           ONCE per infoset per iteration as R(I)/visits(I). Mirrors
//           DeepCFRSolver._cfrplus_regret (the validated Python fix).
// INSTANT : legacy per-(infoset,action) instantaneous cf-regret, emitted at
//           every node visit. Kept for A/B comparison (DEEPCFR_TARGET=instant).
enum class RegretTarget { CFRPLUS, INSTANT };

struct TraversalConfig {
    int    n_traversals     = 1000;
    int    iteration        = 0;
    size_t regret_capacity  = 1 << 20;
    size_t strategy_capacity= 1 << 20;
    bool   collect_strategy = true;
    uint64_t seed           = 42;
    RegretTarget target     = RegretTarget::CFRPLUS;
};

// Persistent per-infoset CFR+ accumulator state.
struct CfrPlusEntry {
    std::array<float, MAX_ACTIONS> R{};   // clipped cumulative regret per action
    int32_t visits   = 0;                 // per-infoset visit count
    int8_t  n_actions = 0;
};

class MCCFREngine {
public:
    explicit MCCFREngine(const TraversalConfig& config);

    void run_traversals(int traversing_player, const StrategyFn& strategy_fn);
    void run_traversals_uniform(int traversing_player);

    // Deterministic FULL traversal (vanilla CFR, no sampling). Used for
    // bit-exact parity testing of the CFR+ target against the Python reference.
    // Expands all actions at every node; weights regret by opponent reach.
    void run_full_traversal(int traversing_player, const StrategyFn& strategy_fn);
    void run_full_traversal_uniform(int traversing_player);

    const ReservoirBuffer<RegretSample>&   regret_buffer()   const { return regret_buf_; }
    const ReservoirBuffer<StrategySample>& strategy_buffer() const { return strategy_buf_; }
    ReservoirBuffer<RegretSample>&   regret_buffer()   { return regret_buf_; }
    ReservoirBuffer<StrategySample>& strategy_buffer() { return strategy_buf_; }

    void   clear_buffers()       { regret_buf_.clear(); strategy_buf_.clear(); }
    void   set_iteration(int it) { config_.iteration = it; }
    void   reset_cfrplus()       { cfrplus_.clear(); }
    // Emit the current CFR+ accumulator (R/visits) into the regret buffer:
    // ONE sample per (infoset, action). Call once after each iteration's
    // traversals, mirroring the Python emit-per-iteration semantics so that
    // the Python-side _collapse_by_state sees one row per infoset.
    void   emit_cfrplus_targets();
    size_t regret_buffer_size()   const { return regret_buf_.size(); }
    size_t strategy_buffer_size() const { return strategy_buf_.size(); }

    struct BufferExport {
        std::vector<std::string> info_sets;
        std::vector<int8_t>      actions;
        std::vector<float>       values;
        std::vector<int32_t>     iterations;
    };
    BufferExport export_regret_buffer()   const;
    BufferExport export_strategy_buffer() const;

private:
    float traverse(
        const LeducState& state, int traversing_player,
        float reach_prob_traverser, float reach_prob_opponent,
        const StrategyFn& strategy_fn);

    // Full (vanilla) traversal: returns node value, accumulates the CFR+ target
    // for the traversing player at each of its nodes (weighted by opp reach).
    float traverse_full(
        const LeducState& state, int traversing_player,
        float reach_prob_opponent, const StrategyFn& strategy_fn);

    // Fold the instantaneous regret vector into the persistent CFR+ accumulator.
    void accumulate_cfrplus(const std::string& iset,
                            const std::vector<Action>& legal,
                            const std::vector<float>& instant_regret);

    std::vector<float> regret_match(
        const std::string& info_set, const std::vector<Action>& legal_actions,
        const StrategyFn& strategy_fn);

    TraversalConfig                 config_;
    ReservoirBuffer<RegretSample>   regret_buf_;
    ReservoirBuffer<StrategySample> strategy_buf_;
    std::mt19937                    rng_;
    std::vector<LeducDeal>          deals_;
    std::unordered_map<std::string, CfrPlusEntry> cfrplus_;  // persistent
};

} // namespace cfr