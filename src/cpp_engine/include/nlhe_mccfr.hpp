#pragma once
#include "nlhe_game.hpp"
#include "mccfr.hpp"
#include "torch_model.hpp"
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace cfr {

using NLHEStrategyFn = std::function<std::vector<float>(
    const std::string&, const std::vector<NLHEAction>&)>;

// ── NLHE-specific sample types ────────────────────────────────────────────────
// Store the full 122-dim state vector instead of the info_set string.
// Training features == inference features — no mismatch possible.

struct NLHERegretSample {
    float   state[NLHEStateEncoder::STATE_SIZE];
    int8_t  action;
    float   regret;
    int32_t iteration;
};

struct NLHEStrategySample {
    float   state[NLHEStateEncoder::STATE_SIZE];
    int8_t  action;
    float   probability;
    int32_t iteration;
};

// Flat export for Python (pybind11 → numpy).
// states is a flat float array of shape [n_samples, STATE_SIZE].
struct NLHEBufferExport {
    std::vector<float>   states;      // [n_samples * STATE_SIZE]
    std::vector<int8_t>  actions;     // [n_samples]
    std::vector<float>   values;      // [n_samples]
    std::vector<int32_t> iterations;  // [n_samples]
    size_t n_samples = 0;
    static constexpr size_t state_size = NLHEStateEncoder::STATE_SIZE;
    size_t __len__() const { return n_samples; }
};

// ── Traversal config ──────────────────────────────────────────────────────────
struct NLHETraversalConfig {
    int      n_traversals      = 500;
    int      iteration         = 0;
    size_t   regret_capacity   = 1 << 20;
    size_t   strategy_capacity = 1 << 20;
    bool     collect_strategy  = true;
    uint64_t seed              = 42;
    int      max_actions       = 4;
    NLHEGameConfig game_cfg;
};

// ── Engine ────────────────────────────────────────────────────────────────────
class NLHEMCCFREngine {
public:
    explicit NLHEMCCFREngine(const NLHETraversalConfig& config);

    // Traversal
    void run_traversals(int traversing_player, const NLHEStrategyFn& fn);
    void run_traversals_uniform(int traversing_player);
    bool load_model(const std::string& path);
    bool model_loaded() const { return model_.loaded(); }
    void run_traversals_model(int traversing_player);

    // Strategy evaluation
    bool load_strategy_model(const std::string& path);
    bool strategy_model_loaded() const { return strategy_model_.loaded(); }
    std::vector<float> query_strategy(
        int hole0, int hole1, int street,
        const std::vector<int>& board,
        float pot, float to_call, float my_stack) const;
    std::vector<float> query_preflop_strategy(int hole0, int hole1) const;

    // Buffers
    void   clear_buffers()       { regret_buf_.clear(); strategy_buf_.clear(); }
    void   set_iteration(int it) { config_.iteration = it; }
    size_t regret_buffer_size()   const { return regret_buf_.size(); }
    size_t strategy_buffer_size() const { return strategy_buf_.size(); }
    NLHEBufferExport export_regret_buffer()   const;
    NLHEBufferExport export_strategy_buffer() const;

private:
    float traverse_cb(const NLHEState&, int, float, float, const NLHEStrategyFn&);
    float traverse_model(const NLHEState&, int, float, float);
    std::vector<float> model_strategy(const NLHEState&, int,
                                       const std::vector<NLHEAction>&);
    std::vector<float> uniform_strategy(int n);

    NLHETraversalConfig                 config_;
    ReservoirBuffer<NLHERegretSample>   regret_buf_;
    ReservoirBuffer<NLHEStrategySample> strategy_buf_;
    std::mt19937                        rng_;
    TorchModel                          model_;
    TorchModel                          strategy_model_;
};

} // namespace cfr