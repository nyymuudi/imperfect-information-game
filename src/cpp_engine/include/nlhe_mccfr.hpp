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
    const std::string& info_set,
    const std::vector<NLHEAction>& actions
)>;

struct NLHETraversalConfig {
    int      n_traversals      = 500;
    int      iteration         = 0;
    size_t   regret_capacity   = 1 << 20;
    size_t   strategy_capacity = 1 << 20;
    bool     collect_strategy  = true;
    uint64_t seed              = 42;
    int      max_actions       = 6;
};

class NLHEMCCFREngine {
public:
    explicit NLHEMCCFREngine(const NLHETraversalConfig& config);

    // ── Traversal ─────────────────────────────────────────────────────────────
    void run_traversals(int traversing_player, const NLHEStrategyFn& fn);
    void run_traversals_uniform(int traversing_player);

    // LibTorch-based traversal (zero Python callbacks)
    bool load_model(const std::string& path);
    bool model_loaded() const { return model_.loaded(); }
    void run_traversals_model(int traversing_player);

    // ── Strategy evaluation (C++ replaces Python get_strategy) ───────────────
    // Load trained strategy network (TorchScript).
    bool load_strategy_model(const std::string& path);
    bool strategy_model_loaded() const { return strategy_model_.loaded(); }

    // Query strategy for a given NLHEState from player's perspective.
    // Returns probability vector over 4 Python-compatible action slots:
    //   [0]=fold/check, [1]=check/call, [2]=raise, [3]=all-in
    std::vector<float> query_strategy(
        int hole0, int hole1,   // player's hole cards (0-51)
        int street,             // 0=preflop
        const std::vector<int>& board,   // visible board cards
        float pot, float to_call, float my_stack
    ) const;

    // Convenience: query preflop strategy for SB (player 0, initial state).
    std::vector<float> query_preflop_strategy(int hole0, int hole1) const;

    // ── Buffers ───────────────────────────────────────────────────────────────
    void   clear_buffers()        { regret_buf_.clear(); strategy_buf_.clear(); }
    void   set_iteration(int it)  { config_.iteration = it; }
    size_t regret_buffer_size()   const { return regret_buf_.size(); }
    size_t strategy_buffer_size() const { return strategy_buf_.size(); }

    MCCFREngine::BufferExport export_regret_buffer()   const;
    MCCFREngine::BufferExport export_strategy_buffer() const;

private:
    float traverse_cb(const NLHEState&, int, float, float, const NLHEStrategyFn&);
    float traverse_model(const NLHEState&, int, float, float);
    std::vector<float> model_strategy(const NLHEState&, int,
                                       const std::vector<NLHEAction>&);
    std::vector<float> uniform_strategy(int n);

    NLHETraversalConfig              config_;
    ReservoirBuffer<RegretSample>    regret_buf_;
    ReservoirBuffer<StrategySample>  strategy_buf_;
    std::mt19937                     rng_;
    TorchModel                       model_;           // regret net
    TorchModel                       strategy_model_;  // strategy net
};

} // namespace cfr