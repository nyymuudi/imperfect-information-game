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

struct NLHETraversalConfig {
    int      n_traversals      = 500;
    int      iteration         = 0;
    size_t   regret_capacity   = 1 << 20;
    size_t   strategy_capacity = 1 << 20;
    bool     collect_strategy  = true;
    uint64_t seed              = 42;
    int      max_actions       = 4;   // matches Python 4-action space
    NLHEGameConfig game_cfg;          // passed to initial_state
};

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
    MCCFREngine::BufferExport export_regret_buffer()   const;
    MCCFREngine::BufferExport export_strategy_buffer() const;

private:
    float traverse_cb(const NLHEState&, int, float, float, const NLHEStrategyFn&);
    float traverse_model(const NLHEState&, int, float, float);
    // No mapping needed: C++ uses 4 actions natively
    std::vector<float> model_strategy(const NLHEState&, int,
                                       const std::vector<NLHEAction>&);
    std::vector<float> uniform_strategy(int n);

    NLHETraversalConfig             config_;
    ReservoirBuffer<RegretSample>   regret_buf_;
    ReservoirBuffer<StrategySample> strategy_buf_;
    std::mt19937                    rng_;
    TorchModel                      model_;
    TorchModel                      strategy_model_;
};

} // namespace cfr