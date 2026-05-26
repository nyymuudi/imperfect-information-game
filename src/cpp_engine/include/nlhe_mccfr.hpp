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

struct NLHEBufferExport {
    std::vector<float>   states;
    std::vector<int8_t>  actions;
    std::vector<float>   values;
    std::vector<int32_t> iterations;
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

    // ── Traversal (regret network = TorchScript via LibTorch) ────────────────
    void run_traversals(int traversing_player, const NLHEStrategyFn& fn);
    void run_traversals_uniform(int traversing_player);
    bool load_model(const std::string& path);          // loads regret TorchScript
    bool model_loaded() const { return model_.loaded(); }
    void run_traversals_model(int traversing_player);

    // ── Strategy queries (blueprint = ONNX via ONNX Runtime) ─────────────────
    // load_strategy_model() expects a .onnx file (from Blueprint.save()).
    bool load_strategy_model(const std::string& path);
    bool strategy_model_loaded() const { return strategy_model_.loaded(); }

    std::vector<float> query_strategy(
        int hole0, int hole1, int street,
        const std::vector<int>& board,
        float pot, float to_call, float my_stack) const;

    std::vector<float> query_preflop_strategy(int hole0, int hole1) const;

    // ── Buffers ───────────────────────────────────────────────────────────────
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

    TorchModel         model_;           // regret net  — TorchScript (LibTorch)
    OnnxStrategyModel  strategy_model_;  // strategy net — ONNX (ONNX Runtime)
};

} // namespace cfr