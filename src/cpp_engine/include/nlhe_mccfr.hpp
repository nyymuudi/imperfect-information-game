#pragma once
#include "nlhe_game.hpp"
#include "mccfr.hpp"
#include "torch_model.hpp"
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>

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

// ── State-vector-keyed CFR+ accumulator ───────────────────────────────────────
// NLHE samples carry the 124-d state vector, not an info-set string, and the
// Python _collapse_by_state groups by EXACT state vector. To stay consistent
// across the whole pipeline (accumulate == network input == collapse key) we
// accumulate CFR+ regret keyed on the state vector itself, NOT on
// NLHEGame::info_set_key (which quantises the pot and uses full history, so it
// is a DIFFERENT granularity than the encoder). See design notes in
// nlhe_mccfr.cpp.
struct NLHECfrPlusEntry {
    std::array<float, NLHE_NUM_ACTIONS> R{};         // clipped cumulative regret
    // Previous-iteration per-action instantaneous regret. Used by
    // Predictive CFR+ (Brown 2020) to add a momentum term:
    //     R_{t+1} = max(R_t + r_t + alpha*(r_t - prev_r), 0)
    // The alpha coefficient is config_.predictive_alpha. Zeroed before any
    // visits → first-iteration update degenerates to vanilla CFR+.
    std::array<float, NLHE_NUM_ACTIONS> prev_r{};
    std::array<float, NLHEStateEncoder::STATE_SIZE> state{};  // encoder output
    int32_t visits    = 0;
    int8_t  n_actions = 0;
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
    RegretTarget target        = RegretTarget::CFRPLUS;  // shares Leduc enum
    NLHEGameConfig game_cfg;

    // Pluribus-style dynamic pruning (Brown & Sandholm 2019).
    // At a traversing-player node, actions with blueprint probability below
    // prune_threshold are skipped (not rolled out, not regret-updated).
    // The node value is computed over the visited actions only, renormalised.
    // 0.0 = pruning disabled. Recommended 0.001…0.01. Pruning activates only
    // when iteration >= prune_after_iter so early-iteration regret signal
    // (which often shows uniform-ish blueprint output) is not lost.
    float prune_threshold     = 0.0f;
    int   prune_after_iter    = 0;

    // Predictive CFR+ momentum (Brown 2020). When > 0 AND target == CFRPLUS,
    // each iteration's CFR+ update folds in (predictive_alpha) × (r_t - r_{t-1})
    // so the cumulative regret tracks not just the current iter's regret but
    // also its rate of change → 3-5× faster convergence empirically on the
    // games Brown benchmarked. 0.0 = vanilla CFR+ (default; safe fallback).
    // Brown's paper used alpha=1.0 implicitly via R_{t+1} = max(R_t + 2r_t - r_{t-1}).
    float predictive_alpha    = 0.0f;
};

// ── Engine ────────────────────────────────────────────────────────────────────

class NLHEMCCFREngine {
public:
    explicit NLHEMCCFREngine(const NLHETraversalConfig& config);

    // ── Traversal (regret network = TorchScript via LibTorch) ────────────────
    void run_traversals(int traversing_player, const NLHEStrategyFn& fn);
    void run_traversals_uniform(int traversing_player);
    bool load_model(const std::string& path);
    bool model_loaded() const { return model_.loaded(); }
    void run_traversals_model(int traversing_player);

    // ── Deterministic full traversal for parity testing ──────────────────────
    // Expands ALL actions at every node (vanilla CFR), on a FIXED deal so the
    // tree is enumerable. Used to validate the CFR+ target against the Python
    // PostflopNLHE reference on a small subtree (low stack, max_raises=1).
    void run_full_traversal_deal_uniform(int traversing_player, const NLHEDeal& deal);

    // ── CFR+ accumulator controls (mirrors Leduc MCCFREngine) ────────────────
    void   reset_cfrplus() { cfrplus_.clear(); }
    void   emit_cfrplus_targets();

    // ── Strategy queries (blueprint = ONNX via ONNX Runtime) ─────────────────
    bool load_strategy_model(const std::string& path);
    bool strategy_model_loaded() const { return strategy_model_.loaded(); }
    std::vector<float> query_strategy(
        int hole0, int hole1, int street,
        const std::vector<int>& board,
        float pot, float to_call, float my_stack) const;
    std::vector<float> query_preflop_strategy(int hole0, int hole1) const;

    // ── EV-adjusted terminals (variance-reducing regret targets) ──────────────
    // When enabled, terminal nodes that reached showdown via all-in (both
    // players' stacks hit 0 before the river) use equity over remaining board
    // cards instead of the deal-specific realised outcome. Same regret target
    // for the same strategic spot regardless of which runout was sampled.
    static void set_ev_adjusted_terminals(bool on);
    static bool get_ev_adjusted_terminals();

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

    // Full (vanilla) traversal, all actions expanded; accumulates CFR+ target
    // for the traversing player at each of its nodes (weighted by opp reach).
    float traverse_full(const NLHEState&, int, float, const NLHEStrategyFn&);

    // Fold instantaneous regret into the state-vector-keyed CFR+ accumulator.
    void accumulate_cfrplus(const float* state_vec,
                            const std::vector<NLHEAction>& legal,
                            const std::vector<float>& instant_regret);

    // Route a computed instantaneous-regret vector to the configured target:
    // INSTANT -> emit per (state, action) now; CFRPLUS -> fold into accumulator.
    void emit_or_accumulate(const float* state_vec,
                            const std::vector<NLHEAction>& legal,
                            const std::vector<float>& instant_regret);

    std::vector<float> model_strategy(const NLHEState&, int,
                                       const std::vector<NLHEAction>&);
    std::vector<float> uniform_strategy(int n);

    // Canonical hash key for a state vector (quantised bytes) -> accumulator.
    static std::string state_key(const float* state_vec);

    NLHETraversalConfig                 config_;
    ReservoirBuffer<NLHERegretSample>   regret_buf_;
    ReservoirBuffer<NLHEStrategySample> strategy_buf_;
    std::mt19937                        rng_;

    std::unordered_map<std::string, NLHECfrPlusEntry> cfrplus_;  // persistent

    TorchModel         model_;           // regret net  — TorchScript (LibTorch)
    OnnxStrategyModel  strategy_model_;  // strategy net — ONNX (ONNX Runtime)
};

} // namespace cfr