#pragma once
#include "nlhe_game.hpp"
#include <string>
#include <vector>
#include <memory>

#ifdef CFR_TORCH_AVAILABLE
#include <torch/script.h>
#endif

#ifdef CFR_ORT_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

namespace cfr {

// ── Preflop equity (HEURISTIC FALLBACK ONLY) ──────────────────────────────────
//
// dim 120 of the encoder is the preflop equity of the player's hand. The
// CANONICAL value comes from the deterministic Monte-Carlo table that the
// Python side writes to
//     src/abstraction/_equity_cache/preflop_equity_<sims>.json
// and that NLHEStateEncoder::encode loads (see torch_model.cpp,
// PreflopEquityTable). That table is what gives dim-120 PARITY with Python.
//
// This closed-form heuristic is ONLY a fallback used when the JSON table is
// absent (e.g. a fresh build before any Python run has generated it), so that
// the engine still produces a plausible, monotone equity rather than zero. It
// is intentionally crude and is NOT parity-accurate; do not rely on it when a
// trained pipeline is expected. Renamed with a _heuristic suffix so call sites
// make the fallback nature explicit.
inline float preflop_equity_heuristic(int rank_high, int rank_low, bool suited) {
    float base = 0.30f + rank_high * 0.026f;
    if (rank_high == rank_low) base += 0.15f;
    if (rank_high - rank_low == 1) base += 0.02f;
    if (rank_high - rank_low == 2) base += 0.01f;
    if (suited) base += 0.03f;
    return std::min(base, 0.88f);
}

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
// Card-abstracted encoder — produces the same 36-dim tensor as Python NLHEEncoder.
//
// Layout (36 dims):
//   [0:8]   Preflop hand bucket one-hot (K=8, bins on equity ∈ [0,1])
//   [8:16]  Board EHS bucket one-hot    (K=8; zeros preflop)
//   [16:20] Street one-hot
//   [20:24] Betting scalars (pot, to_call, stacks)
//   [24:32] Action history (8 slots, NLHE_ACTION_ENC values)
//   [32]    Preflop equity (continuous)
//   [33]    Board strength (continuous, hand-eval score / MAX_SCORE)
//   [34]    Pot odds
//   [35]    SPR
//
// PARITY NOTES:
//   * dim 32 (preflop equity): loaded from equity-table JSON for parity with
//     Python; falls back to preflop_equity_heuristic if absent.
//   * dims [0:8] (preflop bucket): derived from dim 32 → min(int(eq*8), 7).
//     Bit-identical to Python as long as equity values agree.
//   * dims [8:16] (board bucket): derived from board_strength → min(int(s*8),7).
//     C++ board_strength uses a different score packing than Python (accepted
//     residual — both are monotone, wide bins minimise boundary disagreements).
//   * dims 20-35 quantised to 1e-6 grid.
class NLHEStateEncoder {
public:
    static constexpr int K_PREFLOP  = 8;
    static constexpr int K_BOARD    = 8;
    // STATE_SIZE = 8 (preflop) + K_BOARD + 4 (street) + 4 (betting)
    //            + 8 (history) + 4 (continuous tail) = 28 + K_BOARD.
    static constexpr int STATE_SIZE = 28 + K_BOARD;

    // Bucket scheme — runtime-switchable so training and h2h can use the same
    // .so. 0 = flat (K=8 one-hot, v3 production), 1 = tree (4-hot super +
    // 4-hot fine within super, two-hot in 8 dims). State size unchanged.
    enum class Scheme : int { Flat = 0, Tree = 1 };
    static void set_scheme(int s);
    static int  get_scheme();

    static void encode(const NLHEState& state, int player, float* out);
    static std::vector<float> encode_vec(const NLHEState& state, int player);

#ifdef CFR_TORCH_AVAILABLE
    static torch::Tensor encode_tensor(const NLHEState& state, int player);
#endif

private:
    static float board_strength(const NLHEState& state, int player);
};

// ── TorchModel ────────────────────────────────────────────────────────────────
// Wraps a TorchScript REGRET network loaded via LibTorch.
// Used in the MCCFR traversal hot path (every iteration).
// NOTE: TorchScript export is deprecated in newer PyTorch but LibTorch
// loading (torch::jit::load) is unaffected — regret export stays TorchScript.

class TorchModel {
public:
    TorchModel() = default;
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }
    std::vector<float> forward(const std::vector<float>& state_vec,
                               int max_actions) const;

#ifdef CFR_TORCH_AVAILABLE
    std::vector<float> forward_tensor(torch::Tensor input,
                                      int max_actions) const;
#endif

private:
    bool loaded_ = false;
#ifdef CFR_TORCH_AVAILABLE
    mutable torch::jit::script::Module module_;
#endif
};

// ── OnnxStrategyModel ─────────────────────────────────────────────────────────
// Loads a blueprint STRATEGY network from an ONNX file via ONNX Runtime.
// Used for blueprint querying during subgame solving (not the training loop).
//
// Inputs:  "state"       [batch, 124] float32
//          "action_mask" [batch, 4]   float32   (1=legal, 0=illegal)
// Output:  "probs"       [batch, 4]   float32   (masked softmax)
//
// Build prerequisite:  brew install onnxruntime
// CMake flag:          -DONNXRUNTIME_ROOT=... (auto-detected from homebrew)

class OnnxStrategyModel {
public:
    OnnxStrategyModel();

    // Load ONNX model from file. Returns true on success.
    bool load(const std::string& path);
    bool loaded() const { return loaded_; }

    // Single-state inference.
    // state_vec:   flat float array of length STATE_SIZE (124).
    // max_actions: number of legal actions (1–4); illegal slots masked to 0.
    // Returns:     vector of length 4 (probs[i]=0 for i >= max_actions).
    std::vector<float> forward(const std::vector<float>& state_vec,
                               int max_actions) const;

    // Batch inference — more efficient for subgame solving.
    // states:       [batch * STATE_SIZE] flat row-major float array.
    // max_actions:  per-row action count, length batch.
    // Returns:      [batch * 4] flat row-major float array.
    std::vector<float> forward_batch(
        const std::vector<float>& states,
        const std::vector<int>&   max_actions) const;

private:
#ifdef CFR_ORT_AVAILABLE
    mutable Ort::Env     env_;
    mutable std::unique_ptr<Ort::Session> session_;
#endif
    bool loaded_ = false;
};

} // namespace cfr