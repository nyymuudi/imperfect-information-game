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

// ── Preflop equity approximation ──────────────────────────────────────────────
inline float preflop_equity(int rank_high, int rank_low, bool suited) {
    float base = 0.30f + rank_high * 0.026f;
    if (rank_high == rank_low) base += 0.15f;
    if (rank_high - rank_low == 1) base += 0.02f;
    if (rank_high - rank_low == 2) base += 0.01f;
    if (suited) base += 0.03f;
    return std::min(base, 0.88f);
}

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
// Produces the same 122-dim tensor as Python NLHEEncoder.

class NLHEStateEncoder {
public:
    static constexpr int STATE_SIZE = 122;
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
// Inputs:  "state"       [batch, 122] float32
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
    // state_vec:   flat float array of length STATE_SIZE (122).
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