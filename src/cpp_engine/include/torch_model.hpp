#pragma once
#include "nlhe_game.hpp"
#include <string>
#include <vector>

#ifdef CFR_TORCH_AVAILABLE
#include <torch/script.h>
#endif

namespace cfr {

// ── Preflop equity approximation ──────────────────────────────────────────────
// rank_high ∈ [0,12], rank_low ∈ [0,12], suited ∈ {0,1}
// Returns approximate equity vs random hand (0..1).
inline float preflop_equity(int rank_high, int rank_low, bool suited) {
    float base = 0.30f + rank_high * 0.026f;   // 2=0.30, A=0.61
    if (rank_high == rank_low) base += 0.15f;   // pocket pair
    if (rank_high - rank_low == 1) base += 0.02f;
    if (rank_high - rank_low == 2) base += 0.01f;
    if (suited) base += 0.03f;
    return std::min(base, 0.88f);
}

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
// Produces the same 122-dim tensor as Python NLHEEncoder, computed from
// the raw NLHEState struct. No string parsing, no Python callbacks.

class NLHEStateEncoder {
public:
    static constexpr int STATE_SIZE = 122;

    // Encode state from player's perspective into a flat float array.
    // out must have capacity STATE_SIZE.
    static void encode(const NLHEState& state, int player, float* out);

    // Convenience: returns std::vector<float>
    static std::vector<float> encode_vec(const NLHEState& state, int player);

#ifdef CFR_TORCH_AVAILABLE
    // Returns a [1, STATE_SIZE] float tensor ready for network inference.
    static torch::Tensor encode_tensor(const NLHEState& state, int player);
#endif

private:
    static float board_strength(const NLHEState& state, int player);
};

// ── TorchModel ────────────────────────────────────────────────────────────────
// Wraps a TorchScript regret network. Thread-safe for read (inference only).

class TorchModel {
public:
    TorchModel() = default;

    // Load model from TorchScript file. Returns true on success.
    bool load(const std::string& path);

    bool loaded() const { return loaded_; }

    // Run inference: input [1, 122] → output [max_actions].
    // Returns empty vector if not loaded.
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

} // namespace cfr