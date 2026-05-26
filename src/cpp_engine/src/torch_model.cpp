#include "torch_model.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef CFR_TORCH_AVAILABLE
#include <torch/script.h>
#endif

namespace cfr {

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
//
// Tensor layout (122 dims) — matches Python NLHEEncoder EXACTLY:
//   [0:52]    hole cards one-hot
//   [52:104]  visible board cards one-hot
//   [104:108] street one-hot (0=preflop,1=flop,2=turn,3=river)
//   [108]     pot / (2 * starting_stack)
//   [109]     to_call / (2 * starting_stack)
//   [110]     my_stack / starting_stack
//   [111]     opp_stack / starting_stack
//   [112:120] action history (last 8 actions, ACTION_ENC values)
//   [120]     preflop equity
//   [121]     board_strength

void NLHEStateEncoder::encode(const NLHEState& state, int player, float* out) {
    const float STACK = state.cfg.starting_stack;
    const float NORM  = 2.0f * STACK;

    std::fill(out, out + STATE_SIZE, 0.0f);

    int c0 = state.hole_cards[player][0];
    int c1 = state.hole_cards[player][1];
    if(c0 >= 0 && c0 < 52) out[c0] = 1.0f;
    if(c1 >= 0 && c1 < 52) out[c1] = 1.0f;

    int n_visible = BOARD_CARDS_BY_STREET[state.street];
    for(int i = 0; i < n_visible; ++i) {
        int card = state.board[i];
        if(card >= 0 && card < 52) out[52 + card] = 1.0f;
    }

    int street = std::min((int)state.street, 3);
    out[104 + street] = 1.0f;

    float to_call = state.street_invest[1-player] - state.street_invest[player];
    to_call = std::max(0.0f, to_call);
    out[108] = std::min(state.pot              / NORM,  1.0f);
    out[109] = std::min(to_call                / NORM,  1.0f);
    out[110] = std::min(state.stacks[player]   / STACK, 1.0f);
    out[111] = std::min(state.stacks[1-player] / STACK, 1.0f);

    static constexpr int HIST_START = 112;
    static constexpr int HIST_SLOTS = 8;
    int hist_begin = std::max(0, state.action_count - HIST_SLOTS);
    for(int i = hist_begin; i < state.action_count; ++i) {
        int slot  = HIST_START + (i - hist_begin);
        int8_t act = state.action_history[i];
        if(act >= 0 && act < 4)
            out[slot] = NLHE_ACTION_ENC[act];
    }

    int r0 = card_rank(c0), r1 = card_rank(c1);
    int s0 = card_suit(c0), s1 = card_suit(c1);
    int rh = std::max(r0,r1), rl = std::min(r0,r1);
    bool suited = (s0 == s1) && (c0 != c1);
    out[120] = preflop_equity(rh, rl, suited);
    out[121] = board_strength(state, player);
}

float NLHEStateEncoder::board_strength(const NLHEState& state, int player) {
    int n_visible = BOARD_CARDS_BY_STREET[state.street];
    if(n_visible < 3) return 0.0f;
    int8_t cards[7];
    cards[0] = state.hole_cards[player][0];
    cards[1] = state.hole_cards[player][1];
    for(int i = 0; i < n_visible && i < 5; ++i) cards[2+i] = state.board[i];
    int n = 2 + std::min(n_visible, 5);
    int32_t score = HandEvaluator::evaluate(cards, n);
    constexpr float MAX_SCORE = static_cast<float>(8 << 24);
    return std::min((float)score / MAX_SCORE, 1.0f);
}

std::vector<float> NLHEStateEncoder::encode_vec(const NLHEState& state, int player) {
    std::vector<float> v(STATE_SIZE);
    encode(state, player, v.data());
    return v;
}

#ifdef CFR_TORCH_AVAILABLE
torch::Tensor NLHEStateEncoder::encode_tensor(const NLHEState& state, int player) {
    auto t = torch::zeros({1, STATE_SIZE});
    encode(state, player, t.data_ptr<float>());
    return t;
}
#endif

// ── TorchModel ────────────────────────────────────────────────────────────────

bool TorchModel::load(const std::string& path) {
#ifdef CFR_TORCH_AVAILABLE
    try {
        module_ = torch::jit::load(path);
        module_.eval();
        loaded_ = true;
        return true;
    } catch(const c10::Error&) { loaded_=false; return false; }
#else
    (void)path; return false;
#endif
}

std::vector<float> TorchModel::forward(const std::vector<float>& sv, int max_actions) const {
#ifdef CFR_TORCH_AVAILABLE
    if(!loaded_) return {};
    auto t = torch::from_blob(
        const_cast<float*>(sv.data()),
        {1,(long)sv.size()}, torch::kFloat32).clone();
    return forward_tensor(t, max_actions);
#else
    (void)sv;(void)max_actions; return {};
#endif
}

#ifdef CFR_TORCH_AVAILABLE
std::vector<float> TorchModel::forward_tensor(torch::Tensor input, int max_actions) const {
    if(!loaded_) return {};
    torch::NoGradGuard ng;
    auto out = module_.forward({input}).toTensor().squeeze(0).slice(0, 0, max_actions);

    // Regret matching: clamp negative regrets to 0, normalise
    auto pos   = out.clamp_min(0.0f);
    float total = pos.sum().item<float>();

    std::vector<float> probs(max_actions);
    if(total > 1e-7f) {
        auto p = pos / total;
        std::copy(p.data_ptr<float>(), p.data_ptr<float>() + max_actions, probs.begin());
    } else {
        std::fill(probs.begin(), probs.end(), 1.0f / max_actions);
    }
    return probs;
}
#endif

} // namespace cfr