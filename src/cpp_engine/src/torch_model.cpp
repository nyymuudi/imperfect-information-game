#include "torch_model.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <fstream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <string>

#ifdef CFR_TORCH_AVAILABLE
#include <torch/script.h>
#endif

#ifdef CFR_ORT_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

namespace cfr {

// ── Canonical preflop equity table (parity with Python) ───────────────────────
//
// dim 120 must match Python NLHEEncoder, which fills it from
// abstraction/equity.canonical_preflop_equity (a deterministic per-class Monte
// Carlo). Re-running Monte Carlo here would NOT reproduce those exact values,
// so instead we LOAD the table Python writes to disk
// (abstraction/_equity_cache/preflop_equity_<sims>.json) and look hands up by
// canonical class. If the file is absent we fall back to the old heuristic so
// builds/tests without the table still run — but a trained pipeline always has
// it, giving exact dim-120 parity.
//
// The lookup key is the canonical class string ("AA", "AKs", "72o", ...),
// identical to Python's canonical_hand_class.

namespace {

const char* RANK_NAMES = "23456789TJQKA";

std::string canonical_class(int rank_high, int rank_low, bool suited, bool pair) {
    std::string s;
    s += RANK_NAMES[rank_high];
    s += RANK_NAMES[rank_low];
    if (!pair) s += (suited ? 's' : 'o');
    return s;
}

class PreflopEquityTable {
public:
    static const PreflopEquityTable& instance() {
        static PreflopEquityTable t;
        return t;
    }

    // Returns equity in [0,1] for the given canonical class, or -1 if the
    // table was not loaded (caller falls back to the heuristic).
    float lookup(const std::string& cls) const {
        if (!loaded_) return -1.0f;
        auto it = table_.find(cls);
        return (it == table_.end()) ? -1.0f : it->second;
    }

    bool  loaded()  const { return loaded_; }
    float eq_min()  const { return eq_min_; }
    float eq_max()  const { return eq_max_; }

private:
    PreflopEquityTable() { load(); }

    void load() {
        // Search the same candidate paths the Python package uses, relative to
        // a few plausible working directories. Best-effort: silent on failure.
        const char* candidates[] = {
            "src/abstraction/_equity_cache/preflop_equity_2000.json",
            "../src/abstraction/_equity_cache/preflop_equity_2000.json",
            "../../src/abstraction/_equity_cache/preflop_equity_2000.json",
            "abstraction/_equity_cache/preflop_equity_2000.json",
        };
        for (const char* path : candidates) {
            if (try_load(path)) { loaded_ = true; return; }
        }
        loaded_ = false;
    }

    // Minimal JSON object parser for {"AA": 0.84, ...} — values are plain
    // floats, keys are 2-3 char class strings. Avoids a JSON dependency.
    bool try_load(const char* path) {
        std::ifstream f(path);
        if (!f.good()) return false;
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        size_t i = 0;
        const size_t n = content.size();
        auto skip_ws = [&]() { while (i < n && (content[i]==' '||content[i]=='\n'||
                                                content[i]=='\t'||content[i]=='\r')) ++i; };
        skip_ws();
        if (i >= n || content[i] != '{') return false;
        ++i;
        while (i < n) {
            skip_ws();
            if (i < n && content[i] == '}') break;
            if (i >= n || content[i] != '"') return false;
            ++i;
            std::string key;
            while (i < n && content[i] != '"') key += content[i++];
            if (i >= n) return false;
            ++i;  // closing quote
            skip_ws();
            if (i >= n || content[i] != ':') return false;
            ++i;
            skip_ws();
            size_t start = i;
            while (i < n && content[i] != ',' && content[i] != '}') ++i;
            try {
                float v = std::stof(content.substr(start, i - start));
                table_[key] = v;
            } catch (...) { return false; }
            skip_ws();
            if (i < n && content[i] == ',') ++i;
        }
        if (!table_.empty()) {
            eq_min_ = std::numeric_limits<float>::infinity();
            eq_max_ = -std::numeric_limits<float>::infinity();
            for (const auto& kv : table_) {
                eq_min_ = std::min(eq_min_, kv.second);
                eq_max_ = std::max(eq_max_, kv.second);
            }
        }
        return !table_.empty();
    }

    bool loaded_ = false;
    float eq_min_ = 0.316f;   // fallback: weakest hand (72o) approx
    float eq_max_ = 0.842f;   // fallback: strongest hand (AA) approx
    std::unordered_map<std::string, float> table_;
};

}  // namespace

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
// Card-abstracted encoder. Layout — see torch_model.hpp comment block.
// NLHE_ACTION_ENC is defined in nlhe_game.hpp.

// Process-global scheme. Set from Python (bindings) before training so the
// MCCFR loop's encoder calls produce the right layout. Default: flat (v3).
static int g_scheme = 0;
void NLHEStateEncoder::set_scheme(int s) { g_scheme = s; }
int  NLHEStateEncoder::get_scheme()      { return g_scheme; }

void NLHEStateEncoder::encode(const NLHEState& state, int player, float* out) {
    const float STACK = state.cfg.starting_stack;
    const float NORM  = 2.0f * STACK;

    std::fill(out, out + STATE_SIZE, 0.0f);

    int c0 = state.hole_cards[player][0];
    int c1 = state.hole_cards[player][1];

    // dims [0:8]: preflop equity bucket one-hot — normalised to [eq_min, eq_max]
    // so all K bins are populated and AA/72o are K-1 bins apart.
    int r0 = card_rank(c0), r1 = card_rank(c1);
    int s0 = card_suit(c0), s1 = card_suit(c1);
    int rh = std::max(r0,r1), rl = std::min(r0,r1);
    bool suited = (s0 == s1) && (c0 != c1);
    bool pair   = (r0 == r1);
    float eq = -1.0f;
    float eq_min = 0.316f, eq_max = 0.842f;  // fallback constants
    {
        const auto& tbl = PreflopEquityTable::instance();
        if (tbl.loaded()) {
            eq = tbl.lookup(canonical_class(rh, rl, suited, pair));
            eq_min = tbl.eq_min();
            eq_max = tbl.eq_max();
        }
    }
    float equity = (eq >= 0.0f) ? eq : preflop_equity_heuristic(rh, rl, suited);
    float eq_range = eq_max - eq_min;
    float eq_norm  = (eq_range > 0.0f) ? ((equity - eq_min) / eq_range) : equity;
    int pf_bucket = std::min((int)(eq_norm * K_PREFLOP), K_PREFLOP - 1);
    if (pf_bucket < 0) pf_bucket = 0;
    out[pf_bucket] = 1.0f;

    // Dim offsets parameterised on K_BOARD (must match Python encoder).
    constexpr int BRD_OFF    = K_PREFLOP;               // 8
    constexpr int STREET_OFF = BRD_OFF + K_BOARD;       // 24 when K_BOARD=16
    constexpr int BET_OFF    = STREET_OFF + 4;          // 28
    constexpr int HIST_OFF   = BET_OFF + 4;             // 32
    constexpr int CONT_OFF   = HIST_OFF + 8;            // 40

    // Board bucket — scheme-dependent.
    //   Flat (g_scheme=0, v3 production): K_BOARD-way one-hot in [BRD_OFF:BRD_OFF+8].
    //   Tree (g_scheme=1, experimental v5): two-hot — super in [BRD_OFF:BRD_OFF+4],
    //                                       fine within super in [BRD_OFF+4:BRD_OFF+8].
    int n_visible = BOARD_CARDS_BY_STREET[state.street];
    float brd_str = board_strength(state, player);
    if (n_visible >= 3) {
        int8_t cards[7];
        cards[0] = state.hole_cards[player][0];
        cards[1] = state.hole_cards[player][1];
        for (int i = 0; i < n_visible && i < 5; ++i) cards[2+i] = state.board[i];
        int n_eval = 2 + std::min(n_visible, 5);
        int32_t score = HandEvaluator::evaluate(cards, n_eval);
        int category = (score >> 24) & 0xFF;

        // Rank multiplicities for sub-bin extraction (shared by both schemes).
        int counts[13] = {0};
        int top_rank  = 0;
        for (int i = 0; i < n_eval; ++i) {
            int r = cards[i] / 4;
            if (r >= 0 && r < 13) counts[r]++;
            if (r > top_rank) top_rank = r;
        }
        int pair_rank = -1;
        for (int r = 12; r >= 0; --r) {
            if (counts[r] >= 2) { pair_rank = r; break; }
        }

        if (g_scheme == 1 || g_scheme == 2 || g_scheme == 3) {
            // ── Tree / super / tree42 scheme ──────────────────────────────────
            // All three share super_idx computation; differ in fine layout:
            //   scheme 1 (tree):   4-hot fine at [BRD+4:BRD+8]
            //   scheme 2 (super):  no fine (zero)
            //   scheme 3 (tree42): 2-hot fine at [BRD+4:BRD+6] (collapsed half)
            int super_idx, fine_idx;
            if (category == 0) {
                super_idx = 0;
                if (top_rank == 12)      fine_idx = 3;   // A
                else if (top_rank == 11) fine_idx = 2;   // K
                else if (top_rank >= 8)  fine_idx = 1;   // T-Q
                else                     fine_idx = 0;   // 2-9
            } else if (category == 1) {
                super_idx = 1;
                if (pair_rank == 12)      fine_idx = 3;
                else if (pair_rank >= 10) fine_idx = 2;
                else if (pair_rank >= 7)  fine_idx = 1;
                else                      fine_idx = 0;
            } else if (category >= 2 && category <= 5) {
                super_idx = 2;
                fine_idx  = category - 2;   // {twopair, trips, straight, flush}
            } else {
                super_idx = 3;
                if (category == 6)      fine_idx = 0;    // FH
                else if (category == 7) fine_idx = 1;    // quads
                else                    fine_idx = (top_rank == 12) ? 3 : 2;  // royal vs SF
            }
            out[BRD_OFF + super_idx] = 1.0f;
            if (g_scheme == 1) {
                // Tree: 4-hot fine.
                out[BRD_OFF + 4 + fine_idx] = 1.0f;
            } else if (g_scheme == 3) {
                // Tree42: collapse fine to 2-bin (0,1 → 0; 2,3 → 1).
                int fine_2 = (fine_idx <= 1) ? 0 : 1;
                out[BRD_OFF + 4 + fine_2] = 1.0f;
            }
            // Super (g_scheme == 2): only super_idx set.
        } else {
            // ── Flat scheme (v3 production) ───────────────────────────────────
            int brd_bucket;
            if (category >= 6) {
                brd_bucket = 7;
            } else if (category == 4 || category == 5) {
                brd_bucket = 6;
            } else if (category == 3) {
                brd_bucket = 5;
            } else if (category == 2) {
                brd_bucket = 4;
            } else if (category == 1) {
                brd_bucket = (pair_rank >= 7) ? 3 : 2;
            } else {
                brd_bucket = (top_rank >= 9) ? 1 : 0;
            }
            out[BRD_OFF + brd_bucket] = 1.0f;
        }
    }

    // Street one-hot
    int street = std::min((int)state.street, 3);
    out[STREET_OFF + street] = 1.0f;

    // Betting scalars
    float to_call = state.street_invest[1-player] - state.street_invest[player];
    to_call = std::max(0.0f, to_call);
    out[BET_OFF + 0] = std::min(state.pot              / NORM,  1.0f);
    out[BET_OFF + 1] = std::min(to_call                / NORM,  1.0f);
    out[BET_OFF + 2] = std::min(state.stacks[player]   / STACK, 1.0f);
    out[BET_OFF + 3] = std::min(state.stacks[1-player] / STACK, 1.0f);

    // Action history (last 8 actions)
    static constexpr int HIST_SLOTS = 8;
    int hist_begin = std::max(0, state.action_count - HIST_SLOTS);
    for(int i = hist_begin; i < state.action_count; ++i) {
        int slot   = HIST_OFF + (i - hist_begin);
        int8_t act = state.action_history[i];
        if(act >= 0 && act < 4)
            out[slot] = NLHE_ACTION_ENC[act];
    }

    // Continuous tail
    out[CONT_OFF + 0] = equity;
    out[CONT_OFF + 1] = brd_str;
    float pot_plus_call = state.pot + to_call;
    out[CONT_OFF + 2] = (pot_plus_call > 1e-6f) ? (to_call / pot_plus_call) : 0.0f;
    float eff_stack = std::min(state.stacks[0], state.stacks[1]);
    out[CONT_OFF + 3] = (state.pot > 1e-6f)
                       ? std::min(eff_stack / state.pot, 10.0f) / 10.0f
                       : 1.0f;

    // Quantise BET_OFF..STATE_SIZE to FEATURE_QUANT grid. floor(x/q + 0.5)
    // matches numpy semantics on the 0.5 boundary (vs std::round).
    constexpr float FEATURE_QUANT = 0.05f;
    for (int i = BET_OFF; i < STATE_SIZE; ++i)
        out[i] = std::floor(out[i] / FEATURE_QUANT + 0.5f) * FEATURE_QUANT;
}

// NOTE: this file previously hosted a Monte Carlo EHS cache + canonical-key
// suit-isomorphism packing for board_strength. That path implemented v8's
// continuous board feature (MC EHS vs random opponent), which regressed v3 by
// +629 mbb/h in CRN+EV h2h (z=4.17). Reverted to the monotone hand-rank proxy
// below. The MC EHS scaffolding lived in git history before this cleanup pass.

float NLHEStateEncoder::board_strength(const NLHEState& state, int player) {
    // v3 production: monotone hand-rank proxy normalised by this evaluator's
    // max-possible packed score. Yksinkertainen, deterministinen, halpa.
    // MC EHS -versio (v8) regressoi v3:a +629 mbb/h h2h:ssa, joten palautettu.
    int n_visible = BOARD_CARDS_BY_STREET[state.street];
    if (n_visible < 3) return 0.0f;
    int8_t cards[7];
    cards[0] = state.hole_cards[player][0];
    cards[1] = state.hole_cards[player][1];
    for (int i = 0; i < n_visible && i < 5; ++i) cards[2 + i] = state.board[i];
    int n = 2 + std::min(n_visible, 5);
    int32_t score = HandEvaluator::evaluate(cards, n);
    // MAX_SCORE = straight flush, ace high = encode_score(8, {12}).
    constexpr float MAX_SCORE = static_cast<float>((8 << 24) | (12 << 20));
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

// ── OnnxStrategyModel ─────────────────────────────────────────────────────────

OnnxStrategyModel::OnnxStrategyModel()
#ifdef CFR_ORT_AVAILABLE
    : env_(ORT_LOGGING_LEVEL_WARNING, "strategy_model")
#endif
{}

bool OnnxStrategyModel::load(const std::string& path) {
#ifdef CFR_ORT_AVAILABLE
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);
        loaded_ = true;
        return true;
    } catch (const Ort::Exception&) {
        loaded_ = false;
        return false;
    }
#else
    (void)path;
    return false;
#endif
}

std::vector<float> OnnxStrategyModel::forward(
    const std::vector<float>& state_vec,
    int max_actions) const
{
#ifdef CFR_ORT_AVAILABLE
    if (!loaded_) return {};

    std::vector<float> mask(4, 0.0f);
    for (int i = 0; i < max_actions && i < 4; ++i) mask[i] = 1.0f;

    std::vector<int64_t> state_shape = {1, (int64_t)state_vec.size()};
    std::vector<int64_t> mask_shape  = {1, 4};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<Ort::Value, 2> inputs = {
        Ort::Value::CreateTensor<float>(mem,
            const_cast<float*>(state_vec.data()), state_vec.size(),
            state_shape.data(), state_shape.size()),
        Ort::Value::CreateTensor<float>(mem,
            mask.data(), mask.size(),
            mask_shape.data(), mask_shape.size()),
    };

    const char* input_names[]  = {"state", "action_mask"};
    const char* output_names[] = {"probs"};

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 1
    );

    auto* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + 4);
#else
    (void)state_vec; (void)max_actions;
    return {};
#endif
}

std::vector<float> OnnxStrategyModel::forward_batch(
    const std::vector<float>& states,
    const std::vector<int>&   max_actions_vec) const
{
#ifdef CFR_ORT_AVAILABLE
    if (!loaded_) return {};

    int batch      = (int)max_actions_vec.size();
    int state_size = (int)states.size() / batch;

    std::vector<float> masks(batch * 4, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int a = 0; a < max_actions_vec[b] && a < 4; ++a)
            masks[b * 4 + a] = 1.0f;

    std::vector<int64_t> state_shape = {batch, state_size};
    std::vector<int64_t> mask_shape  = {batch, 4};

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<Ort::Value, 2> inputs = {
        Ort::Value::CreateTensor<float>(mem,
            const_cast<float*>(states.data()), states.size(),
            state_shape.data(), state_shape.size()),
        Ort::Value::CreateTensor<float>(mem,
            masks.data(), masks.size(),
            mask_shape.data(), mask_shape.size()),
    };

    const char* input_names[]  = {"state", "action_mask"};
    const char* output_names[] = {"probs"};

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 2,
        output_names, 1
    );

    auto* data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(data, data + batch * 4);
#else
    (void)states; (void)max_actions_vec;
    return {};
#endif
}

} // namespace cfr