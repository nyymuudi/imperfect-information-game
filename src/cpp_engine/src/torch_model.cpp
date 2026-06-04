#include "torch_model.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <fstream>
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

    bool loaded() const { return loaded_; }

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
        return !table_.empty();
    }

    bool loaded_ = false;
    std::unordered_map<std::string, float> table_;
};

}  // namespace

// ── NLHEStateEncoder ──────────────────────────────────────────────────────────
// NLHE_ACTION_ENC is defined in nlhe_game.hpp — do NOT redefine here.

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
    out[109] = std::min(to_call               / NORM,  1.0f);
    out[110] = std::min(state.stacks[player]   / STACK, 1.0f);
    out[111] = std::min(state.stacks[1-player] / STACK, 1.0f);

    static constexpr int HIST_START = 112;
    static constexpr int HIST_SLOTS = 8;
    int hist_begin = std::max(0, state.action_count - HIST_SLOTS);
    for(int i = hist_begin; i < state.action_count; ++i) {
        int slot   = HIST_START + (i - hist_begin);
        int8_t act = state.action_history[i];
        if(act >= 0 && act < 4)
            out[slot] = NLHE_ACTION_ENC[act];   // defined in nlhe_game.hpp
    }

    // dim [120]: preflop equity. Parity path: look up the deterministic table
    // Python wrote; only if it is unavailable do we fall back to the heuristic.
    int r0 = card_rank(c0), r1 = card_rank(c1);
    int s0 = card_suit(c0), s1 = card_suit(c1);
    int rh = std::max(r0,r1), rl = std::min(r0,r1);
    bool suited = (s0 == s1) && (c0 != c1);
    bool pair   = (r0 == r1);
    float eq = -1.0f;
    {
        const auto& tbl = PreflopEquityTable::instance();
        if (tbl.loaded()) {
            eq = tbl.lookup(canonical_class(rh, rl, suited, pair));
        }
    }
    out[120] = (eq >= 0.0f) ? eq : preflop_equity_heuristic(rh, rl, suited);
    out[121] = board_strength(state, player);

    // dim [122]: pot odds = to_call / (pot + to_call)
    float pot_plus_call = state.pot + to_call;
    out[122] = (pot_plus_call > 1e-6f) ? (to_call / pot_plus_call) : 0.0f;

    // dim [123]: SPR = min(stacks) / pot, normalised (cap at 10)
    float eff_stack = std::min(state.stacks[0], state.stacks[1]);
    out[123] = (state.pot > 1e-6f)
               ? std::min(eff_stack / state.pot, 10.0f) / 10.0f
               : 1.0f;

    // Quantise the continuous dims to the SAME 1e-6 grid the Python encoder
    // uses on output, so identical nodes hash/group identically across
    // implementations (DeepCFRSolver._collapse_by_state + NLHE state_key).
    for (int i = 108; i < STATE_SIZE; ++i)
        out[i] = std::round(out[i] * 1e6f) / 1e6f;
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

    // NORMALISATION PARITY NOTE
    // -------------------------
    // Python normalises evaluate_7card by MAX_HAND_SCORE = _pack(8, 12), where
    // _pack uses a base-15 positional scheme. This C++ HandEvaluator uses a
    // different packing (encode_score: category<<24 | ranks<<...), so the raw
    // integer scores are NOT comparable to Python's even though both rank hands
    // identically. dim 121 is therefore a MONOTONE-but-not-bit-identical
    // feature across implementations; this is an accepted residual documented
    // in tests/test_parity.py (the parity test compares dims 0-111, not 120-121).
    //
    // We normalise by THIS evaluator's true maximum — straight flush, ace high:
    // encode_score(8, {12}) = (8<<24) | (12<<20) — so the value lands in [0,1]
    // without saturating the top category (the old 8<<24 constant let a royal
    // flush exceed 1.0 before clamping, collapsing the strongest hands).
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