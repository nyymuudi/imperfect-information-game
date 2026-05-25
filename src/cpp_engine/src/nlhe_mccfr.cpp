#include "nlhe_mccfr.hpp"
#include <cstring>
#include <algorithm>
#include <numeric>

namespace cfr {

NLHEMCCFREngine::NLHEMCCFREngine(const NLHETraversalConfig& config)
    : config_(config),
      regret_buf_(config.regret_capacity),
      strategy_buf_(config.strategy_capacity),
      rng_(config.seed)
{}

bool NLHEMCCFREngine::load_model(const std::string& path) {
    return model_.load(path);
}

std::vector<float> NLHEMCCFREngine::uniform_strategy(int n) {
    return std::vector<float>(n, 1.0f / n);
}

std::vector<float> NLHEMCCFREngine::model_strategy(
    const NLHEState& state, int player,
    const std::vector<NLHEAction>& actions)
{
    if (!model_.loaded()) return uniform_strategy(actions.size());

#ifdef CFR_TORCH_AVAILABLE
    // C++ 6-action → Python 4-slot mapping:
    // FOLD(0)→0, CHECK(1)→1, CALL(2)→1, BET_HALF(3)→2, BET_POT(4)→2, ALL_IN(5)→3
    static const int CPP_TO_PY[6] = {0, 1, 1, 2, 2, 3};
    const int py_actions = 4;

    auto input = NLHEStateEncoder::encode_tensor(state, player);
    auto probs_full = model_.forward_tensor(input, py_actions);

    std::vector<float> result;
    result.reserve(actions.size());
    for (auto a : actions) {
        int py_slot = CPP_TO_PY[static_cast<int>(a)];
        result.push_back(py_slot < (int)probs_full.size() ? probs_full[py_slot] : 0.0f);
    }
    float total = 0.0f;
    for (float p : result) total += p;
    if (total > 1e-7f) for (float& p : result) p /= total;
    else std::fill(result.begin(), result.end(), 1.0f / result.size());
    return result;
#else
    return uniform_strategy(actions.size());
#endif
}

void NLHEMCCFREngine::run_traversals_uniform(int traversing_player) {
    NLHEStrategyFn fn = [](const std::string&, const std::vector<NLHEAction>& a) {
        return std::vector<float>(a.size(), 1.0f / a.size());
    };
    run_traversals(traversing_player, fn);
}

void NLHEMCCFREngine::run_traversals(int traversing_player, const NLHEStrategyFn& fn) {
    for (int t = 0; t < config_.n_traversals; ++t) {
        NLHEDeal deal = NLHEGame::sample_deal(rng_);
        traverse_cb(NLHEGame::initial_state(deal), traversing_player, 1.0f, 1.0f, fn);
    }
}

float NLHEMCCFREngine::traverse_cb(
    const NLHEState& state, int tp, float r_tp, float r_opp,
    const NLHEStrategyFn& fn)
{
    if (state.terminal)
        return (tp == 0) ? state.payoff_p0 : -state.payoff_p0;

    const int p = state.current_player;
    const auto legal = NLHEGame::legal_actions(state);
    const int  n     = static_cast<int>(legal.size());
    const std::string iset = NLHEGame::info_set_key(state, p);
    auto probs = fn(iset, legal);

    if (p != tp) {
        if (config_.collect_strategy) {
            for (int a = 0; a < n; ++a) {
                StrategySample ss{};
                std::strncpy(ss.info_set, iset.c_str(), 31);
                ss.action = static_cast<int8_t>(legal[a]);
                ss.probability = probs[a] * r_opp;
                ss.iteration = config_.iteration;
                strategy_buf_.insert(ss);
            }
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng_);
        return traverse_cb(NLHEGame::apply_action(state, legal[idx]),
                           tp, r_tp, r_opp * probs[idx], fn);
    }

    std::vector<float> vals(n);
    for (int a = 0; a < n; ++a)
        vals[a] = traverse_cb(NLHEGame::apply_action(state, legal[a]),
                              tp, r_tp * probs[a], r_opp, fn);
    float nv = 0.0f;
    for (int a = 0; a < n; ++a) nv += probs[a] * vals[a];
    for (int a = 0; a < n; ++a) {
        RegretSample rs{};
        std::strncpy(rs.info_set, iset.c_str(), 31);
        rs.action = static_cast<int8_t>(legal[a]);
        rs.regret = r_opp * (vals[a] - nv);
        rs.iteration = config_.iteration;
        regret_buf_.insert(rs);
    }
    return nv;
}

void NLHEMCCFREngine::run_traversals_model(int traversing_player) {
    for (int t = 0; t < config_.n_traversals; ++t) {
        NLHEDeal deal = NLHEGame::sample_deal(rng_);
        traverse_model(NLHEGame::initial_state(deal), traversing_player, 1.0f, 1.0f);
    }
}

float NLHEMCCFREngine::traverse_model(
    const NLHEState& state, int tp, float r_tp, float r_opp)
{
    if (state.terminal)
        return (tp == 0) ? state.payoff_p0 : -state.payoff_p0;

    const int p = state.current_player;
    const auto legal = NLHEGame::legal_actions(state);
    const int  n     = static_cast<int>(legal.size());
    auto probs = model_strategy(state, p, legal);

    if (p != tp) {
        if (config_.collect_strategy) {
            const std::string iset = NLHEGame::info_set_key(state, p);
            for (int a = 0; a < n; ++a) {
                StrategySample ss{};
                std::strncpy(ss.info_set, iset.c_str(), 31);
                ss.action = static_cast<int8_t>(legal[a]);
                ss.probability = probs[a] * r_opp;
                ss.iteration = config_.iteration;
                strategy_buf_.insert(ss);
            }
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int idx = dist(rng_);
        return traverse_model(NLHEGame::apply_action(state, legal[idx]),
                              tp, r_tp, r_opp * probs[idx]);
    }

    std::vector<float> vals(n);
    for (int a = 0; a < n; ++a)
        vals[a] = traverse_model(NLHEGame::apply_action(state, legal[a]),
                                 tp, r_tp * probs[a], r_opp);
    float nv = 0.0f;
    for (int a = 0; a < n; ++a) nv += probs[a] * vals[a];

    const std::string iset = NLHEGame::info_set_key(state, p);
    for (int a = 0; a < n; ++a) {
        RegretSample rs{};
        std::strncpy(rs.info_set, iset.c_str(), 31);
        rs.action = static_cast<int8_t>(legal[a]);
        rs.regret = r_opp * (vals[a] - nv);
        rs.iteration = config_.iteration;
        regret_buf_.insert(rs);
    }
    return nv;
}

MCCFREngine::BufferExport NLHEMCCFREngine::export_regret_buffer() const {
    MCCFREngine::BufferExport exp;
    const size_t n = regret_buf_.size();
    exp.info_sets.reserve(n); exp.actions.reserve(n);
    exp.values.reserve(n);    exp.iterations.reserve(n);
    const auto* d = regret_buf_.data();
    for (size_t i = 0; i < n; ++i) {
        exp.info_sets.push_back(std::string(d[i].info_set));
        exp.actions.push_back(d[i].action);
        exp.values.push_back(d[i].regret);
        exp.iterations.push_back(d[i].iteration);
    }
    return exp;
}

MCCFREngine::BufferExport NLHEMCCFREngine::export_strategy_buffer() const {
    MCCFREngine::BufferExport exp;
    const size_t n = strategy_buf_.size();
    exp.info_sets.reserve(n); exp.actions.reserve(n);
    exp.values.reserve(n);    exp.iterations.reserve(n);
    const auto* d = strategy_buf_.data();
    for (size_t i = 0; i < n; ++i) {
        exp.info_sets.push_back(std::string(d[i].info_set));
        exp.actions.push_back(d[i].action);
        exp.values.push_back(d[i].probability);
        exp.iterations.push_back(d[i].iteration);
    }
    return exp;
}

bool NLHEMCCFREngine::load_strategy_model(const std::string& path) {
    return strategy_model_.load(path);
}

std::vector<float> NLHEMCCFREngine::query_strategy(
    int hole0, int hole1, int street,
    const std::vector<int>& board,
    float pot, float to_call, float my_stack) const
{
    if (!strategy_model_.loaded()) return std::vector<float>(4, 0.25f);

    NLHEState s{};
    s.hole_cards[0][0] = static_cast<int8_t>(hole0);
    s.hole_cards[0][1] = static_cast<int8_t>(hole1);
    s.hole_cards[1][0] = 0; s.hole_cards[1][1] = 1; // dummy opponent
    s.street           = static_cast<int8_t>(street);
    for (int i = 0; i < 5 && i < (int)board.size(); ++i)
        s.board[i] = static_cast<int8_t>(board[i]);
    s.pot              = pot;
    s.stacks[0]        = my_stack;
    s.stacks[1]        = NLHE_STACK;
    s.street_invest[0] = (to_call > 0) ? 0.0f : NLHE_SB;
    s.street_invest[1] = (to_call > 0) ? to_call : NLHE_BB;
    s.current_player   = 0;
    s.action_count     = 0;

#ifdef CFR_TORCH_AVAILABLE
    auto input = NLHEStateEncoder::encode_tensor(s, 0);
    return strategy_model_.forward_tensor(input, 4);
#else
    return std::vector<float>(4, 0.25f);
#endif
}

std::vector<float> NLHEMCCFREngine::query_preflop_strategy(
    int hole0, int hole1) const
{
    return query_strategy(
        hole0, hole1, 0, {},
        NLHE_SB + NLHE_BB,
        NLHE_BB - NLHE_SB,
        NLHE_STACK - NLHE_SB
    );
}

} // namespace cfr