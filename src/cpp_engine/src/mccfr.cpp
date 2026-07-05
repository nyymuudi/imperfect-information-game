#include "mccfr.hpp"
#include <numeric>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace cfr {

// ── ReservoirBuffer ───────────────────────────────────────────────────────────

template <typename T>
bool ReservoirBuffer<T>::insert(const T& sample) {
    if (size_ < capacity_) {
        data_.push_back(sample);
        ++size_;
        ++total_seen_;
        return true;
    }
    std::uniform_int_distribution<size_t> dist(0, total_seen_);
    size_t idx = dist(rng_);
    ++total_seen_;
    if (idx < capacity_) {
        data_[idx] = sample;
        return true;
    }
    return false;
}

template class ReservoirBuffer<RegretSample>;
template class ReservoirBuffer<StrategySample>;

// ── MCCFREngine ───────────────────────────────────────────────────────────────

MCCFREngine::MCCFREngine(const TraversalConfig& config)
    : config_(config),
      regret_buf_(config.regret_capacity),
      strategy_buf_(config.strategy_capacity),
      rng_(config.seed),
      deals_(LeducGame::all_deals())
{}

void MCCFREngine::run_traversals_uniform(int traversing_player) {
    StrategyFn uniform_fn = [](const std::string&,
                                const std::vector<Action>& actions) {
        return std::vector<float>(actions.size(), 1.0f / actions.size());
    };
    run_traversals(traversing_player, uniform_fn);
}

void MCCFREngine::run_full_traversal_uniform(int traversing_player) {
    StrategyFn uniform_fn = [](const std::string&,
                                const std::vector<Action>& actions) {
        return std::vector<float>(actions.size(), 1.0f / actions.size());
    };
    run_full_traversal(traversing_player, uniform_fn);
}

void MCCFREngine::run_traversals(int traversing_player,
                                  const StrategyFn& strategy_fn) {
    std::uniform_int_distribution<size_t> deal_dist(0, deals_.size() - 1);
    for (int t = 0; t < config_.n_traversals; ++t) {
        const LeducDeal& deal = deals_[deal_dist(rng_)];
        LeducState root = LeducGame::initial_state(deal);
        traverse(root, traversing_player, 1.0f, 1.0f, strategy_fn);
    }
}

void MCCFREngine::run_full_traversal(int traversing_player,
                                     const StrategyFn& strategy_fn) {
    // Vanilla CFR: iterate every deal once, weight by its chance probability.
    for (const auto& deal : deals_) {
        LeducState root = LeducGame::initial_state(deal);
        traverse_full(root, traversing_player, deal.probability, strategy_fn);
    }
}

std::vector<float> MCCFREngine::regret_match(
    const std::string& info_set,
    const std::vector<Action>& legal_actions,
    const StrategyFn& strategy_fn)
{
    return strategy_fn(info_set, legal_actions);
}

void MCCFREngine::accumulate_cfrplus(const std::string& iset,
                                     const std::vector<Action>& legal,
                                     const std::vector<float>& instant_regret) {
    auto& e = cfrplus_[iset];
    e.n_actions = static_cast<int8_t>(legal.size());
    for (size_t a = 0; a < legal.size() && a < MAX_ACTIONS; ++a) {
        // R(I) <- max(R(I) + r^t, 0)  — CFR+ clip, correct sign.
        e.R[a] = std::max(e.R[a] + instant_regret[a], 0.0f);
    }
    e.visits += 1;
}

// ── External-sampling traversal (MCCFR) ───────────────────────────────────────

float MCCFREngine::traverse(
    const LeducState& state,
    int               traversing_player,
    float             reach_prob_traverser,
    float             reach_prob_opponent,
    const StrategyFn& strategy_fn)
{
    if (state.terminal) {
        return (traversing_player == 0) ? state.payoff_p0 : -state.payoff_p0;
    }

    const int p = state.current_player;
    const auto legal = LeducGame::legal_actions(state);
    const int  n_actions = static_cast<int>(legal.size());
    const std::string iset = LeducGame::info_set_key(state, p);

    if (p != traversing_player) {
        std::vector<float> probs = regret_match(iset, legal, strategy_fn);
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int sampled_idx = dist(rng_);
        Action sampled_action = legal[sampled_idx];

        if (config_.collect_strategy) {
            for (int a = 0; a < n_actions; ++a) {
                StrategySample ss{};
                std::strncpy(ss.info_set, iset.c_str(), 31);
                ss.action      = static_cast<int8_t>(legal[a]);
                ss.probability = probs[a] * reach_prob_opponent;
                ss.iteration   = config_.iteration;
                strategy_buf_.insert(ss);
            }
        }

        LeducState next = LeducGame::apply_action(state, sampled_action);
        return traverse(next, traversing_player,
                        reach_prob_traverser,
                        reach_prob_opponent * probs[sampled_idx],
                        strategy_fn);
    }

    std::vector<float> probs = regret_match(iset, legal, strategy_fn);
    std::vector<float> action_values(n_actions, 0.0f);
    for (int a = 0; a < n_actions; ++a) {
        LeducState next = LeducGame::apply_action(state, legal[a]);
        action_values[a] = traverse(next, traversing_player,
                                     reach_prob_traverser * probs[a],
                                     reach_prob_opponent, strategy_fn);
    }
    float node_value = 0.0f;
    for (int a = 0; a < n_actions; ++a)
        node_value += probs[a] * action_values[a];

    std::vector<float> instant(n_actions);
    for (int a = 0; a < n_actions; ++a)
        instant[a] = reach_prob_opponent * (action_values[a] - node_value);

    if (config_.target == RegretTarget::INSTANT) {
        // Legacy path: emit instantaneous regret per (infoset, action).
        for (int a = 0; a < n_actions; ++a) {
            RegretSample rs{};
            std::strncpy(rs.info_set, iset.c_str(), 31);
            rs.action    = static_cast<int8_t>(legal[a]);
            rs.regret    = instant[a];
            rs.iteration = config_.iteration;
            regret_buf_.insert(rs);
        }
    } else {
        // CFR+ path: fold into the persistent accumulator. Emission happens
        // once per iteration via emit_cfrplus_targets().
        accumulate_cfrplus(iset, legal, instant);
    }

    return node_value;
}

// ── Full (vanilla) traversal — deterministic, for parity testing ──────────────

float MCCFREngine::traverse_full(
    const LeducState& state,
    int               traversing_player,
    float             reach_prob_opponent,
    const StrategyFn& strategy_fn)
{
    if (state.terminal) {
        return (traversing_player == 0) ? state.payoff_p0 : -state.payoff_p0;
    }

    const int p = state.current_player;
    const auto legal = LeducGame::legal_actions(state);
    const int  n_actions = static_cast<int>(legal.size());
    const std::string iset = LeducGame::info_set_key(state, p);
    std::vector<float> probs = regret_match(iset, legal, strategy_fn);

    // Expand ALL actions regardless of who acts (vanilla CFR).
    std::vector<float> action_values(n_actions, 0.0f);
    for (int a = 0; a < n_actions; ++a) {
        LeducState next = LeducGame::apply_action(state, legal[a]);
        float opp_reach = (p == traversing_player)
                          ? reach_prob_opponent
                          : reach_prob_opponent * probs[a];
        action_values[a] = traverse_full(next, traversing_player,
                                          opp_reach, strategy_fn);
    }

    float node_value = 0.0f;
    for (int a = 0; a < n_actions; ++a)
        node_value += probs[a] * action_values[a];

    if (p == traversing_player) {
        std::vector<float> instant(n_actions);
        for (int a = 0; a < n_actions; ++a)
            instant[a] = reach_prob_opponent * (action_values[a] - node_value);
        if (config_.target == RegretTarget::INSTANT) {
            for (int a = 0; a < n_actions; ++a) {
                RegretSample rs{};
                std::strncpy(rs.info_set, iset.c_str(), 31);
                rs.action    = static_cast<int8_t>(legal[a]);
                rs.regret    = instant[a];
                rs.iteration = config_.iteration;
                regret_buf_.insert(rs);
            }
        } else {
            accumulate_cfrplus(iset, legal, instant);
        }
    }

    return node_value;
}

// ── Emit CFR+ targets (once per iteration) ────────────────────────────────────

void MCCFREngine::emit_cfrplus_targets() {
    for (const auto& kv : cfrplus_) {
        const std::string& iset = kv.first;
        const CfrPlusEntry& e   = kv.second;
        if (e.visits <= 0) continue;
        const float inv = 1.0f / static_cast<float>(e.visits);
        for (int a = 0; a < e.n_actions && a < MAX_ACTIONS; ++a) {
            RegretSample rs{};
            std::strncpy(rs.info_set, iset.c_str(), 31);
            rs.action    = static_cast<int8_t>(a);
            rs.regret    = e.R[a] * inv;     // R(I)/visits(I)
            rs.iteration = config_.iteration;
            regret_buf_.insert(rs);
        }
    }
}

// ── Buffer export ─────────────────────────────────────────────────────────────

MCCFREngine::BufferExport MCCFREngine::export_regret_buffer() const {
    BufferExport exp;
    const size_t n = regret_buf_.size();
    exp.info_sets.reserve(n); exp.actions.reserve(n);
    exp.values.reserve(n);    exp.iterations.reserve(n);
    const auto* data = regret_buf_.data();
    for (size_t i = 0; i < n; ++i) {
        exp.info_sets.push_back(std::string(data[i].info_set));
        exp.actions.push_back(data[i].action);
        exp.values.push_back(data[i].regret);
        exp.iterations.push_back(data[i].iteration);
    }
    return exp;
}

MCCFREngine::BufferExport MCCFREngine::export_strategy_buffer() const {
    BufferExport exp;
    const size_t n = strategy_buf_.size();
    exp.info_sets.reserve(n); exp.actions.reserve(n);
    exp.values.reserve(n);    exp.iterations.reserve(n);
    const auto* data = strategy_buf_.data();
    for (size_t i = 0; i < n; ++i) {
        exp.info_sets.push_back(std::string(data[i].info_set));
        exp.actions.push_back(data[i].action);
        exp.values.push_back(data[i].probability);
        exp.iterations.push_back(data[i].iteration);
    }
    return exp;
}

} // namespace cfr

#include "nlhe_mccfr.hpp"

namespace cfr {
template class ReservoirBuffer<NLHERegretSample>;
template class ReservoirBuffer<NLHEStrategySample>;
} // namespace cfr