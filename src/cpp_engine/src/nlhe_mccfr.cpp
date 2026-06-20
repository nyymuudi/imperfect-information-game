#include "nlhe_mccfr.hpp"
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>

namespace cfr {

// ── EV-adjusted terminal values ──────────────────────────────────────────────
// At all-in showdown terminals, replacing the deterministic deal-specific
// runout with the equity over remaining board cards is a variance-reduction
// technique known in poker as "all-in adjusted" or "Sklansky bucks". For
// Deep CFR it has the same effect on regret targets: the same all-in spot
// played multiple times always gets the same expected value rather than a
// noisy realisation. Tighter regret targets → faster, cleaner convergence.
//
// Runtime cost: equity enumeration scales with remaining cards. Flop-all-in
// = C(45,2) = 990 evaluations. Turn-all-in = 44. Preflop-all-in = MC over
// 2000 samples. Cached by (hole_a, hole_b, board, n_visible) — all-in spots
// repeat aggressively during training so cache hit rate is very high.
static bool g_ev_adjusted_terminals = false;

// Cache key packs hole_a (2 cards) + hole_b (2 cards) + board (up to 5).
// Hole pairs are sorted independently so order within a pair doesn't matter.
namespace {
    inline uint64_t pack_equity_key(const int8_t* ha, const int8_t* hb,
                                     const int8_t* board, int n_visible) {
        int a0 = std::min((int)ha[0], (int)ha[1]);
        int a1 = std::max((int)ha[0], (int)ha[1]);
        int b0 = std::min((int)hb[0], (int)hb[1]);
        int b1 = std::max((int)hb[0], (int)hb[1]);
        // Canonical orientation: order players so a < b lexicographically.
        // This makes equity_for(a,b) == 1 - equity_for(b,a) consistent and
        // doubles cache hit rate.
        if (a0 > b0 || (a0 == b0 && a1 > b1)) {
            std::swap(a0, b0); std::swap(a1, b1);
        }
        uint64_t k = (uint64_t)(a0 & 0xFF)        // 8 bits
                   | ((uint64_t)(a1 & 0xFF) << 8)
                   | ((uint64_t)(b0 & 0xFF) << 16)
                   | ((uint64_t)(b1 & 0xFF) << 24);
        for (int i = 0; i < n_visible && i < 5; ++i) {
            k |= ((uint64_t)(uint8_t)board[i]) << (32 + i * 6);
        }
        // Sign bit indicates whether we swapped — caller uses this to flip.
        return k;
    }

    static std::unordered_map<uint64_t, float> g_equity_cache;
    constexpr size_t EQUITY_CACHE_MAX = 50'000'000ULL;

    inline uint64_t splitmix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // Compute p0's equity vs p1's specific hand over remaining board cards.
    // Cached. Returns float in [0, 1].
    float equity_p0_vs_p1(const int8_t* ha, const int8_t* hb,
                           const int8_t* board, int n_visible) {
        // Cache lookup with player swap.
        int a0 = std::min((int)ha[0], (int)ha[1]);
        int a1 = std::max((int)ha[0], (int)ha[1]);
        int b0 = std::min((int)hb[0], (int)hb[1]);
        int b1 = std::max((int)hb[0], (int)hb[1]);
        bool swapped = (a0 > b0 || (a0 == b0 && a1 > b1));

        uint64_t key = pack_equity_key(ha, hb, board, n_visible);
        auto it = g_equity_cache.find(key);
        if (it != g_equity_cache.end()) {
            return swapped ? (1.0f - it->second) : it->second;
        }

        // Compute.
        bool dead[52] = {false};
        dead[ha[0]] = dead[ha[1]] = dead[hb[0]] = dead[hb[1]] = true;
        for (int i = 0; i < n_visible; ++i) dead[board[i]] = true;
        int live[52], live_n = 0;
        for (int c = 0; c < 52; ++c) if (!dead[c]) live[live_n++] = c;

        int needed = 5 - n_visible;
        int8_t hand_a[7], hand_b[7];
        hand_a[0] = ha[0]; hand_a[1] = ha[1];
        hand_b[0] = hb[0]; hand_b[1] = hb[1];
        for (int i = 0; i < n_visible; ++i) {
            hand_a[2 + i] = board[i];
            hand_b[2 + i] = board[i];
        }

        float wins = 0.0f;
        int   total = 0;

        if (needed == 0) {
            int32_t sa = HandEvaluator::evaluate(hand_a, 7);
            int32_t sb = HandEvaluator::evaluate(hand_b, 7);
            wins = (sa > sb) ? 1.0f : (sa < sb ? 0.0f : 0.5f);
            total = 1;
        } else if (needed == 1) {
            for (int i = 0; i < live_n; ++i) {
                hand_a[2 + n_visible] = (int8_t)live[i];
                hand_b[2 + n_visible] = (int8_t)live[i];
                int32_t sa = HandEvaluator::evaluate(hand_a, 7);
                int32_t sb = HandEvaluator::evaluate(hand_b, 7);
                if (sa > sb) wins += 1.0f;
                else if (sa == sb) wins += 0.5f;
                ++total;
            }
        } else if (needed == 2) {
            for (int i = 0; i < live_n; ++i) {
                for (int j = i + 1; j < live_n; ++j) {
                    hand_a[2 + n_visible]     = (int8_t)live[i];
                    hand_a[2 + n_visible + 1] = (int8_t)live[j];
                    hand_b[2 + n_visible]     = (int8_t)live[i];
                    hand_b[2 + n_visible + 1] = (int8_t)live[j];
                    int32_t sa = HandEvaluator::evaluate(hand_a, 7);
                    int32_t sb = HandEvaluator::evaluate(hand_b, 7);
                    if (sa > sb) wins += 1.0f;
                    else if (sa == sb) wins += 0.5f;
                    ++total;
                }
            }
        } else {
            // Preflop / flop all-in needing 3-5 more cards: MC.
            constexpr int N_SIMS = 2000;
            uint64_t seed = splitmix64(key);
            for (int s = 0; s < N_SIMS; ++s) {
                // Partial Fisher-Yates.
                int pool[52];
                for (int i = 0; i < live_n; ++i) pool[i] = live[i];
                for (int i = 0; i < needed; ++i) {
                    seed = splitmix64(seed);
                    int r = i + int(seed % uint64_t(live_n - i));
                    std::swap(pool[i], pool[r]);
                    hand_a[2 + n_visible + i] = (int8_t)pool[i];
                    hand_b[2 + n_visible + i] = (int8_t)pool[i];
                }
                int32_t sa = HandEvaluator::evaluate(hand_a, 7);
                int32_t sb = HandEvaluator::evaluate(hand_b, 7);
                if (sa > sb) wins += 1.0f;
                else if (sa == sb) wins += 0.5f;
                ++total;
            }
        }

        float eq = (total > 0) ? (wins / float(total)) : 0.5f;
        if (g_equity_cache.size() < EQUITY_CACHE_MAX) {
            g_equity_cache[key] = eq;
        }
        return swapped ? (1.0f - eq) : eq;
    }
}  // namespace

// Public API surfaced via bindings.
void NLHEMCCFREngine::set_ev_adjusted_terminals(bool on) {
    g_ev_adjusted_terminals = on;
}
bool NLHEMCCFREngine::get_ev_adjusted_terminals() {
    return g_ev_adjusted_terminals;
}

// Compute the terminal value for player 0. Falls back to state.payoff_p0
// unless EV-adjusted mode is on AND this terminal was an all-in showdown
// (both hands present, betting ended before the river).
inline float terminal_payoff_p0(const NLHEState& s) {
    if (!g_ev_adjusted_terminals) return s.payoff_p0;
    if (s.folded[0] || s.folded[1])  return s.payoff_p0;
    if (s.all_in_street < 0 || s.all_in_street >= 3) return s.payoff_p0;
    int n_visible = BOARD_CARDS_BY_STREET[s.all_in_street];
    float eq_p0 = equity_p0_vs_p1(
        s.hole_cards[0], s.hole_cards[1], s.board, n_visible);
    float inv0 = s.cfg.starting_stack - s.stacks[0];
    float inv1 = s.cfg.starting_stack - s.stacks[1];
    float pot = inv0 + inv1;
    return eq_p0 * pot - inv0;
}

NLHEMCCFREngine::NLHEMCCFREngine(const NLHETraversalConfig& config)
    : config_(config),
      regret_buf_(config.regret_capacity),
      strategy_buf_(config.strategy_capacity),
      rng_(config.seed)
{}

bool NLHEMCCFREngine::load_model(const std::string& p) { return model_.load(p); }
bool NLHEMCCFREngine::load_strategy_model(const std::string& p) { return strategy_model_.load(p); }

std::vector<float> NLHEMCCFREngine::uniform_strategy(int n) {
    return std::vector<float>(n, 1.0f / n);
}

// ── State-vector hash key ─────────────────────────────────────────────────────
// We accumulate CFR+ regret keyed on the ENCODER OUTPUT, not info_set_key.
// Rationale: the network input, the Python _collapse_by_state grouping, and the
// accumulation granularity must all be the SAME quantity. info_set_key quantises
// the pot into buckets and uses the full action history, so it groups states
// DIFFERENTLY than the encoder (continuous pot, last-8 actions). Keying on the
// encoder output keeps one consistent granularity through the whole pipeline.
//
// The encoder is a deterministic function of the game state, so the same node
// yields a bit-identical vector; we still quantise to a fixed grid before
// hashing as a guard against any platform float noise, matching the bit-exact
// grouping np.unique performs on the Python side at full float precision.
std::string NLHEMCCFREngine::state_key(const float* sv) {
    // Quantise to 1e-6 grid; round-half-away-from-zero into int32, pack as bytes.
    constexpr int N = NLHEStateEncoder::STATE_SIZE;
    std::string key;
    key.resize(N * sizeof(int32_t));
    char* out = key.data();
    for (int i = 0; i < N; ++i) {
        int32_t q = static_cast<int32_t>(std::lround(sv[i] * 1e6f));
        std::memcpy(out + i * sizeof(int32_t), &q, sizeof(int32_t));
    }
    return key;
}

// ── Model strategy: direct 4-slot inference, no remapping ────────────────────
std::vector<float> NLHEMCCFREngine::model_strategy(
    const NLHEState& state, int player,
    const std::vector<NLHEAction>& actions)
{
    if(!model_.loaded()) return uniform_strategy(actions.size());
#ifdef CFR_TORCH_AVAILABLE
    auto input = NLHEStateEncoder::encode_tensor(state, player);
    auto probs = model_.forward_tensor(input, config_.max_actions);
    std::vector<float> result;
    result.reserve(actions.size());
    for(auto a : actions) {
        // Remap ALL_IN to slot 3 in single-raise mode (max_actions==4) so
        // the all-in probability is read from a trained slot, not from out-
        // of-bounds slot 5. Multi-raise keeps enum value as-is.
        int slot = nlhe_action_to_slot(a, config_.max_actions);
        result.push_back(slot < (int)probs.size() ? probs[slot] : 0.0f);
    }
    float total = 0.0f;
    for(float p : result) total += p;
    if(total > 1e-7f) for(float& p : result) p /= total;
    else std::fill(result.begin(), result.end(), 1.0f/result.size());
    return result;
#else
    return uniform_strategy(actions.size());
#endif
}

void NLHEMCCFREngine::run_traversals_uniform(int tp) {
    NLHEStrategyFn fn=[](const std::string&,const std::vector<NLHEAction>&a){
        return std::vector<float>(a.size(),1.0f/a.size());};
    run_traversals(tp,fn);
}

void NLHEMCCFREngine::run_traversals(int tp, const NLHEStrategyFn& fn) {
    for(int t=0;t<config_.n_traversals;++t){
        NLHEDeal deal=NLHEGame::sample_deal(rng_);
        traverse_cb(NLHEGame::initial_state(deal,config_.game_cfg),tp,1.0f,1.0f,fn);
    }
}

// ── CFR+ accumulation (state-vector-keyed) ────────────────────────────────────

void NLHEMCCFREngine::accumulate_cfrplus(
    const float* state_vec,
    const std::vector<NLHEAction>& legal,
    const std::vector<float>& instant_regret)
{
    std::string k = state_key(state_vec);
    auto& e = cfrplus_[k];
    if (e.visits == 0) {
        // First time we see this state vector: store it for emission.
        std::memcpy(e.state.data(), state_vec,
                    NLHEStateEncoder::STATE_SIZE * sizeof(float));
        e.n_actions = static_cast<int8_t>(legal.size());
    }
    for (size_t i = 0; i < legal.size() && i < NLHE_NUM_ACTIONS; ++i) {
        // R(I) <- max(R(I) + r^t, 0). Index by remapped slot so ALL_IN
        // (enum 5) lands in slot 3 when max_actions==4 (single-raise puu)
        // and gets actual training signal instead of being filtered out.
        int slot = nlhe_action_to_slot(legal[i], config_.max_actions);
        if (slot >= 0 && slot < NLHE_NUM_ACTIONS)
            e.R[slot] = std::max(e.R[slot] + instant_regret[i], 0.0f);
    }
    e.visits += 1;
}

void NLHEMCCFREngine::emit_or_accumulate(
    const float* state_vec,
    const std::vector<NLHEAction>& legal,
    const std::vector<float>& instant_regret)
{
    if (config_.target == RegretTarget::INSTANT) {
        for (size_t a = 0; a < legal.size(); ++a) {
            NLHERegretSample rs{};
            std::memcpy(rs.state, state_vec,
                        NLHEStateEncoder::STATE_SIZE * sizeof(float));
            // Same remap as in accumulate_cfrplus: ALL_IN→slot 3 in
            // single-raise mode so the Python-side `a_np < max_actions`
            // filter keeps it.
            rs.action    = static_cast<int8_t>(
                nlhe_action_to_slot(legal[a], config_.max_actions));
            rs.regret    = instant_regret[a];
            rs.iteration = config_.iteration;
            regret_buf_.insert(rs);
        }
    } else {
        accumulate_cfrplus(state_vec, legal, instant_regret);
    }
}

void NLHEMCCFREngine::emit_cfrplus_targets() {
    for (const auto& kv : cfrplus_) {
        const NLHECfrPlusEntry& e = kv.second;
        if (e.visits <= 0) continue;
        const float inv = 1.0f / static_cast<float>(e.visits);
        // Emit one sample per ACTION SLOT that is legal at this state. We don't
        // store the legal-action list, but R[slot] is only ever nonzero/updated
        // for legal slots; emit all slots that were touched (visits>0 implies
        // every legal slot was updated each visit). To preserve the action set,
        // emit every slot 0..n_actions-1 by enum order is NOT correct because
        // legal actions are a subset. Instead emit slots whose R could have been
        // written: we emit all 4 slots but the Python side masks a >= max_actions
        // and _collapse_by_state sums only present (state,action) pairs. Slots
        // never legal here stay 0 and are harmless (they match the network's
        // masked output of 0). Emit exactly the slots 0..NLHE_NUM_ACTIONS-1.
        for (int slot = 0; slot < NLHE_NUM_ACTIONS; ++slot) {
            NLHERegretSample rs{};
            std::memcpy(rs.state, e.state.data(),
                        NLHEStateEncoder::STATE_SIZE * sizeof(float));
            rs.action    = static_cast<int8_t>(slot);
            rs.regret    = e.R[slot] * inv;     // R(slot)/visits
            rs.iteration = config_.iteration;
            regret_buf_.insert(rs);
        }
    }
}

// ── External-sampling traversal (callback strategy) ───────────────────────────

float NLHEMCCFREngine::traverse_cb(
    const NLHEState& state,int tp,float r_tp,float r_opp,
    const NLHEStrategyFn& fn)
{
    if(state.terminal) { float v = terminal_payoff_p0(state); return (tp==0)?v:-v; }
    const int p=state.current_player;
    const auto legal=NLHEGame::legal_actions(state);
    const int  n=(int)legal.size();
    const std::string iset=NLHEGame::info_set_key(state,p);
    auto probs=fn(iset,legal);

    if(p!=tp){
        if(config_.collect_strategy)
            for(int a=0;a<n;++a){
                NLHEStrategySample ss{};
                NLHEStateEncoder::encode(state, p, ss.state);
                ss.action=static_cast<int8_t>(
                    nlhe_action_to_slot(legal[a], config_.max_actions));
                ss.probability=probs[a]*r_opp;
                ss.iteration=config_.iteration;
                strategy_buf_.insert(ss);
            }
        std::discrete_distribution<int>d(probs.begin(),probs.end());
        int idx=d(rng_);
        return traverse_cb(NLHEGame::apply_action(state,legal[idx]),
                           tp,r_tp,r_opp*probs[idx],fn);
    }

    std::vector<float>vals(n);
    for(int a=0;a<n;++a)
        vals[a]=traverse_cb(NLHEGame::apply_action(state,legal[a]),
                            tp,r_tp*probs[a],r_opp,fn);
    float nv=0.0f;
    for(int a=0;a<n;++a) nv+=probs[a]*vals[a];

    float sv[NLHEStateEncoder::STATE_SIZE];
    NLHEStateEncoder::encode(state, p, sv);
    std::vector<float> instant(n);
    for(int a=0;a<n;++a) instant[a]=r_opp*(vals[a]-nv);
    emit_or_accumulate(sv, legal, instant);
    return nv;
}

// ── External-sampling traversal (LibTorch model strategy) ─────────────────────

void NLHEMCCFREngine::run_traversals_model(int tp) {
    for(int t=0;t<config_.n_traversals;++t){
        NLHEDeal deal=NLHEGame::sample_deal(rng_);
        traverse_model(NLHEGame::initial_state(deal,config_.game_cfg),tp,1.0f,1.0f);
    }
}

float NLHEMCCFREngine::traverse_model(
    const NLHEState& state,int tp,float r_tp,float r_opp)
{
    if(state.terminal) { float v = terminal_payoff_p0(state); return (tp==0)?v:-v; }
    const int p=state.current_player;
    const auto legal=NLHEGame::legal_actions(state);
    const int  n=(int)legal.size();
    auto probs=model_strategy(state,p,legal);

    if(p!=tp){
        if(config_.collect_strategy){
            for(int a=0;a<n;++a){
                NLHEStrategySample ss{};
                NLHEStateEncoder::encode(state, p, ss.state);
                ss.action=static_cast<int8_t>(
                    nlhe_action_to_slot(legal[a], config_.max_actions));
                ss.probability=probs[a]*r_opp;
                ss.iteration=config_.iteration;
                strategy_buf_.insert(ss);
            }
        }
        std::discrete_distribution<int>d(probs.begin(),probs.end());
        int idx=d(rng_);
        return traverse_model(NLHEGame::apply_action(state,legal[idx]),
                              tp,r_tp,r_opp*probs[idx]);
    }

    // Pluribus-style dynamic pruning (Brown & Sandholm 2019, Sec. S6):
    // skip rollouts for actions whose blueprint probability is below
    // prune_threshold. node value is computed over visited actions only,
    // renormalised; pruned actions get no regret update. Pruning activates
    // only when iteration >= prune_after_iter to avoid early-iter signal loss.
    const bool prune_enabled =
        (config_.prune_threshold > 0.0f) &&
        (config_.iteration >= config_.prune_after_iter);

    std::vector<float> vals(n, 0.0f);
    std::vector<bool>  visited(n, true);
    float visited_prob_sum = 0.0f;

    for(int a=0;a<n;++a) {
        if (prune_enabled && probs[a] < config_.prune_threshold) {
            visited[a] = false;  // skipped — vals[a] stays at 0
            continue;
        }
        visited_prob_sum += probs[a];
        vals[a] = traverse_model(NLHEGame::apply_action(state,legal[a]),
                                 tp, r_tp*probs[a], r_opp);
    }

    // Renormalise node value over the visited actions only. If all were
    // pruned (visited_prob_sum ≈ 0, shouldn't happen if argmax never gets
    // pruned, but be safe), fall back to expanding every action.
    float nv = 0.0f;
    if (visited_prob_sum > 1e-9f) {
        for(int a=0;a<n;++a) {
            if (visited[a]) nv += (probs[a] / visited_prob_sum) * vals[a];
        }
    } else {
        for(int a=0;a<n;++a) {
            vals[a] = traverse_model(NLHEGame::apply_action(state,legal[a]),
                                     tp, r_tp*probs[a], r_opp);
            visited[a] = true;
        }
        for(int a=0;a<n;++a) nv += probs[a]*vals[a];
    }

    float sv[NLHEStateEncoder::STATE_SIZE];
    NLHEStateEncoder::encode(state, p, sv);
    std::vector<float> instant(n, 0.0f);
    for(int a=0;a<n;++a) {
        if (visited[a]) instant[a] = r_opp * (vals[a] - nv);
        // Pruned actions emit 0 regret (no update) — matches Brown's treatment.
    }
    emit_or_accumulate(sv, legal, instant);
    return nv;
}

// ── Deterministic full traversal (parity testing) ─────────────────────────────

void NLHEMCCFREngine::run_full_traversal_deal_uniform(int tp, const NLHEDeal& deal) {
    NLHEStrategyFn fn=[](const std::string&,const std::vector<NLHEAction>&a){
        return std::vector<float>(a.size(),1.0f/a.size());};
    traverse_full(NLHEGame::initial_state(deal,config_.game_cfg), tp, 1.0f, fn);
}

float NLHEMCCFREngine::traverse_full(
    const NLHEState& state, int tp, float r_opp, const NLHEStrategyFn& fn)
{
    if(state.terminal) { float v = terminal_payoff_p0(state); return (tp==0)?v:-v; }
    const int p=state.current_player;
    const auto legal=NLHEGame::legal_actions(state);
    const int  n=(int)legal.size();
    const std::string iset=NLHEGame::info_set_key(state,p);
    auto probs=fn(iset,legal);

    std::vector<float> vals(n,0.0f);
    for(int a=0;a<n;++a){
        float opp_reach = (p==tp) ? r_opp : r_opp*probs[a];
        vals[a]=traverse_full(NLHEGame::apply_action(state,legal[a]),
                              tp, opp_reach, fn);
    }
    float nv=0.0f;
    for(int a=0;a<n;++a) nv+=probs[a]*vals[a];

    if(p==tp){
        float sv[NLHEStateEncoder::STATE_SIZE];
        NLHEStateEncoder::encode(state, p, sv);
        std::vector<float> instant(n);
        for(int a=0;a<n;++a) instant[a]=r_opp*(vals[a]-nv);
        emit_or_accumulate(sv, legal, instant);
    }
    return nv;
}

// ── Buffer export ─────────────────────────────────────────────────────────────

NLHEBufferExport NLHEMCCFREngine::export_regret_buffer() const {
    NLHEBufferExport e;
    const size_t n = regret_buf_.size();
    e.n_samples = n;
    e.states.reserve(n * NLHEStateEncoder::STATE_SIZE);
    e.actions.reserve(n); e.values.reserve(n); e.iterations.reserve(n);
    const auto* d = regret_buf_.data();
    for(size_t i=0;i<n;++i){
        e.states.insert(e.states.end(), d[i].state,
                        d[i].state + NLHEStateEncoder::STATE_SIZE);
        e.actions.push_back(d[i].action);
        e.values.push_back(d[i].regret);
        e.iterations.push_back(d[i].iteration);
    }
    return e;
}

NLHEBufferExport NLHEMCCFREngine::export_strategy_buffer() const {
    NLHEBufferExport e;
    const size_t n = strategy_buf_.size();
    e.n_samples = n;
    e.states.reserve(n * NLHEStateEncoder::STATE_SIZE);
    e.actions.reserve(n); e.values.reserve(n); e.iterations.reserve(n);
    const auto* d = strategy_buf_.data();
    for(size_t i=0;i<n;++i){
        e.states.insert(e.states.end(), d[i].state,
                        d[i].state + NLHEStateEncoder::STATE_SIZE);
        e.actions.push_back(d[i].action);
        e.values.push_back(d[i].probability);
        e.iterations.push_back(d[i].iteration);
    }
    return e;
}

// ── Strategy queries ──────────────────────────────────────────────────────────

std::vector<float> NLHEMCCFREngine::query_strategy(
    int hole0, int hole1, int street,
    const std::vector<int>& board,
    float pot, float to_call, float my_stack) const
{
    if (!strategy_model_.loaded())
        return std::vector<float>(4, 0.25f);
    NLHEState s{};
    s.cfg                 = config_.game_cfg;
    s.hole_cards[0][0]    = (int8_t)hole0;
    s.hole_cards[0][1]    = (int8_t)hole1;
    s.hole_cards[1][0]    = 0;
    s.hole_cards[1][1]    = 1;
    s.street              = (int8_t)std::min(street, 3);
    for (int i = 0; i < 5 && i < (int)board.size(); ++i)
        s.board[i]        = (int8_t)board[i];
    s.pot                 = pot;
    s.stacks[0]           = my_stack;
    s.stacks[1]           = s.cfg.starting_stack;
    s.street_invest[0]    = (to_call > 0) ? 0.0f : s.cfg.sb;
    s.street_invest[1]    = (to_call > 0) ? to_call : s.cfg.bb;
    s.current_player      = 0;
    s.action_count        = 0;
    std::vector<float> state_vec = NLHEStateEncoder::encode_vec(s, 0);
    return strategy_model_.forward(state_vec, 4);
}

std::vector<float> NLHEMCCFREngine::query_preflop_strategy(
    int hole0, int hole1) const
{
    const auto& c = config_.game_cfg;
    return query_strategy(
        hole0, hole1, 0, {},
        c.sb + c.bb, c.bb - c.sb, c.starting_stack - c.sb);
}

} // namespace cfr