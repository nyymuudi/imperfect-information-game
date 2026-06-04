#include "nlhe_mccfr.hpp"
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cfr {

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
        int slot = static_cast<int>(a);
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
        // R(I) <- max(R(I) + r^t, 0). Index by ACTION ENUM SLOT (legal[i]) so
        // the same action always lands in the same R slot regardless of which
        // subset of actions is legal at this node (matches network output slot).
        int slot = static_cast<int>(legal[i]);
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
            rs.action    = static_cast<int8_t>(legal[a]);
            rs.regret    = instant_regret[a];
            rs.iteration = config_.iteration;
            regret_buf_.insert(rs);
        }
    } else {
        accumulate_cfrplus(state_vec, legal, instant_regret);
    }
}

void NLHEMCCFREngine::emit_cfrplus_targets() {
    // Snapshot-semantiikka: tyhjennä reservoir, jotta tämä emit
    // korvaa edellisen iteraation R^T-snapshotin eikä monista sitä.
    regret_buf_.clear();
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
    if(state.terminal) return (tp==0)?state.payoff_p0:-state.payoff_p0;
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
                ss.action=static_cast<int8_t>(legal[a]);
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
    if(state.terminal) return (tp==0)?state.payoff_p0:-state.payoff_p0;
    const int p=state.current_player;
    const auto legal=NLHEGame::legal_actions(state);
    const int  n=(int)legal.size();
    auto probs=model_strategy(state,p,legal);

    if(p!=tp){
        if(config_.collect_strategy){
            for(int a=0;a<n;++a){
                NLHEStrategySample ss{};
                NLHEStateEncoder::encode(state, p, ss.state);
                ss.action=static_cast<int8_t>(legal[a]);
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

    std::vector<float>vals(n);
    for(int a=0;a<n;++a)
        vals[a]=traverse_model(NLHEGame::apply_action(state,legal[a]),
                               tp,r_tp*probs[a],r_opp);
    float nv=0.0f;
    for(int a=0;a<n;++a) nv+=probs[a]*vals[a];

    float sv[NLHEStateEncoder::STATE_SIZE];
    NLHEStateEncoder::encode(state, p, sv);
    std::vector<float> instant(n);
    for(int a=0;a<n;++a) instant[a]=r_opp*(vals[a]-nv);
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
    if(state.terminal) return (tp==0)?state.payoff_p0:-state.payoff_p0;
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