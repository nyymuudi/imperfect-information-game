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

bool NLHEMCCFREngine::load_model(const std::string& p) { return model_.load(p); }
bool NLHEMCCFREngine::load_strategy_model(const std::string& p) { return strategy_model_.load(p); }

std::vector<float> NLHEMCCFREngine::uniform_strategy(int n) {
    return std::vector<float>(n, 1.0f/n);
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
    for(int a=0;a<n;++a){
        NLHERegretSample rs{};
        NLHEStateEncoder::encode(state, p, rs.state);
        rs.action=static_cast<int8_t>(legal[a]);
        rs.regret=r_opp*(vals[a]-nv);
        rs.iteration=config_.iteration;
        regret_buf_.insert(rs);
    }
    return nv;
}

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
    for(int a=0;a<n;++a){
        NLHERegretSample rs{};
        NLHEStateEncoder::encode(state, p, rs.state);
        rs.action=static_cast<int8_t>(legal[a]);
        rs.regret=r_opp*(vals[a]-nv);
        rs.iteration=config_.iteration;
        regret_buf_.insert(rs);
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
// Uses OnnxStrategyModel (strategy_model_) — not TorchModel.
// encode_vec() produces float[122] directly; forward() calls ONNX Runtime.
// No CFR_TORCH_AVAILABLE guard needed here.

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

    // encode_vec() is always available (no LibTorch dependency)
    std::vector<float> state_vec = NLHEStateEncoder::encode_vec(s, 0);

    // OnnxStrategyModel::forward() — guarded by CFR_ORT_AVAILABLE internally
    return strategy_model_.forward(state_vec, 4);
}

std::vector<float> NLHEMCCFREngine::query_preflop_strategy(
    int hole0, int hole1) const
{
    const auto& c = config_.game_cfg;
    return query_strategy(
        hole0, hole1,
        /*street=*/0, /*board=*/{},
        c.sb + c.bb,
        c.bb - c.sb,
        c.starting_stack - c.sb
    );
}

} // namespace cfr