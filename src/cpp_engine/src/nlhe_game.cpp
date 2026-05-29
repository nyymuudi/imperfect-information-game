#include "nlhe_game.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace cfr {

// ── Hand Evaluator ─────────────────────────────────────────────────────────────
static int32_t encode_score(int category, const int* ranks, int n) {
    int32_t score = category << 24;
    for (int i = 0; i < n && i < 5; ++i)
        score |= (ranks[i] & 0xF) << (20 - i * 4);
    return score;
}

int32_t HandEvaluator::eval5(const int8_t* cards) {
    int ranks[5], suits[5];
    for (int i = 0; i < 5; ++i) { ranks[i]=card_rank(cards[i]); suits[i]=card_suit(cards[i]); }
    int sr[5]; std::copy(ranks,ranks+5,sr); std::sort(sr,sr+5,std::greater<int>());
    bool flush=(suits[0]==suits[1]&&suits[1]==suits[2]&&suits[2]==suits[3]&&suits[3]==suits[4]);
    bool straight=false; int sh=0;
    if(sr[0]-sr[4]==4&&sr[0]!=sr[1]&&sr[1]!=sr[2]&&sr[2]!=sr[3]&&sr[3]!=sr[4]){straight=true;sh=sr[0];}
    if(sr[0]==12&&sr[1]==3&&sr[2]==2&&sr[3]==1&&sr[4]==0){straight=true;sh=3;}
    if(flush&&straight) return encode_score(8,&sh,1);
    int cnt[13]={};
    for(int r:ranks) cnt[r]++;
    std::vector<std::pair<int,int>> g;
    for(int r=12;r>=0;--r) if(cnt[r]>0) g.push_back({cnt[r],r});
    std::sort(g.begin(),g.end(),[](auto&a,auto&b){return a.first!=b.first?a.first>b.first:a.second>b.second;});
    int c0=g[0].first;
    if(c0==4){int r[]={g[0].second,g[1].second};return encode_score(7,r,2);}
    if(c0==3&&g.size()>=2&&g[1].first>=2){int r[]={g[0].second,g[1].second};return encode_score(6,r,2);}
    if(flush) return encode_score(5,sr,5);
    if(straight) return encode_score(4,&sh,1);
    if(c0==3){int r[]={g[0].second,g[1].second,g[2].second};return encode_score(3,r,3);}
    if(c0==2&&g.size()>=2&&g[1].first==2){int r[]={g[0].second,g[1].second,g[2].second};return encode_score(2,r,3);}
    if(c0==2){int r[4]={g[0].second};int idx=1;for(size_t i=1;i<g.size()&&idx<4;++i)r[idx++]=g[i].second;return encode_score(1,r,4);}
    return encode_score(0,sr,5);
}

int32_t HandEvaluator::best_of_seven(const int8_t* cards) {
    static const int C[21][5]={{0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},{0,1,2,5,6},{0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},{0,2,3,4,5},{0,2,3,4,6},{0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},{1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},{1,2,4,5,6},{1,3,4,5,6},{2,3,4,5,6}};
    int32_t best=-1;
    for(auto&c:C){int8_t h[5]={cards[c[0]],cards[c[1]],cards[c[2]],cards[c[3]],cards[c[4]]};best=std::max(best,eval5(h));}
    return best;
}

int32_t HandEvaluator::evaluate(const int8_t* cards, int n) {
    if(n==5) return eval5(cards);
    if(n>=7) return best_of_seven(cards);
    int32_t best=-1;
    for(int skip=0;skip<6;++skip){int8_t h[5];int idx=0;for(int i=0;i<6;++i)if(i!=skip)h[idx++]=cards[i];best=std::max(best,eval5(h));}
    return best;
}

int HandEvaluator::compare_hands(const int8_t* h0,const int8_t* h1,const int8_t* board,int nb){
    int8_t a[7],b[7];a[0]=h0[0];a[1]=h0[1];b[0]=h1[0];b[1]=h1[1];
    for(int i=0;i<nb;++i){a[2+i]=board[i];b[2+i]=board[i];}
    int n=2+nb; int32_t s0=evaluate(a,n),s1=evaluate(b,n);
    return(s0>s1)?1:(s1>s0)?-1:0;
}

// ── Dealing ───────────────────────────────────────────────────────────────────
NLHEDeal NLHEGame::sample_deal(std::mt19937& rng) {
    int8_t deck[52]; std::iota(deck,deck+52,0);
    for(int i=51;i>43;--i){std::uniform_int_distribution<int>d(0,i);std::swap(deck[i],deck[d(rng)]);}
    NLHEDeal d;
    d.hole_cards[0][0]=deck[51]; d.hole_cards[0][1]=deck[50];
    d.hole_cards[1][0]=deck[49]; d.hole_cards[1][1]=deck[48];
    d.board[0]=deck[47];d.board[1]=deck[46];d.board[2]=deck[45];d.board[3]=deck[44];d.board[4]=deck[43];
    return d;
}

// ── Initial state (matches Python PostflopNLHE exactly) ───────────────────────
NLHEState NLHEGame::initial_state(const NLHEDeal& deal, const NLHEGameConfig& cfg) {
    NLHEState s{};
    s.hole_cards[0][0]=deal.hole_cards[0][0]; s.hole_cards[0][1]=deal.hole_cards[0][1];
    s.hole_cards[1][0]=deal.hole_cards[1][0]; s.hole_cards[1][1]=deal.hole_cards[1][1];
    for(int i=0;i<5;++i) s.board[i]=deal.board[i];
    s.cfg             = cfg;
    s.street          = 0;
    s.current_player  = 0;    // SB acts first preflop
    s.raises_this_street = 0;
    s.last_aggressor  = -1;
    // Blinds — matches Python: stacks=[stack-sb, stack-bb], pot=sb+bb
    s.stacks[0]       = cfg.starting_stack - cfg.sb;
    s.stacks[1]       = cfg.starting_stack - cfg.bb;
    s.pot             = cfg.sb + cfg.bb;
    s.street_invest[0]= cfg.sb;
    s.street_invest[1]= cfg.bb;
    s.action_count    = 0;
    s.folded[0]=s.folded[1]=false;
    s.terminal        = false;
    s.payoff_p0       = 0.0f;
    std::fill(s.action_history, s.action_history+NLHE_MAX_HISTORY, int8_t(-1));
    return s;
}

// ── Legal actions (mirrors Python PostflopNLHE.legal_actions) ─────────────────
//
// Python (condensed):
//   if to_call > 0:
//       result = ["f", "k"]
//       if raises_ok: result.append("r")
//       if stack > 0: result.append("a")   [conditional on raise being different]
//   else:
//       result = ["c"]
//       if raises_ok: result.append("r")
//       if stack > 0: result.append("a")
std::vector<NLHEAction> NLHEGame::legal_actions(const NLHEState& s) {
    if(s.terminal) return {};
    std::vector<NLHEAction> a;
    const int   p   = s.current_player;
    const float owe = s.street_invest[1-p] - s.street_invest[p];
    const bool  bet = (owe > 0.001f);
    const bool  can_raise = (s.raises_this_street < s.cfg.max_raises);
    const float my_stack  = s.stacks[p];

    if(bet) {
        a.push_back(NLHE_FOLD_OR_CHECK);  // fold
        a.push_back(NLHE_CALL);
    } else {
        a.push_back(NLHE_FOLD_OR_CHECK);  // check
    }

    if(can_raise && my_stack > owe + 0.01f) {
        float raise_add = bet_amount(s);
        if(raise_add > 0.01f)
            a.push_back(NLHE_RAISE);
    }

    // All-in: available if has chips, and is different from raise
    if(my_stack > owe + 0.01f) {
        float allin_add = my_stack - owe;
        float raise_add = (can_raise && my_stack > owe + 0.01f) ? bet_amount(s) : -1.0f;
        // Add ALL_IN if raise option exists and all-in is larger, or no raise option
        if(!can_raise || allin_add > raise_add + 0.01f)
            a.push_back(NLHE_ALL_IN);
    }
    return a;
}

// ── Bet sizing: 75% pot (matching Python raise_fractions[0]=0.75) ─────────────
float NLHEGame::bet_amount(const NLHEState& s) {
    const int   p   = s.current_player;
    const float owe = s.street_invest[1-p] - s.street_invest[p];
    const float effective_pot = s.pot + owe;   // pot after calling
    float raise_add = effective_pot * s.cfg.raise_fraction;
    return std::min(raise_add, s.stacks[p] - owe);
}

// ── Street transition ─────────────────────────────────────────────────────────
static void advance_street(NLHEState& s) {
    s.street++;
    s.raises_this_street = 0;
    s.last_aggressor     = -1;
    s.street_invest[0]   = s.street_invest[1] = 0.0f;

    if(s.street >= 4) {
        // Showdown
        int r = HandEvaluator::compare_hands(
            s.hole_cards[0], s.hole_cards[1], s.board, 5);
        float inv0 = s.cfg.starting_stack - s.stacks[0];
        float inv1 = s.cfg.starting_stack - s.stacks[1];
        s.terminal = true;
        if(r==1)       s.payoff_p0 =  inv1;
        else if(r==-1) s.payoff_p0 = -inv0;
        else           s.payoff_p0 =  0.0f;
    } else {
        // Postflop: BB (player 1) acts first out of position
        s.current_player = 1;
    }
}

// ── Apply action ──────────────────────────────────────────────────────────────
NLHEState NLHEGame::apply_action(const NLHEState& state, NLHEAction action) {
    NLHEState s = state;
    if(s.action_count < NLHE_MAX_HISTORY)
        s.action_history[s.action_count++] = static_cast<int8_t>(action);

    const int   p   = s.current_player;
    const int   opp = 1-p;
    const float owe = s.street_invest[opp] - s.street_invest[p];

    switch(action) {
    case NLHE_FOLD_OR_CHECK:
        if(owe > 0.001f) {
            // Fold
            s.folded[p] = true;
            s.terminal  = true;
            s.payoff_p0 = (p==0)
                ? -(s.cfg.starting_stack - s.stacks[0])
                :  (s.cfg.starting_stack - s.stacks[1]);
        } else {
            // Check
            if(s.last_aggressor == -1) {
                s.last_aggressor = -2;   // first check
                s.current_player = opp;
            } else {
                advance_street(s);       // both checked
            }
        }
        break;

    case NLHE_CALL: {
        float call_amt = std::min(owe, s.stacks[p]);
        s.stacks[p]        -= call_amt;
        s.street_invest[p] += call_amt;
        s.pot              += call_amt;
        advance_street(s);
        break;
    }

    case NLHE_RAISE: {
        float raise_add = bet_amount(s);
        float total = owe + raise_add;
        total = std::min(total, s.stacks[p]);
        s.stacks[p]        -= total;
        s.street_invest[p] += total;
        s.pot              += total;
        s.raises_this_street++;
        s.last_aggressor   = static_cast<int8_t>(p);
        s.current_player   = static_cast<int8_t>(opp);
        break;
    }

    case NLHE_ALL_IN: {
        float allin = s.stacks[p];
        s.stacks[p]        = 0.0f;
        s.street_invest[p] += allin;
        s.pot              += allin;
        s.raises_this_street++;
        s.last_aggressor   = static_cast<int8_t>(p);
        s.current_player   = static_cast<int8_t>(opp);
        break;
    }

    default: break;
    }
    return s;
}

// ── Info set key ──────────────────────────────────────────────────────────────
std::string NLHEGame::info_set_key(const NLHEState& s, int player) {
    std::ostringstream oss; oss << std::hex;
    int c0=s.hole_cards[player][0], c1=s.hole_cards[player][1];
    if(c0>c1) std::swap(c0,c1);
    oss << 'H' << std::setw(2)<<std::setfill('0')<<c0
               << std::setw(2)<<std::setfill('0')<<c1;
    oss << std::dec << "|S" << (int)s.street;
    oss << "|B";
    int nv=visible_board_cards(s.street);
    for(int i=0;i<nv;++i)
        oss<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)s.board[i];
    // Scale bucket boundaries with starting_stack so all stack sizes
    // (10BB, 50BB, 200BB) produce meaningful coverage (not all bucket 0).
    // With starting_stack=200: bucket_width=50 → identical to before.
    float bucket_width = s.cfg.starting_stack / 4.0f;
    int pot_bucket = std::min(7, (int)(s.pot / bucket_width));
    oss<<std::dec<<"|P"<<pot_bucket<<"|A";
    for(int i=0;i<s.action_count;++i) oss<<(int)s.action_history[i];
    return oss.str();
}

} // namespace cfr