#include "nlhe_game.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace cfr {

// ── Hand Evaluator ─────────────────────────────────────────────────────────────
//
// Score encoding (higher = better):
//   bits 24-26: hand category (0-8)
//   bits 0-23:  tiebreaker ranks (up to 5 ranks, 4 bits each)
//
// Categories:
//   8 = straight flush    7 = four of a kind   6 = full house
//   5 = flush             4 = straight         3 = three of a kind
//   2 = two pair          1 = one pair         0 = high card

static int32_t encode_score(int category, const int* ranks, int n) {
    int32_t score = category << 24;
    for (int i = 0; i < n && i < 5; ++i) {
        score |= (ranks[i] & 0xF) << (20 - i * 4);
    }
    return score;
}

int32_t HandEvaluator::eval5(const int8_t* cards) {
    int ranks[5], suits[5];
    for (int i = 0; i < 5; ++i) {
        ranks[i] = card_rank(cards[i]);
        suits[i] = card_suit(cards[i]);
    }

    // Sort ranks descending
    int sr[5];
    std::copy(ranks, ranks + 5, sr);
    std::sort(sr, sr + 5, std::greater<int>());

    // Check flush
    bool flush = (suits[0] == suits[1] && suits[1] == suits[2] &&
                  suits[2] == suits[3] && suits[3] == suits[4]);

    // Check straight (including A-5 wheel)
    bool straight = false;
    int  straight_high = 0;
    if (sr[0] - sr[4] == 4 && sr[0] != sr[1] && sr[1] != sr[2] &&
        sr[2] != sr[3] && sr[3] != sr[4]) {
        straight = true;
        straight_high = sr[0];
    }
    // A-2-3-4-5 wheel: ranks sorted are [12,3,2,1,0]
    if (sr[0] == 12 && sr[1] == 3 && sr[2] == 2 && sr[3] == 1 && sr[4] == 0) {
        straight = true;
        straight_high = 3;  // 5-high straight
    }

    if (flush && straight) {
        return encode_score(8, &straight_high, 1);
    }

    // Count rank occurrences
    int counts[13] = {};
    for (int r : ranks) counts[r]++;

    // Group by count
    std::vector<std::pair<int,int>> groups;  // (count, rank)
    for (int r = 12; r >= 0; --r) {
        if (counts[r] > 0) groups.push_back({counts[r], r});
    }
    // Sort: higher count first, then higher rank
    std::sort(groups.begin(), groups.end(), [](auto& a, auto& b){
        return a.first != b.first ? a.first > b.first : a.second > b.second;
    });

    int c0 = groups[0].first;

    if (c0 == 4) {
        // Four of a kind
        int r[2] = {groups[0].second, groups[1].second};
        return encode_score(7, r, 2);
    }
    if (c0 == 3 && groups.size() >= 2 && groups[1].first >= 2) {
        // Full house
        int r[2] = {groups[0].second, groups[1].second};
        return encode_score(6, r, 2);
    }
    if (flush) {
        return encode_score(5, sr, 5);
    }
    if (straight) {
        return encode_score(4, &straight_high, 1);
    }
    if (c0 == 3) {
        // Three of a kind
        int r[3] = {groups[0].second, groups[1].second, groups[2].second};
        return encode_score(3, r, 3);
    }
    if (c0 == 2 && groups.size() >= 2 && groups[1].first == 2) {
        // Two pair
        int r[3] = {groups[0].second, groups[1].second, groups[2].second};
        return encode_score(2, r, 3);
    }
    if (c0 == 2) {
        // One pair
        int r[4] = {groups[0].second};
        int idx = 1;
        for (size_t i = 1; i < groups.size() && idx < 4; ++i)
            r[idx++] = groups[i].second;
        return encode_score(1, r, 4);
    }
    // High card
    return encode_score(0, sr, 5);
}

int32_t HandEvaluator::best_of_seven(const int8_t* cards) {
    // C(7,5) = 21 combinations
    static const int combos[21][5] = {
        {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},
        {0,1,2,5,6},{0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},
        {0,2,3,4,5},{0,2,3,4,6},{0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},
        {1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},{1,2,4,5,6},{1,3,4,5,6},
        {2,3,4,5,6}
    };
    int32_t best = -1;
    for (auto& c : combos) {
        int8_t hand[5] = {cards[c[0]], cards[c[1]], cards[c[2]],
                          cards[c[3]], cards[c[4]]};
        best = std::max(best, eval5(hand));
    }
    return best;
}

int32_t HandEvaluator::evaluate(const int8_t* cards, int n_cards) {
    if (n_cards == 5) return eval5(cards);
    if (n_cards >= 7) return best_of_seven(cards);
    // 6 cards: best of C(6,5)=6
    int32_t best = -1;
    for (int skip = 0; skip < 6; ++skip) {
        int8_t hand[5]; int idx = 0;
        for (int i = 0; i < 6; ++i) if (i != skip) hand[idx++] = cards[i];
        best = std::max(best, eval5(hand));
    }
    return best;
}

int HandEvaluator::compare_hands(
    const int8_t* hole0, const int8_t* hole1,
    const int8_t* board,  int n_board)
{
    int8_t hand0[7], hand1[7];
    hand0[0] = hole0[0]; hand0[1] = hole0[1];
    hand1[0] = hole1[0]; hand1[1] = hole1[1];
    for (int i = 0; i < n_board; ++i) {
        hand0[2 + i] = board[i];
        hand1[2 + i] = board[i];
    }
    int n = 2 + n_board;
    int32_t s0 = evaluate(hand0, n);
    int32_t s1 = evaluate(hand1, n);
    return (s0 > s1) ? 1 : (s1 > s0) ? -1 : 0;
}

// ── Dealing ───────────────────────────────────────────────────────────────────

NLHEDeal NLHEGame::sample_deal(std::mt19937& rng) {
    // Fisher-Yates on a 52-card deck, take first 9 cards
    int8_t deck[52];
    std::iota(deck, deck + 52, 0);
    for (int i = 51; i > 43; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(deck[i], deck[dist(rng)]);
    }
    NLHEDeal d;
    d.hole_cards[0][0] = deck[51];
    d.hole_cards[0][1] = deck[50];
    d.hole_cards[1][0] = deck[49];
    d.hole_cards[1][1] = deck[48];
    d.board[0] = deck[47]; d.board[1] = deck[46]; d.board[2] = deck[45];
    d.board[3] = deck[44]; d.board[4] = deck[43];
    return d;
}

// ── Initial State ─────────────────────────────────────────────────────────────

NLHEState NLHEGame::initial_state(const NLHEDeal& deal) {
    NLHEState s{};
    s.hole_cards[0][0] = deal.hole_cards[0][0];
    s.hole_cards[0][1] = deal.hole_cards[0][1];
    s.hole_cards[1][0] = deal.hole_cards[1][0];
    s.hole_cards[1][1] = deal.hole_cards[1][1];
    for (int i = 0; i < 5; ++i) s.board[i] = deal.board[i];

    s.street          = 0;        // preflop
    s.current_player  = 0;        // SB/button acts first preflop
    s.raises_this_street = 0;
    s.last_aggressor  = -1;

    // Blinds
    s.stacks[0]       = NLHE_STACK - NLHE_SB;
    s.stacks[1]       = NLHE_STACK - NLHE_BB;
    s.pot             = NLHE_SB + NLHE_BB;
    s.street_invest[0]= NLHE_SB;
    s.street_invest[1]= NLHE_BB;

    s.action_count    = 0;
    s.folded[0]       = false;
    s.folded[1]       = false;
    s.terminal        = false;
    s.payoff_p0       = 0.0f;
    std::fill(s.action_history, s.action_history + NLHE_MAX_HISTORY, int8_t(-1));
    return s;
}

// ── Legal Actions ─────────────────────────────────────────────────────────────

std::vector<NLHEAction> NLHEGame::legal_actions(const NLHEState& state) {
    if (state.terminal) return {};

    std::vector<NLHEAction> actions;
    const int   p   = state.current_player;
    const float owe = state.street_invest[1-p] - state.street_invest[p];
    const bool  bet_out = (owe > 0.0f);
    const bool  can_raise = (state.raises_this_street < NLHE_MAX_RAISES);
    const float my_stack = state.stacks[p];

    if (bet_out) {
        actions.push_back(NLHE_FOLD);
        actions.push_back(NLHE_CALL);
    } else {
        actions.push_back(NLHE_CHECK);
    }

    if (can_raise && my_stack > owe + 0.01f) {
        float half_pot = (state.pot + owe) * 0.5f;
        float full_pot =  state.pot + owe;

        if (half_pot < my_stack - owe) actions.push_back(NLHE_BET_HALF);
        if (full_pot < my_stack - owe) actions.push_back(NLHE_BET_POT);
        actions.push_back(NLHE_ALL_IN);
    }
    return actions;
}

// ── Bet Sizing ─────────────────────────────────────────────────────────────────

float NLHEGame::bet_amount(const NLHEState& state, NLHEAction action) {
    const int   p   = state.current_player;
    const float owe = state.street_invest[1-p] - state.street_invest[p];
    const float pot = state.pot + owe;   // pot after calling

    switch (action) {
    case NLHE_BET_HALF:  return std::min(pot * 0.5f,  state.stacks[p] - owe);
    case NLHE_BET_POT:   return std::min(pot * 1.0f,  state.stacks[p] - owe);
    case NLHE_ALL_IN:    return state.stacks[p] - owe;  // go all-in
    default: return 0.0f;
    }
}

// ── Street Transition ─────────────────────────────────────────────────────────

static void advance_street(NLHEState& s) {
    s.street++;
    s.raises_this_street = 0;
    s.last_aggressor     = -1;
    s.street_invest[0]   = 0.0f;
    s.street_invest[1]   = 0.0f;

    if (s.street >= NLHE_NUM_STREETS) {
        // Showdown
        int n_board = 5;
        int result = HandEvaluator::compare_hands(
            s.hole_cards[0], s.hole_cards[1], s.board, n_board);

        float invested_p0 = NLHE_STACK - s.stacks[0];
        float invested_p1 = NLHE_STACK - s.stacks[1];

        s.terminal = true;
        if (result == 1)       s.payoff_p0 =  invested_p1;
        else if (result == -1) s.payoff_p0 = -invested_p0;
        else                   s.payoff_p0 =  0.0f;
    } else {
        // Postflop: BB (player 1) acts first out of position
        s.current_player = 1;
    }
}

// ── State Transition ─────────────────────────────────────────────────────────

NLHEState NLHEGame::apply_action(const NLHEState& state, NLHEAction action) {
    NLHEState s = state;
    if (s.action_count < NLHE_MAX_HISTORY) {
        s.action_history[s.action_count++] = static_cast<int8_t>(action);
    }

    const int   p   = s.current_player;
    const int   opp = 1 - p;
    const float owe = s.street_invest[opp] - s.street_invest[p];

    switch (action) {
    case NLHE_FOLD:
        s.folded[p] = true;
        s.terminal  = true;
        // Folder loses all invested chips; opponent wins
        s.payoff_p0 = (p == 0)
            ? -(NLHE_STACK - s.stacks[0])
            :  (NLHE_STACK - s.stacks[1]);
        break;

    case NLHE_CHECK:
        if (s.last_aggressor == -1) {
            // First check: opponent acts
            s.last_aggressor = -2;   // sentinel: one check done
            s.current_player = opp;
        } else {
            // Second check: street over
            advance_street(s);
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

    case NLHE_BET_HALF:
    case NLHE_BET_POT:
    case NLHE_ALL_IN: {
        float raise_add = bet_amount(s, action);
        float total_put = owe + raise_add;
        total_put = std::min(total_put, s.stacks[p]);

        s.stacks[p]        -= total_put;
        s.street_invest[p] += total_put;
        s.pot              += total_put;
        s.raises_this_street++;
        s.last_aggressor   = static_cast<int8_t>(p);
        s.current_player   = static_cast<int8_t>(opp);
        break;
    }

    default: break;
    }
    return s;
}

// ── Info Set Key ───────────────────────────────────────────────────────────────
// Format: H{c1}{c2}|S{s}|B{visible_board}|P{pot_bucket}|A{actions}
// Cards encoded as 2-digit hex. Pot bucketed into 8 bins.

std::string NLHEGame::info_set_key(const NLHEState& state, int player) {
    std::ostringstream oss;
    oss << std::hex;

    // Hole cards (sorted so Ah2c == 2cAh)
    int c0 = state.hole_cards[player][0];
    int c1 = state.hole_cards[player][1];
    if (c0 > c1) std::swap(c0, c1);
    oss << 'H' << std::setw(2) << std::setfill('0') << c0
               << std::setw(2) << std::setfill('0') << c1;

    // Street
    oss << std::dec << '|' << 'S' << static_cast<int>(state.street);

    // Visible board cards
    oss << '|' << 'B';
    int n_visible = visible_board_cards(state.street);
    for (int i = 0; i < n_visible; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(state.board[i]);
    }

    // Pot bucket (8 buckets: 0-12.5, 12.5-25, ... , 87.5-100+)
    int pot_bucket = std::min(7, static_cast<int>(state.pot / 25.0f));
    oss << std::dec << '|' << 'P' << pot_bucket;

    // Action history
    oss << '|' << 'A';
    for (int i = 0; i < state.action_count; ++i) {
        oss << static_cast<int>(state.action_history[i]);
    }

    return oss.str();
}

} // namespace cfr