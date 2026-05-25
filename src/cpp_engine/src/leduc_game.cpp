#include "leduc_game.hpp"
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace cfr {

// ── Dealing ───────────────────────────────────────────────────────────────────

std::vector<LeducDeal> LeducGame::all_deals() {
    std::vector<LeducDeal> deals;
    deals.reserve(120);

    // Choose 3 distinct cards: p0_card, p1_card, community_card
    // Total arrangements of 3 from 6 cards (ordered, distinct): 6*5*4 = 120
    for (int p0 = 0; p0 < NUM_CARDS; ++p0) {
        for (int p1 = 0; p1 < NUM_CARDS; ++p1) {
            if (p1 == p0) continue;
            for (int comm = 0; comm < NUM_CARDS; ++comm) {
                if (comm == p0 || comm == p1) continue;
                LeducDeal d;
                d.private_cards[0] = static_cast<int8_t>(p0);
                d.private_cards[1] = static_cast<int8_t>(p1);
                d.community_card   = static_cast<int8_t>(comm);
                d.probability      = 1.0f / 120.0f;
                deals.push_back(d);
            }
        }
    }
    return deals;
}

LeducState LeducGame::initial_state(const LeducDeal& deal) {
    LeducState s{};
    s.private_cards[0]  = deal.private_cards[0];
    s.private_cards[1]  = deal.private_cards[1];
    s.community_card    = -1;           // hidden until round 2
    s.round             = 0;
    s.current_player    = 0;
    s.raises_this_round = 0;
    s.last_bettor       = -1;
    s.pot               = ANTE * NUM_PLAYERS;
    s.contributions[0]  = ANTE;
    s.contributions[1]  = ANTE;
    s.action_count      = 0;
    s.folded[0]         = false;
    s.folded[1]         = false;
    s.terminal          = false;
    s.payoff_p0         = 0.0f;
    std::fill(s.action_history, s.action_history + MAX_HISTORY_LEN, int8_t(-1));
    // Store community card in state (hidden from players until round 2)
    // We encode it as a "future" card — only exposed via info_set_key when round > 0
    // Hack: use a separate field for the known community card
    // Actually, let's store it always; info_set_key simply won't include it in round 0.
    s.community_card = deal.community_card;  // stored, but hidden in infoset until r=1
    return s;
}

// ── Legal Actions ─────────────────────────────────────────────────────────────

std::vector<Action> LeducGame::legal_actions(const LeducState& state) {
    if (state.terminal) return {};

    std::vector<Action> actions;
    actions.reserve(3);

    const int    p    = state.current_player;
    const float  owe  = state.contributions[1 - p] - state.contributions[p];
    const bool   bet_outstanding = (owe > 0.0f);
    const bool   can_raise = (state.raises_this_round < MAX_RAISES);

    if (bet_outstanding) {
        actions.push_back(FOLD);
        actions.push_back(CALL);
        if (can_raise) actions.push_back(RAISE);
    } else {
        actions.push_back(CHECK);
        if (can_raise) actions.push_back(RAISE);
    }
    return actions;
}

// ── Hand Evaluation ───────────────────────────────────────────────────────────

// Returns strength value: pair > high card, pair rank > pair rank.
// Range: [0, 5]. Pair of Ks (rank 2) = 5, pair of Js (rank 0) = 3,
// high K (rank 2, no pair) = 2, high J = 0.
int LeducGame::hand_strength(int card, int community_card) {
    int private_rank  = card / 2;
    int community_rank = community_card / 2;
    if (private_rank == community_rank) {
        return NUM_RANKS + private_rank;   // pair: 3..5
    }
    return private_rank;                    // high card: 0..2
}

// ── State Transition ──────────────────────────────────────────────────────────

static void advance_to_round2_or_terminal(LeducState& s) {
    if (s.round == 0) {
        // Move to round 2 (community card now visible)
        s.round             = 1;
        s.raises_this_round = 0;
        s.last_bettor       = -1;
        s.current_player    = 0;   // P0 acts first in round 2
    } else {
        // Showdown
        int s0 = LeducGame::hand_strength(s.private_cards[0], s.community_card);
        int s1 = LeducGame::hand_strength(s.private_cards[1], s.community_card);
        s.terminal = true;
        if (s0 > s1)       s.payoff_p0 =  s.contributions[1];
        else if (s1 > s0)  s.payoff_p0 = -s.contributions[0];
        else               s.payoff_p0 =  0.0f;   // split pot
    }
}

LeducState LeducGame::apply_action(const LeducState& state, Action action) {
    LeducState s = state;   // copy

    // Record action history
    if (s.action_count < MAX_HISTORY_LEN) {
        s.action_history[s.action_count++] = static_cast<int8_t>(action);
    }

    const int   p     = s.current_player;
    const int   opp   = 1 - p;
    const float bsize = bet_size(s);
    const float owe   = s.contributions[opp] - s.contributions[p];

    switch (action) {
    case FOLD:
        s.folded[p] = true;
        s.terminal  = true;
        // Folder loses their contribution; opponent wins what's already in pot
        s.payoff_p0 = (p == 0) ? -s.contributions[0] : s.contributions[1];
        break;

    case CHECK:
        if (s.last_bettor == -1) {
            // First check: pass to opponent
            s.current_player = opp;
            s.last_bettor    = -2;   // -2 = both have checked once
        } else {
            // Second check: round ends
            advance_to_round2_or_terminal(s);
        }
        break;

    case CALL:
        s.contributions[p] += owe;
        s.pot              += owe;
        // After call, round ends
        advance_to_round2_or_terminal(s);
        break;

    case RAISE:
        s.contributions[p] += owe + bsize;
        s.pot              += owe + bsize;
        s.raises_this_round++;
        s.last_bettor    = static_cast<int8_t>(p);
        s.current_player = static_cast<int8_t>(opp);
        break;
    }

    return s;
}

// ── Info Set Key ──────────────────────────────────────────────────────────────

std::string LeducGame::info_set_key(const LeducState& state, int player) {
    std::ostringstream oss;

    // Private card rank only (suits abstracted away — J♠ = J♥ strategically).
    // This matches the Python LeducEncoder one-hot rank representation.
    int private_rank = state.private_cards[player] / 2;
    oss << 'r' << private_rank;

    // Community card rank (only visible in round 1+)
    if (state.round > 0) {
        int community_rank = state.community_card / 2;
        oss << 'b' << community_rank;
    }

    // Action history (fully public)
    for (int i = 0; i < state.action_count; ++i) {
        oss << 'a' << static_cast<int>(state.action_history[i]);
    }

    return oss.str();
}

} // namespace cfr