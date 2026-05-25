#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <string>

namespace cfr {

// ── Constants ────────────────────────────────────────────────────────────────

// Ranks: J=0, Q=1, K=2. Cards: 0=J♠,1=J♥,2=Q♠,3=Q♥,4=K♠,5=K♥
static constexpr int NUM_CARDS        = 6;
static constexpr int NUM_RANKS        = 3;
static constexpr int NUM_PLAYERS      = 2;
static constexpr int MAX_RAISES       = 2;
static constexpr int MAX_ACTIONS      = 4;   // FOLD, CHECK, CALL, RAISE
static constexpr int MAX_HISTORY_LEN  = 8;

// Ante and bet sizes (standard Leduc)
static constexpr float ANTE           = 1.0f;
static constexpr float BET_ROUND1     = 2.0f;
static constexpr float BET_ROUND2     = 4.0f;

// ── Actions ──────────────────────────────────────────────────────────────────

enum Action : int8_t {
    FOLD  = 0,
    CHECK = 1,
    CALL  = 2,
    RAISE = 3
};

static constexpr const char* ACTION_NAMES[] = {"FOLD", "CHECK", "CALL", "RAISE"};

// ── Game State ───────────────────────────────────────────────────────────────

// Fully self-contained state — no pointers, copyable, hashable.
// Memory layout designed for SoA batching (CUDA-ready).
struct LeducState {
    int8_t  private_cards[NUM_PLAYERS]; // rank*2 + suit; -1 = undealt
    int8_t  community_card;             // rank*2 + suit; -1 = undealt
    int8_t  round;                      // 0 = pre-community, 1 = post-community
    int8_t  current_player;
    int8_t  raises_this_round;
    int8_t  last_bettor;                // -1 if no bet in current round
    float   pot;
    float   contributions[NUM_PLAYERS]; // total chips each player put in
    int8_t  action_history[MAX_HISTORY_LEN];
    int8_t  action_count;
    bool    folded[NUM_PLAYERS];
    bool    terminal;
    float   payoff_p0;                  // set when terminal (P0's perspective)
};

// ── Dealing ──────────────────────────────────────────────────────────────────

// One initial deal: private cards + community card pre-assigned.
// Probability = 1 / C(6,2) * 1/4 = 1/120 each unique deal.
struct LeducDeal {
    int8_t  private_cards[NUM_PLAYERS];
    int8_t  community_card;
    float   probability;
};

// ── Public API ───────────────────────────────────────────────────────────────

class LeducGame {
public:
    LeducGame() = default;

    // All possible initial deals (120 total: 30 hole-card combos × 4 community).
    static std::vector<LeducDeal> all_deals();

    // Create initial state from a deal.
    static LeducState initial_state(const LeducDeal& deal);

    // Legal actions in this state (2–3 actions typically).
    static std::vector<Action> legal_actions(const LeducState& state);

    // Apply action, returning new state (immutable — copies state).
    static LeducState apply_action(const LeducState& state, Action action);

    // Whether state is a chance node (community card about to be dealt).
    static bool is_chance_node(const LeducState& state) {
        return state.round == 1 && state.community_card == -1;
    }

    // Canonical info set key for player `p` (opponent card hidden).
    static std::string info_set_key(const LeducState& state, int player);

    // Hand strength at showdown (higher = better).
    static int hand_strength(int card, int community_card);

    // Bet size for current round.
    static float bet_size(const LeducState& state) {
        return state.round == 0 ? BET_ROUND1 : BET_ROUND2;
    }
};

} // namespace cfr