#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include <random>

namespace cfr {

// ── Constants ────────────────────────────────────────────────────────────────

static constexpr int NLHE_DECK_SIZE    = 52;
static constexpr int NLHE_NUM_PLAYERS  = 2;
static constexpr int NLHE_NUM_STREETS  = 4;   // preflop, flop, turn, river
static constexpr int NLHE_MAX_RAISES   = 4;
static constexpr int NLHE_MAX_HISTORY  = 32;
static constexpr float NLHE_STACK      = 100.0f;  // BB units
static constexpr float NLHE_SB         = 0.5f;
static constexpr float NLHE_BB         = 1.0f;

// Board cards per street transition
// After preflop: deal 3 (flop), 1 (turn), 1 (river)
static constexpr int BOARD_CARDS_BY_STREET[4] = {0, 3, 4, 5};

// ── NLHE Actions ─────────────────────────────────────────────────────────────

enum NLHEAction : int8_t {
    NLHE_FOLD      = 0,
    NLHE_CHECK     = 1,
    NLHE_CALL      = 2,
    NLHE_BET_HALF  = 3,   // 50% pot
    NLHE_BET_POT   = 4,   // 100% pot
    NLHE_ALL_IN    = 5,
    NLHE_NUM_ACTIONS = 6
};

static constexpr const char* NLHE_ACTION_NAMES[] = {
    "FOLD", "CHECK", "CALL", "BET_HALF", "BET_POT", "ALL_IN"
};

// ── Card encoding ─────────────────────────────────────────────────────────────
// card = rank * 4 + suit
// rank: 0=2, 1=3, ..., 12=A
// suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades

inline int card_rank(int card) { return card / 4; }
inline int card_suit(int card) { return card % 4; }

// ── NLHE Game State ──────────────────────────────────────────────────────────

struct NLHEState {
    // Cards
    int8_t  hole_cards[NLHE_NUM_PLAYERS][2];  // [player][card_idx]
    int8_t  board[5];                          // all 5 community cards pre-dealt
    int8_t  street;                            // 0-3

    // Betting
    int8_t  current_player;
    int8_t  raises_this_street;
    int8_t  last_aggressor;        // -1 = no bet this street
    float   pot;                   // total pot
    float   stacks[2];             // remaining chips
    float   street_invest[2];      // chips put in THIS street (for call sizing)

    // History (action indices, one per action taken)
    int8_t  action_history[NLHE_MAX_HISTORY];
    int8_t  action_count;

    // Terminal
    bool    folded[2];
    bool    terminal;
    float   payoff_p0;             // P0's NET gain (signed)
};

// ── Deal ─────────────────────────────────────────────────────────────────────

struct NLHEDeal {
    int8_t hole_cards[NLHE_NUM_PLAYERS][2];
    int8_t board[5];
};

// ── Hand Evaluator ────────────────────────────────────────────────────────────
// Evaluates best 5-card hand from up to 7 cards.
// Higher return value = stronger hand.

class HandEvaluator {
public:
    // Evaluate best 5-card hand from cards array (length 5-7).
    // Returns a score: higher is better.
    static int32_t evaluate(const int8_t* cards, int n_cards);

    // Compare two 7-card hands (2 hole + 5 board).
    // Returns: 1 if hand0 wins, -1 if hand1 wins, 0 if tie.
    static int compare_hands(
        const int8_t* hole0, const int8_t* hole1,
        const int8_t* board, int n_board);

private:
    static int32_t eval5(const int8_t* cards);  // exactly 5 cards
    static int32_t best_of_seven(const int8_t* cards);
};

// ── NLHEGame ─────────────────────────────────────────────────────────────────

class NLHEGame {
public:
    // Sample a random deal (uniform over all card orderings).
    static NLHEDeal sample_deal(std::mt19937& rng);

    // Create initial state from deal (antes/blinds posted).
    static NLHEState initial_state(const NLHEDeal& deal);

    // Legal actions in state.
    static std::vector<NLHEAction> legal_actions(const NLHEState& state);

    // Apply action, returning new state.
    static NLHEState apply_action(const NLHEState& state, NLHEAction action);

    // Info set key for player p (hides opponent cards).
    // Format: H{c1}{c2}|S{street}|B{board...}|P{pot_bucket}|A{actions...}
    static std::string info_set_key(const NLHEState& state, int player);

    // Bet size for a given action in state.
    static float bet_amount(const NLHEState& state, NLHEAction action);

    // Visible board cards count for current street.
    static int visible_board_cards(int street) {
        return BOARD_CARDS_BY_STREET[street];
    }
};

// ── Compact state for Python export ──────────────────────────────────────────
// Used to build tensors on Python side without string parsing.

struct NLHEStateExport {
    int8_t  hole_cards[2];       // player's own cards (opponent hidden)
    int8_t  board[5];            // community cards (-1 = not yet visible)
    int8_t  street;
    float   pot_fraction;        // pot / (2 * starting_stack)
    float   to_call_fraction;    // to_call / (2 * starting_stack)
    float   my_stack_fraction;   // stack / starting_stack
    int8_t  action_history[NLHE_MAX_HISTORY];
    int8_t  action_count;
};

} // namespace cfr