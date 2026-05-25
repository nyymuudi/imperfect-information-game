#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include <random>

namespace cfr {

// ── Card encoding ─────────────────────────────────────────────────────────────
// card = rank * 4 + suit
// rank: 0=2, 1=3, ..., 12=A   suit: 0=♣,1=♦,2=♥,3=♠
inline int card_rank(int card) { return card / 4; }
inline int card_suit(int card) { return card % 4; }

// ── Game configuration ────────────────────────────────────────────────────────
// Matches PostflopNLHE Python defaults exactly.
struct NLHEGameConfig {
    float starting_stack = 200.0f;  // chips (BB=2, SB=1)
    float sb             = 1.0f;
    float bb             = 2.0f;
    float raise_fraction = 0.75f;   // fraction of pot for RAISE action
    int   max_raises     = 2;       // per street (matches max_raises_per_street=2)
};

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int NLHE_DECK_SIZE   = 52;
static constexpr int NLHE_NUM_PLAYERS = 2;
static constexpr int NLHE_MAX_HISTORY = 32;

// Default values (matching Python PostflopNLHE)
static constexpr float NLHE_STACK = 200.0f;
static constexpr float NLHE_SB    = 1.0f;
static constexpr float NLHE_BB    = 2.0f;

// Board cards visible after each street (index = street 0-3)
static constexpr int BOARD_CARDS_BY_STREET[4] = {0, 3, 4, 5};

// ── Actions ───────────────────────────────────────────────────────────────────
// 4 actions matching Python PostflopNLHE exactly:
//   No bet context:   [FOLD_OR_CHECK(check), RAISE*, ALL_IN*]
//   Facing bet:       [FOLD_OR_CHECK(fold),  CALL, RAISE*, ALL_IN*]
//
// Network output slots (same as Python NLHEEncoder):
//   slot 0 = "f" when facing bet, "c" when no bet
//   slot 1 = "k" (call) — only legal when facing bet
//   slot 2 = "r" (raise 75% pot)
//   slot 3 = "a" (all-in)
enum NLHEAction : int8_t {
    NLHE_FOLD_OR_CHECK = 0,
    NLHE_CALL          = 1,
    NLHE_RAISE         = 2,
    NLHE_ALL_IN        = 3,
    NLHE_NUM_ACTIONS   = 4
};

// Action history encoding — matches Python ACTION_ENC values
// {"f":-1.0, "c":0.0, "k":0.25, "r":0.5, "a":1.0}
// FOLD_OR_CHECK: use 0.0 (fold terminates; check is main use)
static constexpr float NLHE_ACTION_ENC[4] = {0.0f, 0.25f, 0.5f, 1.0f};

static constexpr const char* NLHE_ACTION_NAMES[] = {
    "FOLD_OR_CHECK", "CALL", "RAISE", "ALL_IN"
};

// ── Game state ────────────────────────────────────────────────────────────────
struct NLHEState {
    int8_t  hole_cards[NLHE_NUM_PLAYERS][2];
    int8_t  board[5];
    int8_t  street;            // 0=preflop, 1=flop, 2=turn, 3=river
    int8_t  current_player;
    int8_t  raises_this_street;
    int8_t  last_aggressor;    // -1=no bet yet, -2=one check done
    float   pot;
    float   stacks[2];
    float   street_invest[2];  // chips into THIS street (for call sizing)
    int8_t  action_history[NLHE_MAX_HISTORY];
    int8_t  action_count;
    bool    folded[2];
    bool    terminal;
    float   payoff_p0;         // P0's net gain (signed)
    NLHEGameConfig cfg;
};

struct NLHEDeal {
    int8_t hole_cards[NLHE_NUM_PLAYERS][2];
    int8_t board[5];
};

// ── Hand evaluator ────────────────────────────────────────────────────────────
class HandEvaluator {
public:
    static int32_t evaluate(const int8_t* cards, int n_cards);
    static int     compare_hands(const int8_t* hole0, const int8_t* hole1,
                                  const int8_t* board,  int n_board);
private:
    static int32_t eval5(const int8_t* cards);
    static int32_t best_of_seven(const int8_t* cards);
};

// ── Game ──────────────────────────────────────────────────────────────────────
class NLHEGame {
public:
    static NLHEDeal  sample_deal(std::mt19937& rng);
    static NLHEState initial_state(const NLHEDeal& deal,
                                    const NLHEGameConfig& cfg = NLHEGameConfig{});
    static std::vector<NLHEAction> legal_actions(const NLHEState& state);
    static NLHEState apply_action(const NLHEState& state, NLHEAction action);
    static std::string info_set_key(const NLHEState& state, int player);
    static float bet_amount(const NLHEState& state);   // size for NLHE_RAISE
    static int   visible_board_cards(int street) { return BOARD_CARDS_BY_STREET[street]; }
};

} // namespace cfr