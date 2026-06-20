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
//
// Multi-raise puu (Pluribus-tyylinen, 2026-06-14):
//   - raise_fractions[0..n_raise_fractions-1] sisältää käytetyt %:t pottista.
//   - raise_fraction (yksittäinen float) on backward-compat: jos
//     n_raise_fractions == 0, käytetään yksittäistä raise_fraction-arvoa.
//   - MAX_RAISE_FRACTIONS = 4 määrittää enum-kapasiteetin (RAISE_0..RAISE_3).
struct NLHEGameConfig {
    static constexpr int MAX_RAISE_FRACTIONS = 4;
    float starting_stack = 200.0f;  // chips (BB=2, SB=1)
    float sb             = 1.0f;
    float bb             = 2.0f;
    float raise_fraction = 0.75f;   // legacy single-raise (backward compat)
    float raise_fractions[MAX_RAISE_FRACTIONS] = {0.0f, 0.0f, 0.0f, 0.0f};
    int   n_raise_fractions = 0;    // 0 = use legacy raise_fraction
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
// 4–7 actions matching Python PostflopNLHE:
//   Single raise (legacy):  [FOLD_OR_CHECK, CALL, RAISE_0, ALL_IN]            (4)
//   Multi raise (Pluribus): [FOLD_OR_CHECK, CALL, RAISE_0..RAISE_K-1, ALL_IN] (3+K)
//
// NLHE_RAISE alias = NLHE_RAISE_0 for backward compatibility with code
// referencing the single-raise action symbol.
//
// Network output capacity = NLHE_NUM_ACTIONS = 6 = 1 + 1 + MAX_RAISE_FRACTIONS=4 ...
// wait — keep enum slot count fixed at 6 to bound regret sample size; ALL_IN
// lives at slot (2 + MAX_RAISE_FRACTIONS-1)+1 = 6. The actual legal_actions
// result is variable length; engine uses cfg.max_actions for masking.
enum NLHEAction : int8_t {
    NLHE_FOLD_OR_CHECK = 0,
    NLHE_CALL          = 1,
    NLHE_RAISE_0       = 2,
    NLHE_RAISE_1       = 3,
    NLHE_RAISE_2       = 4,
    NLHE_ALL_IN        = 5,
    NLHE_NUM_ACTIONS   = 6,
    // Backward-compat alias (callers still reference NLHE_RAISE).
    NLHE_RAISE         = NLHE_RAISE_0
};

// Slot remap for single-raise puu (max_actions==4):
//   FOLD_OR_CHECK=0, CALL=1, RAISE_0=2, ALL_IN→3 (was enum 5, doesn't fit).
// For multi-raise (max_actions==6) the enum value is used as-is.
//
// Pre-2026-06-14 single-raise blueprintit (v11 etc.) had ALL_IN at enum
// value 3 directly. The 2026-06-14 multi-raise refactor moved it to 5,
// silently breaking single-raise training (slot 5 ≥ max_actions=4 → all-in
// samples filtered out). This helper restores parity: single-raise mode
// emits/queries ALL_IN at slot 3 (matching v11 layout), multi-raise mode
// keeps it at slot 5 (where max_actions=6 covers it).
inline int nlhe_action_to_slot(NLHEAction a, int max_actions) {
    int slot = static_cast<int>(a);
    if (slot == NLHE_ALL_IN && max_actions == 4) return 3;
    return slot;
}

// Action history encoding — matches Python NLHEEncoder.ACTION_ENC values.
// Indices follow the enum above so action_history[i] can be used directly.
// {0:0.0, 1:0.25, 2:0.5 (RAISE_0 = legacy 'r'), 3:0.4 (RAISE_1, smaller),
//  4:0.6 (RAISE_2, larger), 5:1.0 (ALL_IN)}.
//
// Backward-compat: legacy single-raise puu emits NLHE_RAISE_0 (= NLHE_RAISE
// alias) whose history encoding 0.5 equals Python's legacy 'r' = 0.5 —
// pariteetti säilyy single-raise-tilassa, kaikki pre-2026-06-14 blueprintit
// jotka käyttivät 'r' = 0.5 saavat saman scalar-signaalin C++:sta.
//
// Pluribus-multi-raise: Python emits 'r0'/'r1'/'r2' with the same scalars
// (0.5/0.4/0.6) → C++ ↔ Python pariteetti säilyy myös multi-raise:ssa.
static constexpr float NLHE_ACTION_ENC[NLHE_NUM_ACTIONS] = {
    0.0f, 0.25f, 0.5f, 0.4f, 0.6f, 1.0f
};

static constexpr const char* NLHE_ACTION_NAMES[] = {
    "FOLD_OR_CHECK", "CALL", "RAISE_0", "RAISE_1", "RAISE_2", "ALL_IN"
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
    // EV-adjusted MCCFR support: track the street at which both players
    // first became all-in (both stacks == 0). -1 means "not all-in" or
    // "one-sided fold". On terminal-with-showdown, if 0 ≤ all_in_street < 3,
    // the post-betting board cards are runout variance that the equity-
    // adjusted terminal value should integrate over rather than use directly.
    int8_t  all_in_street;
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
    static float bet_amount(const NLHEState& state);   // size for NLHE_RAISE_0 (alias)
    // Per-index raise sizing for the Pluribus-style multi-raise puu.
    // raise_idx 0..n_raise_fractions-1 selects cfg.raise_fractions[idx]; if
    // n_raise_fractions == 0, falls back to legacy cfg.raise_fraction for
    // any raise_idx (i.e. single-raise puu).
    static float bet_amount_idx(const NLHEState& state, int raise_idx);
    static int   visible_board_cards(int street) { return BOARD_CARDS_BY_STREET[street]; }
};

} // namespace cfr