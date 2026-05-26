/**
 * TypeScript port of NLHEStateEncoder (src/cpp_engine/src/torch_model.cpp)
 *
 * Tensor layout (122 dims):
 *   [0:52]    hole cards one-hot
 *   [52:104]  visible board cards one-hot
 *   [104:108] street one-hot (0=preflop, 1=flop, 2=turn, 3=river)
 *   [108]     pot / (2 * starting_stack)
 *   [109]     to_call / (2 * starting_stack)
 *   [110]     my_stack / starting_stack
 *   [111]     opp_stack / starting_stack
 *   [112:120] action history — last 8 actions (ACTION_ENC values)
 *   [120]     preflop equity
 *   [121]     board_strength
 *
 * Card encoding: card = rank * 4 + suit
 *   rank: 0=2, 1=3, ..., 12=A
 *   suit: 0=♣, 1=♦, 2=♥, 3=♠
 */

export const STATE_SIZE = 122
export const STARTING_STACK = 200.0
export const BOARD_CARDS_BY_STREET = [0, 3, 4, 5] // indexed by street 0-3

// Must match NLHE_ACTION_ENC[] in nlhe_game.hpp — verify against C++ source
// Actions: 0=fold/check, 1=call, 2=raise, 3=all-in
export const ACTION_ENC: Record<number, number> = {
  0: 0.25, // fold or check
  1: 0.5,  // call
  2: 0.75, // raise
  3: 1.0,  // all-in
}

export interface EncodeInput {
  holeCards: [number, number]        // player's hole cards (0-51)
  boardCards: number[]               // visible board cards (0-5 cards)
  street: 0 | 1 | 2 | 3             // 0=preflop, 1=flop, 2=turn, 3=river
  pot: number
  toCall: number
  myStack: number
  oppStack: number
  actionHistory: number[]            // last ≤8 action ints (0-3)
}

export function cardRank(card: number): number {
  return Math.floor(card / 4)
}

export function cardSuit(card: number): number {
  return card % 4
}

/**
 * Preflop equity approximation — direct port of preflop_equity() in torch_model.hpp
 */
export function preflopEquity(card0: number, card1: number): number {
  const r0 = cardRank(card0)
  const r1 = cardRank(card1)
  const s0 = cardSuit(card0)
  const s1 = cardSuit(card1)
  const rankHigh = Math.max(r0, r1)
  const rankLow  = Math.min(r0, r1)
  const suited   = s0 === s1 && card0 !== card1

  let base = 0.30 + rankHigh * 0.026
  if (rankHigh === rankLow)          base += 0.15  // pocket pair
  if (rankHigh - rankLow === 1)      base += 0.02  // connector
  if (rankHigh - rankLow === 2)      base += 0.01  // one-gapper
  if (suited)                        base += 0.03
  return Math.min(base, 0.88)
}

/**
 * Simplified board strength approximation.
 * Full hand evaluation (HandEvaluator::evaluate) requires a poker hand
 * evaluator library — replace with e.g. `phe` or `pokersolver` for accuracy.
 * Returns 0 preflop (matches C++ behaviour).
 */
export function boardStrength(
  holeCards: [number, number],
  boardCards: number[],
): number {
  if (boardCards.length < 3) return 0.0

  // Approximate: count high cards and pairs
  const allCards = [...holeCards, ...boardCards]
  const ranks = allCards.map(cardRank)
  const rankCounts = new Map<number, number>()
  for (const r of ranks) rankCounts.set(r, (rankCounts.get(r) ?? 0) + 1)

  let score = 0
  for (const [rank, count] of rankCounts) {
    if (count >= 2) score += 0.15 * count   // pair/trips/quads
    score += rank * 0.004                    // high card contribution
  }
  return Math.min(score, 1.0)
}

/**
 * Encode game state into a 122-dim Float32Array.
 * Identical layout to NLHEStateEncoder::encode() in torch_model.cpp.
 */
export function encode(input: EncodeInput): Float32Array {
  const { holeCards, boardCards, street, pot, toCall,
          myStack, oppStack, actionHistory } = input

  const out = new Float32Array(STATE_SIZE)
  const NORM = 2.0 * STARTING_STACK

  // [0:52] hole cards one-hot
  for (const card of holeCards) {
    if (card >= 0 && card < 52) out[card] = 1.0
  }

  // [52:104] visible board cards one-hot
  const nVisible = BOARD_CARDS_BY_STREET[street]
  for (let i = 0; i < nVisible && i < boardCards.length; i++) {
    const card = boardCards[i]
    if (card >= 0 && card < 52) out[52 + card] = 1.0
  }

  // [104:108] street one-hot
  out[104 + Math.min(street, 3)] = 1.0

  // [108:112] betting scalars
  out[108] = Math.min(pot       / NORM,             1.0)
  out[109] = Math.min(toCall    / NORM,             1.0)
  out[110] = Math.min(myStack   / STARTING_STACK,   1.0)
  out[111] = Math.min(oppStack  / STARTING_STACK,   1.0)

  // [112:120] action history (last 8)
  const HIST_SLOTS = 8
  const histStart  = Math.max(0, actionHistory.length - HIST_SLOTS)
  for (let i = histStart; i < actionHistory.length; i++) {
    const slot = 112 + (i - histStart)
    const act  = actionHistory[i]
    if (act >= 0 && act <= 3) out[slot] = ACTION_ENC[act]
  }

  // [120] preflop equity
  out[120] = preflopEquity(holeCards[0], holeCards[1])

  // [121] board strength
  const visibleBoard = boardCards.slice(0, nVisible)
  out[121] = boardStrength(holeCards, visibleBoard)

  return out
}

// ── Card helpers ──────────────────────────────────────────────────────────────

export const RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
export const SUITS = ['♣','♦','♥','♠']

export function cardLabel(card: number): string {
  return RANKS[cardRank(card)] + SUITS[cardSuit(card)]
}

export function makeCard(rank: number, suit: number): number {
  return rank * 4 + suit
}
