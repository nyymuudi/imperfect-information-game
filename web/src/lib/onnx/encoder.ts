/**
 * TypeScript port of NLHEStateEncoder (src/cpp_engine/src/torch_model.cpp).
 *
 * 2026-06-12: aktiivinen blueprint v7_long_production (1500-iter, tree42).
 * Encoder palautettu tree42-skeemaan tämän blueprintin tueksi. v3_coarse
 * (flat) ei toimi tällä encoderilla; jos haluat fallback v3:een, vaihda
 * BUCKET_SCHEME='flat' ja revertaa _boardSuperFine kutsuva kohta encode():ssa.
 *
 * Card-abstracted layout (36 dims, tree42 board):
 *   [0:8]    preflop hand bucket one-hot (K=8, bins on equity ∈ [0,1])
 *   [8:12]   board super one-hot         (4-way: HC/Pair/Made/Premium; zero preflop)
 *   [12:14]  board fine one-hot          (2-way low/high half of fine_4; zero preflop)
 *   [14:16]  RESERVED zero                (säilytetään 36-dim state-size:lle)
 *   [16:20]  street one-hot (0=preflop, 1=flop, 2=turn, 3=river)
 *   [20]     pot / (2 * starting_stack)
 *   [21]     to_call / (2 * starting_stack)
 *   [22]     my_stack / starting_stack
 *   [23]     opp_stack / starting_stack
 *   [24:32]  action history — last 8 actions (ACTION_ENC values)
 *   [32]     preflop equity (continuous)
 *   [33]     board_strength (continuous, hand-rank/MAX proxy)
 *   [34]     pot odds = to_call / (pot + to_call)
 *   [35]     SPR = min(stacks) / pot, normalised (cap at 10)
 *
 * Card encoding: card = rank * 4 + suit
 *   rank: 0=2, 1=3, ..., 12=A
 *   suit: 0=♣, 1=♦, 2=♥, 3=♠
 */

export const STATE_SIZE = 36
export const K_PREFLOP  = 8
export const K_BOARD    = 8
export const BUCKET_SCHEME = 'tree42' as const
// Equity range for normalised preflop bucket (from 2000-sim table):
//   weakest hand 72o ≈ 0.316, strongest AA ≈ 0.842.
export const EQ_MIN = 0.316
export const EQ_MAX = 0.842
// STARTING_STACK must match --stack used during training.
export const STARTING_STACK = 50.0
export const BOARD_CARDS_BY_STREET = [0, 3, 4, 5] // indexed by street 0-3

// Verified against NLHE_ACTION_ENC[4] = {0.0f, 0.25f, 0.5f, 1.0f} in nlhe_game.hpp
// Actions: 0=fold/check, 1=call, 2=raise, 3=all-in
export const ACTION_ENC: Record<number, number> = {
  0: 0.0,  // fold or check
  1: 0.25, // call
  2: 0.5,  // raise
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

// ── Hand evaluator ────────────────────────────────────────────────────────────
// Proper 5-7 card evaluator — monotone with C++ HandEvaluator (not bit-
// identical, but correct relative ranking). Parity test covers dims 0-111 and
// 122-123; dim 121 is an accepted residual (documented in test_parity.py).
//
// Categories: 0=high card, 1=pair, 2=two pair, 3=trips,
//             4=straight, 5=flush, 6=full house, 7=quads, 8=straight flush

function _straightHigh(ranksSorted: number[]): number {
  const u = [...new Set(ranksSorted)].sort((a, b) => b - a)
  // Wheel: A-2-3-4-5
  if (u.includes(12) && u.includes(3) && u.includes(2) && u.includes(1) && u.includes(0)) return 3
  for (let i = 0; i <= u.length - 5; i++) {
    if (u[i] - u[i + 4] === 4) return u[i]
  }
  return -1
}

export function _evaluateCards(cards: readonly number[]): number {
  const ranks = cards.map(c => Math.floor(c / 4))
  const suits = cards.map(c => c % 4)

  // Flush detection
  const suitCnt = [0, 0, 0, 0]
  for (const s of suits) suitCnt[s]++
  const flushSuit = suitCnt.findIndex(c => c >= 5)
  const flushRanks = flushSuit >= 0
    ? cards.filter(c => c % 4 === flushSuit).map(c => Math.floor(c / 4)).sort((a, b) => b - a)
    : []

  const ranksSorted = [...ranks].sort((a, b) => b - a)
  const sfHigh  = flushRanks.length >= 5 ? _straightHigh(flushRanks) : -1
  const strHigh = _straightHigh(ranksSorted)

  const cnt = new Array<number>(13).fill(0)
  for (const r of ranks) cnt[r]++

  const quads = cnt.map((c, r) => c === 4 ? r : -1).filter(r => r >= 0).sort((a, b) => b - a)
  const trips = cnt.map((c, r) => c === 3 ? r : -1).filter(r => r >= 0).sort((a, b) => b - a)
  const pairs = cnt.map((c, r) => c === 2 ? r : -1).filter(r => r >= 0).sort((a, b) => b - a)
  const kickers = ranksSorted.filter(r => cnt[r] === 1)

  let cat: number
  let tb: number[]

  if (sfHigh >= 0)                         { cat = 8; tb = [sfHigh] }
  else if (quads.length > 0)               { cat = 7; tb = [quads[0], kickers[0] ?? 0] }
  else if (trips.length > 0 && (pairs.length > 0 || trips.length > 1))
                                           { cat = 6; tb = [trips[0], pairs[0] ?? trips[1]] }
  else if (flushRanks.length >= 5)         { cat = 5; tb = flushRanks.slice(0, 5) }
  else if (strHigh >= 0)                   { cat = 4; tb = [strHigh] }
  else if (trips.length > 0)              { cat = 3; tb = [trips[0], ...kickers.slice(0, 2)] }
  else if (pairs.length >= 2)             { cat = 2; tb = [pairs[0], pairs[1], kickers[0] ?? 0] }
  else if (pairs.length === 1)            { cat = 1; tb = [pairs[0], ...kickers.slice(0, 3)] }
  else                                    { cat = 0; tb = kickers.slice(0, 5) }

  // Positional encoding (base-13) then normalize
  const BASE = 13
  let score = cat
  for (const t of tb.slice(0, 5)) score = score * BASE + t
  // MAX = straight flush (cat=8) with 5 ace-high tiebreakers
  const MAX = (8 * BASE + 12) * BASE * BASE * BASE * BASE + 12 * BASE * BASE * BASE + 12 * BASE * BASE + 12 * BASE + 12
  return Math.min(score / MAX, 1.0)
}

/**
 * Board strength: proper 5-7 card hand evaluation.
 * Returns 0 preflop (matches C++ behaviour).
 */
export function boardStrength(
  holeCards: [number, number],
  boardCards: number[],
): number {
  if (boardCards.length < 3) return 0.0
  return _evaluateCards([...holeCards, ...boardCards])
}

/**
 * Detect 5-7 card hand category 0-8 (HC..SF). Used by tree42 bucket logic.
 * Matches NLHEStateEncoder hand-evaluator categorisation in C++.
 */
function _handCategory(cards: number[]): number {
  const ranks = cards.map(c => Math.floor(c / 4))
  const suits = cards.map(c => c % 4)
  const suitCnt = [0, 0, 0, 0]
  for (const s of suits) suitCnt[s]++
  const flushSuit = suitCnt.findIndex(c => c >= 5)
  const flushRanks = flushSuit >= 0
    ? cards.filter(c => c % 4 === flushSuit).map(c => Math.floor(c / 4)).sort((a, b) => b - a)
    : []
  const ranksSorted = [...ranks].sort((a, b) => b - a)
  const sfHigh  = flushRanks.length >= 5 ? _straightHigh(flushRanks) : -1
  const strHigh = _straightHigh(ranksSorted)
  const cnt = new Array<number>(13).fill(0)
  for (const r of ranks) cnt[r]++
  const hasQuads = cnt.some(c => c === 4)
  const tripsCnt = cnt.filter(c => c === 3).length
  const pairsCnt = cnt.filter(c => c === 2).length
  if (sfHigh >= 0)                                return 8 // SF
  if (hasQuads)                                   return 7 // quads
  if (tripsCnt > 0 && (pairsCnt > 0 || tripsCnt > 1))
                                                  return 6 // full house
  if (flushRanks.length >= 5)                     return 5 // flush
  if (strHigh >= 0)                               return 4 // straight
  if (tripsCnt > 0)                               return 3 // trips
  if (pairsCnt >= 2)                              return 2 // two pair
  if (pairsCnt === 1)                             return 1 // pair
  return 0 // high card
}

/**
 * Tree42 super+fine board encoding:
 *   super_idx ∈ 0..3 (HC, Pair, MadeNonPaired, Premium)
 *   fine_4    ∈ 0..3 sub-bin within super (kategoriaspesifinen)
 *   fine_2    = fine_4 ≤ 1 ? 0 : 1 (kollapsoitu low/high half)
 *
 * Matches C++ scheme=3 branch in NLHEStateEncoder::encode().
 */
function _boardSuperFine(
  holeCards: [number, number],
  boardCards: number[],
): { superIdx: number; fine2: number } | null {
  if (boardCards.length < 3) return null
  const allCards = [...holeCards, ...boardCards]
  const cat = _handCategory(allCards)
  const ranks = allCards.map(c => Math.floor(c / 4))
  const cnt = new Array<number>(13).fill(0)
  for (const r of ranks) cnt[r]++

  let superIdx = 0
  let fine4 = 0

  if (cat === 0) {
    superIdx = 0
    const topRank = Math.max(...ranks)
    if (topRank === 12)      fine4 = 3   // A
    else if (topRank === 11) fine4 = 2   // K
    else if (topRank >= 8)   fine4 = 1   // T-Q
    else                     fine4 = 0   // 2-9
  } else if (cat === 1) {
    superIdx = 1
    let pairRank = -1
    for (let r = 12; r >= 0; r--) {
      if (cnt[r] >= 2) { pairRank = r; break }
    }
    if (pairRank === 12)      fine4 = 3
    else if (pairRank >= 10)  fine4 = 2
    else if (pairRank >= 7)   fine4 = 1
    else                      fine4 = 0
  } else if (cat >= 2 && cat <= 5) {
    superIdx = 2
    fine4 = cat - 2  // {twopair, trips, straight, flush}
  } else {
    superIdx = 3
    if (cat === 6)      fine4 = 0  // FH
    else if (cat === 7) fine4 = 1  // quads
    else {
      // SF: royal (top rank = A) vs non-royal
      const topRank = Math.max(...ranks)
      fine4 = (topRank === 12) ? 3 : 2
    }
  }

  const fine2 = fine4 <= 1 ? 0 : 1
  return { superIdx, fine2 }
}

/**
 * Encode game state into a 36-dim Float32Array.
 * Identical layout to NLHEStateEncoder::encode() in torch_model.cpp.
 */
export function encode(input: EncodeInput): Float32Array {
  const { holeCards, boardCards, street, pot, toCall,
          myStack, oppStack, actionHistory } = input

  const out = new Float32Array(STATE_SIZE)
  const NORM = 2.0 * STARTING_STACK
  const nVisible = BOARD_CARDS_BY_STREET[street]

  // [0:8] preflop equity bucket one-hot — normalised to [EQ_MIN, EQ_MAX]
  const equity = preflopEquity(holeCards[0], holeCards[1])
  const eqNorm = Math.max(0, Math.min(1, (equity - EQ_MIN) / (EQ_MAX - EQ_MIN)))
  const pfBucket = Math.min(Math.floor(eqNorm * K_PREFLOP), K_PREFLOP - 1)
  out[pfBucket] = 1.0

  // [8:16] tree42 board encoding:
  //   [8:12]  super one-hot
  //   [12:14] fine_2 one-hot
  //   [14:16] reserved zero
  const visibleBoard = boardCards.slice(0, nVisible)
  const brdStr = boardStrength(holeCards, visibleBoard)
  const tree42 = _boardSuperFine(holeCards, visibleBoard)
  if (tree42 !== null) {
    out[8 + tree42.superIdx] = 1.0
    out[12 + tree42.fine2]   = 1.0
  }

  // [16:20] street one-hot
  out[16 + Math.min(street, 3)] = 1.0

  // [20:24] betting scalars
  out[20] = Math.min(pot       / NORM,           1.0)
  out[21] = Math.min(toCall    / NORM,           1.0)
  out[22] = Math.min(myStack   / STARTING_STACK, 1.0)
  out[23] = Math.min(oppStack  / STARTING_STACK, 1.0)

  // [24:32] action history (last 8)
  const HIST_SLOTS = 8
  const histStart  = Math.max(0, actionHistory.length - HIST_SLOTS)
  for (let i = histStart; i < actionHistory.length; i++) {
    const slot = 24 + (i - histStart)
    const act  = actionHistory[i]
    if (act >= 0 && act <= 3) out[slot] = ACTION_ENC[act]
  }

  // [32] preflop equity (continuous)
  out[32] = equity

  // [33] board strength (continuous)
  out[33] = brdStr

  // [34] pot odds
  out[34] = pot + toCall > 0
    ? Math.min(toCall / (pot + toCall), 1.0)
    : 0.0

  // [35] SPR = min(stacks) / pot, normalised (cap at 10)
  const effectiveStack = Math.min(myStack, oppStack)
  out[35] = pot > 1e-6
    ? Math.min(effectiveStack / pot, 10.0) / 10.0
    : 1.0

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
