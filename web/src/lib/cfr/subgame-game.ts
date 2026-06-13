/**
 * SubgameGame + GadgetGame — TypeScript port of src/solvers/subgame_solver.py.
 *
 * SubgameGame: PostflopNLHE restricted to a specific subgame rooted at the
 * current spot. initialHistories enumerates (hero_range × opp_range) into
 * concrete deals; CFR solves over this restricted tree.
 *
 * GadgetGame: SubgameGame + opt-out option for the OPPONENT at the subgame
 * root. The opt-out terminal payoff equals the opponent's blueprint EV for
 * those specific hole cards. Solving to Nash yields a hero strategy that is
 * provably at least as non-exploitable as the blueprint (Moravčík 2016,
 * Burch 2014 Theorem 1).
 */

import type { ExtensiveFormGame, Action, History, InfoSetKey, Player } from './types'
import type { PostflopNLHE, NLHEHistory } from './postflop-nlhe'

/** Hand → probability mass in a player's range. */
export type Range = Map<string, number>   // key = "card1,card2" (sorted)

/** Stable string key for a 2-card hand. */
export function handKey(cards: readonly [number, number]): string {
  const [a, b] = cards
  return a < b ? `${a},${b}` : `${b},${a}`
}

/** Parse a hand key back to a [card1, card2] tuple. */
export function parseHandKey(key: string): [number, number] {
  const [a, b] = key.split(',').map(Number)
  return [a, b]
}

export interface SubgameGameOpts {
  baseGame:     PostflopNLHE
  rootHistory:  NLHEHistory
  heroPlayer:   Player
  heroRange:    Range
  opponentRange: Range
  maxDeals?:    number     // default 300
  rng?:         () => number  // default Math.random
}

export class SubgameGame implements ExtensiveFormGame {
  readonly baseGame:    PostflopNLHE
  readonly rootHistory: NLHEHistory
  readonly heroPlayer:  Player
  protected readonly heroRange: Range
  protected readonly oppRange:  Range
  protected readonly maxDeals:  number
  protected readonly rng: () => number
  /** Number of actions BEFORE the subgame root (prefix to strip). */
  protected readonly prefixLen: number
  /** Pre-computed (deal, prob) tuples — built once on construction. */
  protected readonly initial: Array<[History, number]>

  constructor(opts: SubgameGameOpts) {
    this.baseGame    = opts.baseGame
    this.rootHistory = opts.rootHistory
    this.heroPlayer  = opts.heroPlayer
    this.heroRange   = opts.heroRange
    this.oppRange    = opts.opponentRange
    this.maxDeals    = opts.maxDeals ?? 300
    this.rng         = opts.rng     ?? Math.random
    this.prefixLen   = this.rootHistory.length - 3
    this.initial     = this.buildInitial()
  }

  numPlayers(): number { return 2 }

  initialHistories(): Array<[History, number]> { return this.initial }

  isTerminal(h: History): boolean        { return this.baseGame.isTerminal(h) }
  terminalPayoffs(h: History): number[]  { return this.baseGame.terminalPayoffs(h) }
  currentPlayer(h: History): Player      { return this.baseGame.currentPlayer(h) }
  legalActions(h: History): Action[]     { return this.baseGame.legalActions(h) }
  applyAction(h: History, a: Action): History {
    return this.baseGame.applyAction(h, a)
  }

  /**
   * Local info-set key: strips the root-prefix actions so the subgame CFR
   * builds a fresh strategy table over the SUBGAME-LOCAL action space.
   */
  infoSetKey(history: History, player: Player): InfoSetKey {
    const h = history as NLHEHistory
    const myCards = h[player] as readonly [number, number]
    const visible = this.baseGame.visibleBoard(history)
    const allActions = (h.slice(3) as string[])
    const subgameActions = allActions.slice(this.prefixLen).join('')
    return `${myCards.join(',')}|${visible.join(',')}|${subgameActions}`
  }

  /**
   * Enumerate (hero_card, opp_card) ∈ heroRange × oppRange combos that don't
   * conflict with the board or each other. Normalises probabilities, then
   * sub-samples if over maxDeals.
   */
  protected buildInitial(): Array<[History, number]> {
    const board = this.rootHistory[2]
    const boardSet = new Set<number>(board)
    const result: Array<[History, number]> = []

    for (const [hKey, hProb] of this.heroRange) {
      if (hProb <= 0) continue
      const heroCards = parseHandKey(hKey)
      if (boardSet.has(heroCards[0]) || boardSet.has(heroCards[1])) continue

      for (const [oKey, oProb] of this.oppRange) {
        if (oProb <= 0) continue
        const oppCards = parseHandKey(oKey)
        if (boardSet.has(oppCards[0]) || boardSet.has(oppCards[1])) continue
        if (heroCards[0] === oppCards[0] || heroCards[0] === oppCards[1]) continue
        if (heroCards[1] === oppCards[0] || heroCards[1] === oppCards[1]) continue

        const p0 = this.heroPlayer === 0 ? heroCards : oppCards
        const p1 = this.heroPlayer === 0 ? oppCards : heroCards
        // Substitute these cards into root_history at slots [0] and [1].
        const restOfRoot = this.rootHistory.slice(2)
        const dealHistory = [p0 as [number, number], p1 as [number, number],
                              ...restOfRoot] as unknown as History
        result.push([dealHistory, hProb * oProb])
      }
    }

    if (result.length === 0) return result

    // Normalise.
    let total = 0
    for (const [, p] of result) total += p
    if (total <= 0) return []

    let normalised: Array<[History, number]> = result.map(([h, p]) => [h, p / total])

    // Sub-sample if over maxDeals (no-replacement, probability-weighted).
    if (normalised.length > this.maxDeals) {
      // Cumulative-distribution + random pick × maxDeals (with rejection).
      const picked: Array<[History, number]> = []
      const used = new Set<number>()
      const cdf = new Float64Array(normalised.length)
      let cum = 0
      for (let i = 0; i < normalised.length; i++) {
        cum += normalised[i][1]
        cdf[i] = cum
      }
      let attempts = 0
      const maxAttempts = this.maxDeals * 10
      while (picked.length < this.maxDeals && attempts < maxAttempts) {
        attempts++
        const r = this.rng() * cum
        // Binary search for the first cdf[i] ≥ r.
        let lo = 0, hi = cdf.length - 1
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if (cdf[mid] >= r) hi = mid
          else lo = mid + 1
        }
        if (!used.has(lo)) {
          used.add(lo)
          picked.push(normalised[lo])
        }
      }
      // Re-normalise the sub-sample.
      let sub = 0
      for (const [, p] of picked) sub += p
      normalised = sub > 0 ? picked.map(([h, p]) => [h, p / sub]) : picked
    }

    return normalised
  }
}

// ── GadgetGame ────────────────────────────────────────────────────────────────

/** Sentinel char inserted in history when opp picks opt-out. */
export const OPT_OUT = '__opt_out__'

export interface GadgetGameOpts extends SubgameGameOpts {
  /** Blueprint EV indexed by opponent hole-card key → payoff FOR OPPONENT. */
  blueprintEVByOppCards: Map<string, number>
}

export class GadgetGame extends SubgameGame {
  private readonly bpEV: Map<string, number>
  private readonly oppPlayer: Player

  constructor(opts: GadgetGameOpts) {
    super(opts)
    this.bpEV = opts.blueprintEVByOppCards
    this.oppPlayer = ((1 - this.heroPlayer) as Player)
  }

  /** Opt-out turns the node terminal. */
  override isTerminal(h: History): boolean {
    const nh = h as NLHEHistory
    if (nh.length > 3 && nh[nh.length - 1] === OPT_OUT) return true
    return super.isTerminal(h)
  }

  /** Opt-out payoff = opponent's blueprint EV for their specific hand. */
  override terminalPayoffs(h: History): number[] {
    const nh = h as NLHEHistory
    if (nh.length > 3 && nh[nh.length - 1] === OPT_OUT) {
      const oppCardsTuple = (this.heroPlayer === 0 ? nh[1] : nh[0])
      const oppEv = this.bpEV.get(handKey(oppCardsTuple)) ?? 0
      // Zero-sum convention: hero gets -oppEv.
      return this.heroPlayer === 0 ? [-oppEv, oppEv] : [oppEv, -oppEv]
    }
    return super.terminalPayoffs(h)
  }

  /** Opt-out is available to the opponent ONLY at the subgame root. */
  override legalActions(h: History): Action[] {
    const nh = h as NLHEHistory
    const subgameActions = nh.slice(3 + this.prefixLen)
    if (subgameActions.length === 0 && this.baseGame.currentPlayer(h) === this.oppPlayer) {
      // -1 is our action-id for opt-out; never collides with PostflopNLHE 0-3.
      return [-1, ...super.legalActions(h)]
    }
    return super.legalActions(h)
  }

  override applyAction(h: History, a: Action): History {
    if (a === -1) {
      const nh = h as NLHEHistory
      return [...nh, OPT_OUT] as unknown as History
    }
    return super.applyAction(h, a)
  }
}
