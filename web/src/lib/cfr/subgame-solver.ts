/**
 * SafeSubgameSolver — TypeScript port of src/solvers/subgame_solver.py.
 *
 * Full safe re-solving pipeline:
 *   1. estimateBlueprintEV — sample blueprint EV per opponent hand
 *   2. construct GadgetGame with those EVs as opt-out terminals
 *   3. run CFR on the gadget for `iterations` steps
 *   4. extract hero's average strategy, strip the opt-out slot
 *   5. wrap as SubgameStrategy with .queryAction(history, player)
 *
 * The browser version sticks to small iteration counts (100-200) for
 * interactive feel. Subgame info-set count is bounded by the deal
 * enumeration (heroRange × oppRange × subgame tree), so this stays
 * tractable client-side.
 */

import type { ExtensiveFormGame, Action, History, InfoSetKey, Player } from './types'
import { averageStrategy, currentStrategy } from './types'
import { CFRSolver } from './cfr-solver'
import type { PostflopNLHE, NLHEHistory } from './postflop-nlhe'
import {
  SubgameGame, GadgetGame, OPT_OUT,
  type Range, handKey, parseHandKey,
} from './subgame-game'

/** Function that returns blueprint action probabilities for an encoded state. */
export type BlueprintQuery = (
  state: Float32Array,
  numLegalActions: number,
) => Promise<Float32Array>

/** Encoder from game-state to model input. */
export type Encoder = (history: History, player: Player) => Float32Array

export interface SafeSubgameSolverOpts {
  baseGame:      PostflopNLHE
  blueprint:     BlueprintQuery
  encoder:       Encoder
}

export interface ResolveOpts {
  rootHistory:   NLHEHistory
  heroPlayer:    Player
  heroRange:     Range
  opponentRange: Range
  iterations?:   number     // default 100
  maxDeals?:     number     // default 200
  rng?:          () => number
  onProgress?:   (it: number) => void
}

/** Query wrapper over the solved strategy dict. */
export class SubgameStrategy {
  private readonly dict: Map<InfoSetKey, Float64Array>
  private readonly game: SubgameGame

  constructor(dict: Map<InfoSetKey, Float64Array>, game: SubgameGame) {
    this.dict = dict
    this.game = game
  }

  /** Returns strategy probs at this history for the player to act. */
  query(history: History, player: Player): Float64Array {
    const key = this.game.infoSetKey(history, player)
    const probs = this.dict.get(key)
    if (probs !== undefined) return probs
    // Unseen info set → uniform over current legal actions.
    const actions = this.game.legalActions(history)
    const n = actions.length
    const out = new Float64Array(n)
    out.fill(1 / n)
    return out
  }

  /** Returns the most-likely action (argmax) for the current player. */
  queryAction(history: History, player: Player): Action {
    const probs = this.query(history, player)
    const actions = this.game.legalActions(history)
    let bestI = 0, bestP = -1
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > bestP) { bestI = i; bestP = probs[i] }
    }
    return actions[bestI]
  }

  /** Sample an action from the strategy (stochastic play). */
  sampleAction(history: History, player: Player, rng: () => number): Action {
    const probs = this.query(history, player)
    const actions = this.game.legalActions(history)
    let r = rng()
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i]
      if (r <= 0) return actions[i]
    }
    return actions[actions.length - 1]
  }

  /** Number of solved info sets — useful for diagnostics. */
  size(): number { return this.dict.size }
}

export class SafeSubgameSolver {
  private readonly baseGame:  PostflopNLHE
  private readonly blueprint: BlueprintQuery
  private readonly encoder:   Encoder

  constructor(opts: SafeSubgameSolverOpts) {
    this.baseGame  = opts.baseGame
    this.blueprint = opts.blueprint
    this.encoder   = opts.encoder
  }

  /**
   * Recursive expected-value rollout of the blueprint over a subgame subtree.
   * Returns (p0_ev, p1_ev). Async because each node may need a blueprint
   * inference call.
   */
  private async rolloutEV(history: History): Promise<[number, number]> {
    if (this.baseGame.isTerminal(history)) {
      const p = this.baseGame.terminalPayoffs(history)
      return [p[0], p[1]]
    }

    const player  = this.baseGame.currentPlayer(history)
    const actions = this.baseGame.legalActions(history)
    const stateVec = this.encoder(history, player)
    const probs    = await this.blueprint(stateVec, actions.length)

    let ev0 = 0, ev1 = 0
    for (let i = 0; i < actions.length; i++) {
      const prob = probs[i]
      if (prob < 1e-9) continue
      const nextH = this.baseGame.applyAction(history, actions[i])
      const [n0, n1] = await this.rolloutEV(nextH)
      ev0 += prob * n0
      ev1 += prob * n1
    }
    return [ev0, ev1]
  }

  /**
   * Estimate the blueprint EV for each opponent hand at the subgame root.
   * EV(opp_cards) = Σ_{hero_cards} P(hero_cards) × bp_payoff(opp, hero).
   */
  private async estimateBlueprintEV(
    rootHistory:  NLHEHistory,
    heroPlayer:   Player,
    heroRange:    Range,
    opponentRange: Range,
  ): Promise<Map<string, number>> {
    const oppPlayer: Player = ((1 - heroPlayer) as Player)
    const board   = rootHistory[2]
    const boardSet = new Set<number>(board)
    const evByOpp = new Map<string, number>()

    for (const [oKey, oProb] of opponentRange) {
      if (oProb <= 0) continue
      const oppCards = parseHandKey(oKey)
      if (boardSet.has(oppCards[0]) || boardSet.has(oppCards[1])) continue

      let totalEv     = 0
      let totalWeight = 0

      for (const [hKey, hProb] of heroRange) {
        if (hProb <= 0) continue
        const heroCards = parseHandKey(hKey)
        if (boardSet.has(heroCards[0]) || boardSet.has(heroCards[1])) continue
        if (heroCards[0] === oppCards[0] || heroCards[0] === oppCards[1]) continue
        if (heroCards[1] === oppCards[0] || heroCards[1] === oppCards[1]) continue

        const p0 = heroPlayer === 0 ? heroCards : oppCards
        const p1 = heroPlayer === 0 ? oppCards  : heroCards
        const restOfRoot = rootHistory.slice(2)
        const dealHistory = [p0 as [number, number], p1 as [number, number],
                              ...restOfRoot] as unknown as History

        const [ev0, ev1] = await this.rolloutEV(dealHistory)
        const oppEv = oppPlayer === 1 ? ev1 : ev0
        totalEv     += hProb * oppEv
        totalWeight += hProb
      }

      if (totalWeight > 0) evByOpp.set(oKey, totalEv / totalWeight)
    }

    return evByOpp
  }

  /**
   * Safe re-solve: build the gadget game, solve it with CFR, return the
   * hero's average strategy with the opt-out slot stripped.
   *
   * Returns null if the subgame has no valid initial deals (e.g. all hero/opp
   * combinations conflict with the board).
   */
  async solve(opts: ResolveOpts): Promise<SubgameStrategy | null> {
    const iterations = opts.iterations ?? 100
    const maxDeals   = opts.maxDeals   ?? 200
    const rng        = opts.rng        ?? Math.random

    // 1. Pre-compute opp blueprint EV per hand.
    const bpEV = await this.estimateBlueprintEV(
      opts.rootHistory, opts.heroPlayer,
      opts.heroRange, opts.opponentRange,
    )

    // 2. Build gadget game with those EVs as opt-out terminals.
    const gadget = new GadgetGame({
      baseGame:    this.baseGame,
      rootHistory: opts.rootHistory,
      heroPlayer:  opts.heroPlayer,
      heroRange:   opts.heroRange,
      opponentRange: opts.opponentRange,
      maxDeals, rng,
      blueprintEVByOppCards: bpEV,
    })

    if (gadget.initialHistories().length === 0) return null

    // 3. Solve gadget game with CFR.
    const solver = new CFRSolver(gadget as ExtensiveFormGame, { linearAveraging: true })
    solver.solve(iterations, opts.onProgress)

    // 4. Extract hero strategy, strip the OPT_OUT slot from any opponent info-set
    //    that has it (gadget root only).
    const strategyDict = new Map<InfoSetKey, Float64Array>()
    for (const [key, info] of solver.infoSets) {
      const avg = averageStrategy(info)
      const optIdx = info.actions.indexOf(-1)   // -1 is our OPT_OUT id
      if (optIdx >= 0) {
        // Drop the opt-out slot, re-normalise.
        const real = new Float64Array(avg.length - 1)
        let total = 0
        let j = 0
        for (let i = 0; i < avg.length; i++) {
          if (i === optIdx) continue
          real[j++] = avg[i]
          total += avg[i]
        }
        if (total > 0) {
          for (let i = 0; i < real.length; i++) real[i] /= total
        } else {
          real.fill(1 / real.length)
        }
        strategyDict.set(key, real)
      } else {
        strategyDict.set(key, avg)
      }
    }

    return new SubgameStrategy(strategyDict, gadget)
  }
}
