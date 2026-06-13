/**
 * Browser-native tabular Counterfactual Regret Minimization.
 *
 * Direct port of src/solvers/cfr.py — same algorithm, same convergence
 * guarantees, same calling conventions. Used as the CFR core inside the
 * gadget-game subgame solver. Runs entirely in the user's browser; no
 * server / WASM build needed.
 *
 * Convergence properties (unchanged from Python):
 *   * Linear CFR (Brown & Sandholm 2019): cumulative-strategy weighted by
 *     iteration number → O(1/T) convergence to Nash.
 *   * CFR+ (Tammelin 2014): cumulative regret clamped ≥ 0 for faster
 *     convergence on Leduc-scale games. Off by default.
 *
 * Scale envelope: tabular, so info-set count must be small. Subgame
 * solving (gadget game on a specific flop with bucketed ranges) has
 * O(thousands) of info sets — fine for in-browser. Full NLHE has ~10^160,
 * so CFR is for subgames only, not the whole game.
 */

import type { ExtensiveFormGame, History, InfoSetKey, InfoSetData, Player } from './types'
import { makeInfoSet, currentStrategy } from './types'

export interface CFROptions {
  /** Linear CFR weighting: cumulative_strategy weighted by t. Default true. */
  linearAveraging?: boolean
  /** CFR+: clamp cumulative regret ≥ 0 every iteration. Default false. */
  cfrPlus?: boolean
}

export class CFRSolver {
  readonly game: ExtensiveFormGame
  readonly infoSets: Map<InfoSetKey, InfoSetData> = new Map()
  iterations = 0
  private readonly linearAveraging: boolean
  private readonly cfrPlus: boolean

  constructor(game: ExtensiveFormGame, opts: CFROptions = {}) {
    this.game = game
    this.linearAveraging = opts.linearAveraging ?? true
    this.cfrPlus = opts.cfrPlus ?? false
  }

  /** Pre-allocates buffers for repeated recursion under the hot path. */
  private getOrCreateInfoSet(key: InfoSetKey, actions: number[]): InfoSetData {
    let info = this.infoSets.get(key)
    if (info === undefined) {
      info = makeInfoSet(actions)
      this.infoSets.set(key, info)
    }
    return info
  }

  /**
   * Recursive CFR traversal. Returns counterfactual value for the traverser
   * at this history node.
   */
  private cfrRecursive(
    history: History,
    reachProbs: Float64Array,
    traversingPlayer: Player,
  ): number {
    if (this.game.isTerminal(history)) {
      return this.game.terminalPayoffs(history)[traversingPlayer]
    }

    const player    = this.game.currentPlayer(history)
    const actions   = this.game.legalActions(history)
    const infoKey   = this.game.infoSetKey(history, player)
    const infoSet   = this.getOrCreateInfoSet(infoKey, actions)
    const strategy  = currentStrategy(infoSet)
    const numActions = actions.length

    // Accumulate strategy for the acting player only (matches Python).
    if (player === traversingPlayer) {
      const weight = this.linearAveraging ? (this.iterations + 1) : 1
      const playerReach = reachProbs[player]
      for (let i = 0; i < numActions; i++) {
        infoSet.cumulativeStrategy[i] += weight * playerReach * strategy[i]
      }
    }

    // Compute counterfactual value for each action.
    const actionValues = new Float64Array(numActions)
    for (let i = 0; i < numActions; i++) {
      const newHistory = this.game.applyAction(history, actions[i])
      const newReach   = new Float64Array(reachProbs)
      newReach[player] *= strategy[i]
      actionValues[i] = this.cfrRecursive(newHistory, newReach, traversingPlayer)
    }

    let nodeValue = 0
    for (let i = 0; i < numActions; i++) nodeValue += strategy[i] * actionValues[i]

    // Regret update for the acting player.
    if (player === traversingPlayer) {
      // Counterfactual reach = product of all OTHER players' reach probs.
      let cfReach = 1
      const np = this.game.numPlayers()
      for (let p = 0; p < np; p++) {
        if (p !== player) cfReach *= reachProbs[p]
      }
      for (let i = 0; i < numActions; i++) {
        let updated = infoSet.cumulativeRegret[i]
                      + cfReach * (actionValues[i] - nodeValue)
        if (this.cfrPlus && updated < 0) updated = 0
        infoSet.cumulativeRegret[i] = updated
      }
    }

    return nodeValue
  }

  /**
   * Run iterations of CFR. Returns the average-strategy map after solving.
   *
   * For a 2-player zero-sum game with O(few-thousand) info sets, 100-200
   * iterations suffice for a converged subgame solution.
   */
  solve(iterations: number, onProgress?: (it: number) => void): void {
    const numPlayers = this.game.numPlayers()
    const initial    = this.game.initialHistories()

    for (let t = 1; t <= iterations; t++) {
      for (let player = 0; player < numPlayers; player++) {
        for (const [history, chanceProb] of initial) {
          const reach = new Float64Array(numPlayers)
          reach.fill(chanceProb)
          this.cfrRecursive(history, reach, player)
        }
      }
      this.iterations++
      if (onProgress && (t % 10 === 0 || t === iterations)) onProgress(t)
    }
  }
}
