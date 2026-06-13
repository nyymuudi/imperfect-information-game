/**
 * Game-agnostic types for the in-browser CFR solver.
 *
 * Ports the abstract ExtensiveFormGame interface from src/games/base.py
 * (Python) into TypeScript so the same algorithm can run client-side.
 * Game-specific code (GadgetGame, SubgameGame) implements this interface.
 */

/** Action represented as a numeric id. Game decides mapping. */
export type Action = number

/** Player index, 0..num_players()-1. */
export type Player = number

/** Hashable info-set identifier — string is the safe portable choice. */
export type InfoSetKey = string

/** Game history — game-specific; can be any structured type. */
export type History = unknown

/**
 * Two-player zero-sum extensive-form game interface.
 *
 * MUST be deterministic across instances for CFR convergence: same inputs →
 * same outputs every call. Random choices live INSIDE history (the deal is
 * sampled once before solve, then frozen).
 */
export interface ExtensiveFormGame {
  numPlayers(): number

  /** Initial histories + their chance probabilities (sum to 1). */
  initialHistories(): Array<[History, number]>

  isTerminal(h: History): boolean
  /** Returns each player's payoff at a terminal node. Length = numPlayers(). */
  terminalPayoffs(h: History): number[]

  currentPlayer(h: History): Player
  legalActions(h: History): Action[]
  applyAction(h: History, a: Action): History

  /** Identifier under which CFR accumulates regrets/strategies. */
  infoSetKey(h: History, player: Player): InfoSetKey
}

/**
 * Per-info-set accumulator: cumulative regret + cumulative strategy.
 * Regret-matching converts cumulative regret → current strategy each visit;
 * cumulative strategy / total → Nash-convergent average strategy.
 */
export interface InfoSetData {
  actions: Action[]
  cumulativeRegret: Float64Array
  cumulativeStrategy: Float64Array
}

/** Build a fresh InfoSetData with zeroed accumulators. */
export function makeInfoSet(actions: Action[]): InfoSetData {
  const n = actions.length
  return {
    actions: [...actions],
    cumulativeRegret: new Float64Array(n),
    cumulativeStrategy: new Float64Array(n),
  }
}

/** Regret-matching strategy: σ ∝ R+, else uniform. */
export function currentStrategy(info: InfoSetData): Float64Array {
  const n = info.actions.length
  const out = new Float64Array(n)
  let total = 0
  for (let i = 0; i < n; i++) {
    const r = info.cumulativeRegret[i]
    if (r > 0) {
      out[i] = r
      total += r
    }
  }
  if (total > 0) {
    for (let i = 0; i < n; i++) out[i] /= total
    return out
  }
  out.fill(1 / n)
  return out
}

/** Average strategy: cumulative_strategy / total → Nash on convergence. */
export function averageStrategy(info: InfoSetData): Float64Array {
  const n = info.actions.length
  const out = new Float64Array(n)
  let total = 0
  for (let i = 0; i < n; i++) total += info.cumulativeStrategy[i]
  if (total > 0) {
    for (let i = 0; i < n; i++) out[i] = info.cumulativeStrategy[i] / total
    return out
  }
  out.fill(1 / n)
  return out
}
