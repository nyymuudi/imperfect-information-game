/**
 * Glue layer between the TS CFR/subgame-solver stack and the existing ONNX
 * blueprint inference + state encoder.
 *
 * Provides:
 *   - makeBlueprintQuery — wraps queryStrategyRaw as a BlueprintQuery
 *   - makeEncoder        — wraps encode() as a generic Encoder over NLHEHistory
 *   - createSafeSubgameSolver — convenience factory wiring everything together
 *
 * The TS subgame solver is intentionally decoupled from concrete ONNX /
 * encoder implementations; this file is the only one that imports both
 * `@/lib/onnx/*` and `@/lib/cfr/*`.
 */

import { encode, type EncodeInput, _evaluateCards } from '../onnx/encoder'
import { queryStrategyRaw } from '../onnx/session'
import { PostflopNLHE, type NLHEHistory } from './postflop-nlhe'
import { SafeSubgameSolver, type Encoder, type BlueprintQuery } from './subgame-solver'

// Action char → numeric encoding accepted by encoder's ACTION_ENC.
// (encoder.ts maps 0=fold/check, 1=call, 2=raise, 3=all-in.)
const CHAR_TO_ACTION_ID: Record<string, number> = {
  f: 0, c: 0, k: 1, r: 2, a: 3,
}

export function makeBlueprintQuery(): BlueprintQuery {
  return (state, numLegalActions) =>
    queryStrategyRaw(state, numLegalActions)
}

/**
 * Build an Encoder that maps an NLHEHistory + player → the same Float32Array
 * the deployed ONNX model expects. Internally calls `encode()` from the
 * onnx encoder so any future bucket-scheme changes propagate automatically.
 */
export function makeEncoder(baseGame: PostflopNLHE): Encoder {
  return (history, player) => {
    const h = history as NLHEHistory
    const state = baseGame.parseState(history)
    const ourHole = h[player] as readonly [number, number]
    const visibleBoard = [...baseGame.visibleBoard(history)]
    // Action history: each entry is a single char ∈ {f,c,k,r,a}.
    const actions = (h.slice(3) as string[])
      .map(c => CHAR_TO_ACTION_ID[c])
      .filter((v): v is number => v !== undefined)

    const input: EncodeInput = {
      holeCards:    [ourHole[0], ourHole[1]],
      boardCards:   visibleBoard,
      street:       Math.min(state.streetIdx, 3) as 0 | 1 | 2 | 3,
      pot:          state.pot,
      toCall:       state.toCall,
      myStack:      state.stacks[player],
      oppStack:     state.stacks[1 - player],
      actionHistory: actions,
    }
    return encode(input)
  }
}

/**
 * Convenience factory: returns a SafeSubgameSolver wired up with the
 * project's ONNX blueprint + tree42 encoder + PostflopNLHE state machine.
 *
 * Call once per session and cache it.
 */
export function createSafeSubgameSolver(opts?: {
  startingStack?:      number
  maxRaisesPerStreet?: number
  raiseFraction?:      number
}): SafeSubgameSolver {
  const baseGame = new PostflopNLHE({
    evaluator:           _evaluateCards,
    startingStack:       opts?.startingStack       ?? 50,
    maxRaisesPerStreet:  opts?.maxRaisesPerStreet  ?? 1,
    raiseFraction:       opts?.raiseFraction       ?? 0.75,
  })
  return new SafeSubgameSolver({
    baseGame,
    blueprint: makeBlueprintQuery(),
    encoder:   makeEncoder(baseGame),
  })
}
