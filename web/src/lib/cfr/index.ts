/**
 * Public exports for the browser-native CFR + safe subgame solver.
 *
 * Layers (built bottom-up):
 *   1. types.ts          — generic ExtensiveFormGame interface + InfoSetData utils
 *   2. cfr-solver.ts     — tabular CFR algorithm (linear averaging + CFR+)
 *   3. postflop-nlhe.ts  — bit-identical TS port of PostflopNLHE state machine
 *   4. subgame-game.ts   — SubgameGame + GadgetGame (gadget = safe re-solve wrapper)
 *   5. subgame-solver.ts — SafeSubgameSolver, SubgameStrategy, rollout EV
 *
 * Typical usage:
 *   import { PostflopNLHE, SafeSubgameSolver, handKey } from '@/lib/cfr'
 *   import { encode, _evaluateCards } from '@/lib/onnx/encoder'
 *   import { queryStrategy } from '@/lib/onnx/session'
 *
 *   const base = new PostflopNLHE({ evaluator: _evaluateCards, startingStack: 50 })
 *   const solver = new SafeSubgameSolver({
 *     baseGame: base,
 *     blueprint: queryStrategy,
 *     encoder: (h, p) => encode({ ... }),  // adapt your state to EncodeInput
 *   })
 *   const refined = await solver.solve({
 *     rootHistory, heroPlayer, heroRange, opponentRange,
 *     iterations: 100,
 *   })
 *   const action = refined?.queryAction(rootHistory, heroPlayer)
 */

export type {
  ExtensiveFormGame, Action, Player, InfoSetKey, History, InfoSetData,
} from './types'
export {
  makeInfoSet, currentStrategy, averageStrategy,
} from './types'

export { CFRSolver, type CFROptions } from './cfr-solver'

export {
  PostflopNLHE,
  type NLHEHistory, type NLHEState, type PostflopNLHEConfig, type HandEvaluator,
} from './postflop-nlhe'

export {
  SubgameGame, GadgetGame, OPT_OUT,
  handKey, parseHandKey,
  type Range, type SubgameGameOpts, type GadgetGameOpts,
} from './subgame-game'

export {
  SafeSubgameSolver, SubgameStrategy,
  type BlueprintQuery, type Encoder,
  type SafeSubgameSolverOpts, type ResolveOpts,
} from './subgame-solver'

export {
  makeBlueprintQuery, makeEncoder, createSafeSubgameSolver,
} from './adapter'
