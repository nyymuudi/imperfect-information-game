/**
 * TypeScript port of PostflopNLHE (src/games/postflop_nlhe.py).
 *
 * Bit-identical state machine to the C++ NLHEGame::apply_action that trained
 * the deployed blueprint. The action alphabet matches both: 'f' fold, 'c' check,
 * 'k' call, 'r' raise, 'a' all-in. Cards are 0-51 (rank*4 + suit).
 *
 * Implements the ExtensiveFormGame interface from ./types so subgame solvers
 * can use it as their `baseGame`. Browser-native, no I/O.
 */

import type { ExtensiveFormGame, Action, History, InfoSetKey, Player } from './types'

/** History tuple as a TypeScript-friendly readonly tuple. */
export type NLHEHistory = readonly [
  readonly [number, number],          // p0 hole cards
  readonly [number, number],          // p1 hole cards
  readonly number[],                  // board cards (0-5)
  ...string[],                        // action chars: f, c, k, r, a
]

/** Replayed game state at the end of a history's actions. */
export interface NLHEState {
  stacks: [number, number]
  invested: [number, number]      // cumulative across hand
  streetInvest: [number, number]  // within this street
  pot: number
  toCall: number
  streetIdx: number               // 0=preflop, 1=flop, 2=turn, 3=river, 4=terminal
  streetName: 'preflop' | 'flop' | 'turn' | 'river'
  raisesThisStreet: number
  currentPlayer: 0 | 1
  lastAggressor: number           // -1 no bet, -2 one check done
  folded: [boolean, boolean]
  allIn: boolean
  terminal: boolean
}

export interface PostflopNLHEConfig {
  startingStack?:        number   // default 50 (chips, BB=2, SB=1)
  maxRaisesPerStreet?:   number   // default 1
  sb?:                   number   // default 1
  bb?:                   number   // default 2
  raiseFraction?:        number   // default 0.75 (pot fraction)
}

/** Externally provided hand evaluator. Returns higher = better hand. */
export type HandEvaluator = (cards: readonly number[]) => number

export class PostflopNLHE implements ExtensiveFormGame {
  readonly startingStack:      number
  readonly maxRaisesPerStreet: number
  readonly sb:                 number
  readonly bb:                 number
  readonly raiseFraction:      number
  /** Used for showdown comparison. Inject from caller (encoder.ts evaluator). */
  private readonly evaluator: HandEvaluator

  constructor(cfg: PostflopNLHEConfig & { evaluator: HandEvaluator }) {
    this.startingStack      = cfg.startingStack      ?? 50
    this.maxRaisesPerStreet = cfg.maxRaisesPerStreet ?? 1
    this.sb                 = cfg.sb                 ?? 1
    this.bb                 = cfg.bb                 ?? 2
    this.raiseFraction      = cfg.raiseFraction      ?? 0.75
    this.evaluator          = cfg.evaluator
  }

  numPlayers(): number { return 2 }

  initialHistories(): Array<[History, number]> {
    // PostflopNLHE is too large for tabular CFR — kept here for interface
    // compatibility. Subgame wrappers override initialHistories() with
    // hero_range × opp_range enumeration.
    throw new Error('PostflopNLHE is too large for tabular CFR. Wrap in a SubgameGame.')
  }

  /** Strip actions out of a history. */
  private getActions(history: History): string[] {
    const h = history as NLHEHistory
    return h.slice(3) as string[]
  }

  /**
   * Deterministic replay of the state machine — mirrors Python _parse_state
   * and C++ NLHEGame::apply_action exactly. CALL ends the street, CHECK uses
   * last_aggressor double-tap, raise sizing = (pot+owe) * raiseFraction.
   */
  parseState(history: History): NLHEState {
    const actions = this.getActions(history)

    const stacks: [number, number] = [this.startingStack - this.sb, this.startingStack - this.bb]
    let streetInvest: [number, number] = [this.sb, this.bb]
    let pot = this.sb + this.bb
    let street = 0
    let currentPlayer: 0 | 1 = 0       // SB acts first preflop
    let raisesThisStreet = 0
    let lastAggressor = -1
    const folded: [boolean, boolean] = [false, false]
    let allIn = false
    let terminal = false

    const advanceStreet = (): void => {
      street++
      raisesThisStreet = 0
      lastAggressor = -1
      streetInvest = [0, 0]
      if (street >= 4) {
        terminal = true
      } else {
        currentPlayer = 1
      }
    }

    const betAmount = (p: 0 | 1): number => {
      const owe = streetInvest[1 - p] - streetInvest[p]
      const effectivePot = pot + owe
      const raiseAdd = effectivePot * this.raiseFraction
      return Math.min(raiseAdd, stacks[p] - owe)
    }

    for (const a of actions) {
      if (terminal) break
      const p   = currentPlayer
      const opp: 0 | 1 = (1 - p) as 0 | 1
      const owe = streetInvest[opp] - streetInvest[p]

      if (a === 'f') {
        folded[p] = true
        terminal = true
        break
      } else if (a === 'c') {
        if (lastAggressor === -1) {
          lastAggressor = -2
          currentPlayer = opp
        } else {
          advanceStreet()
        }
      } else if (a === 'k') {
        const callAmt = Math.min(owe, stacks[p])
        stacks[p] -= callAmt
        streetInvest[p] += callAmt
        pot += callAmt
        advanceStreet()
      } else if (a === 'r') {
        const raiseAdd = betAmount(p)
        let total = owe + raiseAdd
        total = Math.min(total, stacks[p])
        stacks[p] -= total
        streetInvest[p] += total
        pot += total
        raisesThisStreet++
        lastAggressor = p
        currentPlayer = opp
      } else if (a === 'a') {
        const allinAdd = stacks[p]
        stacks[p] = 0
        streetInvest[p] += allinAdd
        pot += allinAdd
        allIn = true
        raisesThisStreet++
        lastAggressor = p
        currentPlayer = opp
      }
    }

    const p   = currentPlayer
    const opp: 0 | 1 = (1 - p) as 0 | 1
    const toCall = Math.max(0, streetInvest[opp] - streetInvest[p])
    const invested: [number, number] = [
      this.startingStack - stacks[0],
      this.startingStack - stacks[1],
    ]
    const streetNames = ['preflop', 'flop', 'turn', 'river'] as const

    return {
      stacks, invested, streetInvest, pot, toCall,
      streetIdx: street,
      streetName: streetNames[Math.min(street, 3)],
      raisesThisStreet, currentPlayer, lastAggressor,
      folded, allIn, terminal,
    }
  }

  /** Visible board cards given the current street. */
  visibleBoard(history: History): readonly number[] {
    const h = history as NLHEHistory
    const fullBoard = h[2]
    const street = this.parseState(history).streetIdx
    const n = [0, 3, 4, 5][Math.min(street, 3)]
    return fullBoard.slice(0, n)
  }

  isTerminal(history: History): boolean {
    const actions = this.getActions(history)
    if (actions.length === 0) return false
    return this.parseState(history).terminal
  }

  terminalPayoffs(history: History): number[] {
    if (!this.isTerminal(history)) throw new Error('Non-terminal history')
    const state = this.parseState(history)
    const invested = state.invested

    // Fold path
    if (state.folded[0] || state.folded[1]) {
      const folder = state.folded[0] ? 0 : 1
      const winner = 1 - folder
      const out = [0, 0]
      out[folder] = -invested[folder]
      out[winner] =  invested[folder]
      return out
    }

    // Showdown
    const h = history as NLHEHistory
    const p0Cards = h[0]
    const p1Cards = h[1]
    const board   = h[2].slice(0, 5)
    const h0 = this.evaluator([...p0Cards, ...board])
    const h1 = this.evaluator([...p1Cards, ...board])
    if (h0 > h1) return [ invested[1], -invested[1]]
    if (h1 > h0) return [-invested[0],  invested[0]]
    return [0, 0]  // tie
  }

  currentPlayer(history: History): Player {
    return this.parseState(history).currentPlayer
  }

  legalActions(history: History): Action[] {
    const state = this.parseState(history)
    if (state.terminal) return []
    const p = state.currentPlayer
    const owe = state.toCall
    const bet = owe > 0.001
    const canRaise = state.raisesThisStreet < this.maxRaisesPerStreet
    const myStack = state.stacks[p]
    const pot = state.pot

    const betAmount = (): number => {
      const effectivePot = pot + owe
      const raiseAdd = effectivePot * this.raiseFraction
      return Math.min(raiseAdd, myStack - owe)
    }

    // Actions encoded as integer ids matching the C++ enum
    //   0 = FOLD_OR_CHECK ('f' if facing bet, 'c' otherwise)
    //   1 = CALL ('k')
    //   2 = RAISE ('r')
    //   3 = ALL_IN ('a')
    const out: Action[] = []
    if (bet) {
      out.push(0)  // fold
      out.push(1)  // call
    } else {
      out.push(0)  // check
    }

    if (canRaise && myStack > owe + 0.01) {
      if (betAmount() > 0.01) out.push(2)
    }

    if (myStack > owe + 0.01) {
      const allinAdd = myStack - owe
      const raiseAdd = (canRaise && myStack > owe + 0.01) ? betAmount() : -1
      if (!canRaise || allinAdd > raiseAdd + 0.01) out.push(3)
    }

    return out
  }

  /** Map numeric action → char used in history strings. */
  private actionToChar(action: Action, facingBet: boolean): string {
    if (action === 0) return facingBet ? 'f' : 'c'
    if (action === 1) return 'k'
    if (action === 2) return 'r'
    if (action === 3) return 'a'
    throw new Error(`Unknown action: ${action}`)
  }

  applyAction(history: History, action: Action): History {
    const h = history as NLHEHistory
    const state = this.parseState(history)
    const facingBet = state.toCall > 0.001
    const ch = this.actionToChar(action, facingBet)
    return [...h, ch] as unknown as History
  }

  infoSetKey(history: History, player: Player): InfoSetKey {
    const h = history as NLHEHistory
    const myCards = h[player] as readonly [number, number]
    const board   = this.visibleBoard(history)
    const actions = this.getActions(history).join('')
    return `${myCards.join(',')}|${board.join(',')}|${actions}`
  }
}
