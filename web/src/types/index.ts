export type Street = 0 | 1 | 2 | 3
export const STREET_NAMES: Record<Street, string> = {
  0: 'Preflop',
  1: 'Flop',
  2: 'Turn',
  3: 'River',
}

// Actions match NLHEAction enum in nlhe_game.hpp
export type ActionIndex = 0 | 1 | 2 | 3
export const ACTION_LABELS: Record<ActionIndex, string> = {
  0: 'Fold / Check',
  1: 'Call',
  2: 'Raise',
  3: 'All-in',
}

export interface SolverRun {
  id: string
  created_at: string
  game: string
  iterations: number
  exploitability: number | null
  model_path: string
}
