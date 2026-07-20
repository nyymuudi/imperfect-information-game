/**
 * Lab notebook data — every displayed number carries a `source` (commit
 * hash or committed experiment-log path). No hand-typed number without a
 * source. See /lab. Numbers that could not be traced to a committed
 * source were omitted (listed in the session report, not shown here).
 *
 * Units: LBR is an mbb/decision proxy (local best-response regret), NOT
 * true exploitability. Cross-tree LBR (4 vs 5 vs 6 action slots) is NOT
 * comparable — the best-responder's max grows with the action set. Only
 * same-tree, same-metric numbers are compared on this page.
 */

export type Source = { ref: string; kind: 'commit' | 'log' }

// ── Blueprint lineage: LBR proxy per version ────────────────────────────────

export type BlueprintPoint = {
  version: string
  tree: string          // action-set descriptor (comparability boundary)
  metric: string        // which LBR variant
  lbr: number
  stderr?: number
  status: 'production' | 'superseded' | 'rejected'
  note: string
  source: Source
}

// NOTE: these span three metric generations and three trees. They are
// grouped by comparability on the page, never plotted as one line.
export const BLUEPRINTS: BlueprintPoint[] = [
  {
    version: 'v14d', tree: '4-action (single raise)',
    metric: 'LBR v1 (clairvoyant→LBR rewrite)', lbr: 1384, stderr: 120,
    status: 'superseded', note: 'v11-class baseline, 4-action standard',
    source: { ref: 'project_v14d_production_baseline', kind: 'log' },
  },
  {
    version: 'v15_c2_v5_aux', tree: '4-action + advisor cache',
    metric: 'LBR v1', lbr: 761, stderr: 88,
    status: 'superseded', note: 'CFR advisor cache + aux-EV head; deployed 2026-06',
    source: { ref: 'da77552', kind: 'commit' },
  },
  {
    version: 'v16a', tree: '4-action (single raise, no cache)',
    metric: 'LBR v1 (pad-encoder, n=400 seed42)', lbr: 1194, stderr: 177,
    status: 'superseded', note: 'single-raise control @2500 iters',
    source: { ref: '91d2d02', kind: 'commit' },
  },
  {
    version: 'v16b', tree: '6-action (2 raise sizes, no cache)',
    metric: 'LBR v1 (pad-encoder, n=400 seed42)', lbr: 944, stderr: 138,
    status: 'superseded', note: '2-size tree beats single-raise by 250±225',
    source: { ref: '91d2d02', kind: 'commit' },
  },
  {
    version: 'v16c', tree: '6-action (2 raise sizes + cache)',
    metric: 'LBR v1 (n=400 seed42)', lbr: 753, stderr: 118,
    status: 'production', note: 'PRODUCTION blueprint; 2-size + advisor cache',
    source: { ref: '91d2d02', kind: 'commit' },
  },
  {
    version: 'v16c', tree: '6-action (2 raise sizes + cache)',
    metric: 'LBR v2 (bayes+stratified, 5 seeds)', lbr: 998, stderr: 26,
    status: 'production', note: 'same net, v2 metric — NOT comparable to v1 761',
    source: { ref: '92f41ce', kind: 'commit' },
  },
  {
    version: 'v16e', tree: '6-slot capacity (3 raise sizes + cache)',
    metric: 'LBR v2 (bayes+stratified, 5 seeds)', lbr: 1116, stderr: 50,
    status: 'rejected', note: 'REJECTED: 3 sizes lose to 2 sizes (998±26) — richer tree not worth it at this budget',
    source: { ref: 'gate_b4_vs_b5.log', kind: 'log' },
  },
]

// ── CFV network generations: river-value-net holdout MAE (pot units) ─────────

export type CfvPoint = {
  gen: string
  data: string
  baseMae: number
  onPolicyMae: number
  note: string
  source: Source
}

export const CFV_NETS: CfvPoint[] = [
  {
    gen: 'b2', data: '5k base',
    baseMae: 0.046, onPolicyMae: 0.134,
    note: 'first river net; on-policy 2.9× base — solver-iterate ranges are sharper',
    source: { ref: '5920077', kind: 'commit' },
  },
  {
    gen: 'b4', data: '15k base + 4.5k on-policy',
    baseMae: 0.040, onPolicyMae: 0.068,
    note: 'ReBeL on-policy retraining halves the on-policy gap',
    source: { ref: 'acff67d', kind: 'commit' },
  },
  {
    gen: 'b5', data: '25k base + 10.5k on-policy',
    baseMae: 0.038, onPolicyMae: 0.062,
    note: 'PRODUCTION CFV net; more data, best aggregate MAE',
    source: { ref: 'gate_b4_vs_b5.log', kind: 'log' },
  },
]

// ── Seed variance: the gate replicated across two sampling methods ──────────

export type SeedDelta = { seed: number; delta: number }

// Uniform depth sampling, 20 seeds, b4 net. Source committed to
// docs/experiments/gate_uniform_20seed.txt.
export const GATE_UNIFORM: {
  method: string; deltas: SeedDelta[]; mean: number; se: number; t: number;
  source: Source;
} = {
  method: 'uniform depth sampling · 20 seeds · b4 net',
  deltas: [
    { seed: 42, delta: -91.6 }, { seed: 43, delta: 37.4 },
    { seed: 44, delta: -76.2 }, { seed: 45, delta: -183.7 },
    { seed: 46, delta: 15.2 }, { seed: 47, delta: -40.7 },
    { seed: 48, delta: 28.6 }, { seed: 49, delta: 114.1 },
    { seed: 50, delta: -76.0 }, { seed: 51, delta: 9.5 },
    { seed: 52, delta: -50.5 }, { seed: 53, delta: 33.1 },
    { seed: 54, delta: -157.4 }, { seed: 55, delta: -89.8 },
    { seed: 56, delta: -10.0 }, { seed: 57, delta: -0.9 },
    { seed: 58, delta: -82.4 }, { seed: 59, delta: -4.4 },
    { seed: 60, delta: 15.3 }, { seed: 61, delta: -115.9 },
  ],
  mean: -36.3, se: 16.4, t: -2.22,
  source: { ref: 'gate_uniform_20seed.txt', kind: 'log' },
}

// Stratified depth sampling, seeds 42-45, both nets. Source: gate log.
export const GATE_STRATIFIED: {
  method: string;
  b5: { deltas: SeedDelta[]; mean: number; se: number; t: number };
  b4: { deltas: SeedDelta[]; mean: number; se: number; t: number };
  source: Source;
} = {
  method: 'stratified depth sampling · seeds 42–45 · ~28 override nodes/seed',
  b5: {
    deltas: [
      { seed: 42, delta: -112.1 }, { seed: 43, delta: -7.9 },
      { seed: 44, delta: 8.8 }, { seed: 45, delta: -9.7 },
    ],
    mean: -30.2, se: 27.6, t: -1.10,
  },
  b4: {
    deltas: [
      { seed: 42, delta: -110.9 }, { seed: 43, delta: -8.1 },
      { seed: 44, delta: 8.2 }, { seed: 45, delta: -19.4 },
    ],
    mean: -32.6, se: 26.7, t: -1.22,
  },
  source: { ref: 'gate_b4_vs_b5.log', kind: 'log' },
}

// ── Decision log ────────────────────────────────────────────────────────────

export type Decision = {
  verdict: string
  basis: string
  justification: string
  source: Source
}

export const DECISIONS: Decision[] = [
  {
    verdict: 'Unsafe re-solving kept (no gadget)',
    basis: 'Leduc Phase 1: agent expl 1.365 < blueprint 1.562 (Δ −0.197)',
    justification: 'unsafe re-solving already lowered exploitability; the safe gadget was only to be added if unsafe raised it — it did not.',
    source: { ref: '6caadad', kind: 'commit' },
  },
  {
    verdict: 'LBR metric: Bayes beliefs over uniform',
    basis: 'Leduc: uniform belief floors near equilibrium (exact 0.22 & 0.08 both read ~0.31, order inverts); Bayes r=0.996',
    justification: 'uniform opponent belief cannot separate good from great near equilibrium; reach-weighting restores ordering.',
    source: { ref: '92f41ce', kind: 'commit' },
  },
  {
    verdict: 'River-only search abandoned',
    basis: 'per-river-node Δ = +0.008 ± 0.121 (z=0.07); third independent zero',
    justification: '4-action river decisions are already at the blueprint ceiling; the win had to come from turn + real ranges.',
    source: { ref: '70e8436', kind: 'commit' },
  },
  {
    verdict: '2 raise sizes over 1 (v16b→v16c)',
    basis: 'LBR v1 944±138 (2 sizes) vs 1194±177 (single) @2500 iters',
    justification: 'richer action tree measurably lowered exploitability at equal budget.',
    source: { ref: '91d2d02', kind: 'commit' },
  },
  {
    verdict: '3 raise sizes REJECTED (v16e)',
    basis: 'LBR v2 1116±50 (3 sizes) vs 998±26 (2 sizes), 5 seeds',
    justification: 'a still-richer tree lost — sample starvation at this budget outweighs the finer action grid. @5000-iter retest deferred.',
    source: { ref: 'gate_b4_vs_b5.log', kind: 'log' },
  },
  {
    verdict: 'b5 CFV net chosen over b4',
    basis: 'gate tie (b5 Δ −30.2±27.6, b4 −32.6±26.7); b5 better aggregate MAE',
    justification: 'gates were a statistical tie; b5 wins the tiebreak on more data + lower holdout MAE.',
    source: { ref: 'gate_b4_vs_b5.log', kind: 'log' },
  },
  {
    verdict: 'Depth-limited turn+river search beats blueprint',
    basis: 'uniform 20-seed Δ −36.3±16.4 (t=−2.22); stratified 4-seed ≈ −31',
    justification: 'the gate passes and replicates across two independent sampling methods — the search layer is a real, measured improvement.',
    source: { ref: 'gate_uniform_20seed.txt', kind: 'log' },
  },
]

// ── Metric evolution prose anchors ──────────────────────────────────────────

export const METRIC_ERAS = [
  {
    name: 'Clairvoyant regret',
    problem: 'best-response saw the villain’s exact cards → measured deal variance, not strategy quality. Disagreed with head-to-head results.',
    source: { ref: 'project_exploitability_rewrite', kind: 'log' } as Source,
  },
  {
    name: 'LBR v1',
    problem: 'opponent hand marginalised uniformly. Correct far from equilibrium, but floors near it — cannot separate a 0.22 strategy from a 0.08 one on Leduc.',
    source: { ref: '92f41ce', kind: 'commit' } as Source,
  },
  {
    name: 'LBR v2',
    problem: 'Bayes (reach-weighted) beliefs + stratified depth sampling + per-node samples for correct CRN-paired SE. Ordering restored (r=0.996 vs exact on Leduc).',
    source: { ref: '92f41ce', kind: 'commit' } as Source,
  },
]
