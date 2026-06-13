/**
 * ONNX InferenceSession singleton.
 * Lazy-loads the strategy model once and reuses it across calls.
 * onnxruntime-web runs inference entirely in-browser — no backend needed.
 *
 * Model versioning (Fix 10):
 *   Fetches /models/model_manifest.json first to get the content-hashed
 *   model path (e.g. /models/strategy_net.abc123def456.onnx).
 *   This busts browser and CDN caches automatically on each retrain.
 *   Falls back to NEXT_PUBLIC_MODEL_PATH / '/models/strategy_net.onnx'
 *   if the manifest is absent.
 *
 * Batch inference (Fix 11):
 *   queryStrategyBatch() encodes N states into a single [N, STATE_SIZE]
 *   tensor and runs one session.run — 10-50× faster than N sequential calls.
 *   Used by RangeMatrix to replace 169 individual queryStrategy calls.
 */

import * as ort from 'onnxruntime-web'
import { STATE_SIZE, type EncodeInput, encode } from './encoder'

export type ActionProbs = {
  fold:  number
  call:  number
  raise: number
  allIn: number
}

// Fallback path if manifest not found
const FALLBACK_MODEL_PATH =
  process.env.NEXT_PUBLIC_MODEL_PATH ?? '/models/strategy_net.onnx'

const MANIFEST_PATH = '/models/model_manifest.json'

let _session:      ort.InferenceSession | null = null
let _loading:      Promise<ort.InferenceSession> | null = null
let _queue:        Promise<unknown> = Promise.resolve()
let _resolvedPath: string | null = null

/**
 * Fetch model_manifest.json to get the content-hashed model path.
 * Falls back to FALLBACK_MODEL_PATH if the manifest is absent or malformed.
 */
async function resolveModelPath(): Promise<string> {
  if (_resolvedPath) return _resolvedPath
  try {
    const res = await fetch(MANIFEST_PATH, { cache: 'no-store' })
    if (res.ok) {
      const manifest = await res.json()
      if (typeof manifest.model_path === 'string') {
        _resolvedPath = manifest.model_path
        console.log(`[session] model version: ${manifest.hash ?? 'unknown'} (${manifest.iterations ?? '?'} iter)`)
        return manifest.model_path
      }
    }
  } catch {
    // manifest not found — use fallback
  }
  _resolvedPath = FALLBACK_MODEL_PATH
  return FALLBACK_MODEL_PATH
}

export async function getSession(): Promise<ort.InferenceSession> {
  if (_session) return _session
  if (_loading)  return _loading

  _loading = resolveModelPath().then(path =>
    ort.InferenceSession.create(path, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  ).then(s => {
    _session = s
    _loading = null
    return s
  })

  return _loading
}

// ── Single-state inference ────────────────────────────────────────────────────

/**
 * Run strategy inference for a single game state.
 * Serialised through _queue — onnxruntime-web does not support concurrent run().
 */
export async function queryStrategy(input: EncodeInput): Promise<ActionProbs> {
  const result = await (_queue = _queue.then(() => _inferSingle(input)))
  return result as ActionProbs
}

async function _inferSingle(input: EncodeInput): Promise<ActionProbs> {
  const session    = await getSession()
  const stateVec   = encode(input)
  const stateTensor = new ort.Tensor('float32', stateVec, [1, STATE_SIZE])
  const maskTensor  = new ort.Tensor('float32', new Float32Array([1, 1, 1, 1]), [1, 4])
  const output      = await session.run({ state: stateTensor, action_mask: maskTensor })
  return _toActionProbs(output[session.outputNames[0]].data as Float32Array, 0)
}

/**
 * Raw probability query — accepts an already-encoded state vector and the
 * number of legal actions, returns the action-mask-respecting probability
 * vector as a Float32Array of length `numLegalActions`.
 *
 * Used by SafeSubgameSolver (web/src/lib/cfr/subgame-solver.ts) which encodes
 * its own state vectors via the SubgameGame wrapper. Keeps the solver
 * decoupled from queryStrategy's EncodeInput shape.
 */
export async function queryStrategyRaw(
  state: Float32Array,
  numLegalActions: number,
): Promise<Float32Array> {
  const session = await getSession()
  const stateTensor = new ort.Tensor('float32', state, [1, STATE_SIZE])
  const mask = new Float32Array(4)
  for (let i = 0; i < Math.min(numLegalActions, 4); i++) mask[i] = 1.0
  const maskTensor  = new ort.Tensor('float32', mask, [1, 4])
  const output      = await session.run({ state: stateTensor, action_mask: maskTensor })
  const logits      = output[session.outputNames[0]].data as Float32Array
  const out = new Float32Array(numLegalActions)
  let total = 0
  for (let i = 0; i < numLegalActions; i++) {
    const v = Math.max(logits[i] ?? 0, 0)
    out[i] = v
    total += v
  }
  if (total > 1e-7) {
    for (let i = 0; i < numLegalActions; i++) out[i] /= total
  } else {
    out.fill(1 / numLegalActions)
  }
  return out
}

// ── Batch inference ───────────────────────────────────────────────────────────

/**
 * Run strategy inference for N game states in a single session.run call.
 *
 * Compared to N sequential queryStrategy calls this is 10-50× faster because:
 *   1. Only one JS↔WASM boundary crossing
 *   2. ONNX Runtime can parallelise matrix ops across the batch dimension
 *   3. No _queue serialisation overhead between items
 *
 * Used by RangeMatrix to process all 169 canonical hands at once.
 *
 * @param inputs  Array of game states to encode and query
 * @returns       Array of ActionProbs in the same order as inputs
 */
export async function queryStrategyBatch(inputs: EncodeInput[]): Promise<ActionProbs[]> {
  if (inputs.length === 0) return []

  const result = await (_queue = _queue.then(() => _inferBatch(inputs)))
  return result as ActionProbs[]
}

async function _inferBatch(inputs: EncodeInput[]): Promise<ActionProbs[]> {
  const session = await getSession()
  const n       = inputs.length

  // Build flat [N * STATE_SIZE] Float32Array
  const stateData = new Float32Array(n * STATE_SIZE)
  for (let i = 0; i < n; i++) {
    stateData.set(encode(inputs[i]), i * STATE_SIZE)
  }

  // All actions legal — uniform mask
  const maskData = new Float32Array(n * 4).fill(1.0)

  const stateTensor = new ort.Tensor('float32', stateData, [n, STATE_SIZE])
  const maskTensor  = new ort.Tensor('float32', maskData,  [n, 4])
  const output      = await session.run({ state: stateTensor, action_mask: maskTensor })
  const logits      = output[session.outputNames[0]].data as Float32Array

  // Split flat output [N * 4] → N ActionProbs
  return Array.from({ length: n }, (_, i) => _toActionProbs(logits, i))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _toActionProbs(logits: Float32Array, batchIdx: number): ActionProbs {
  const offset = batchIdx * 4
  const raw    = [logits[offset], logits[offset+1], logits[offset+2], logits[offset+3]]
  const pos    = raw.map(v => Math.max(v, 0))
  const total  = pos.reduce((a, b) => a + b, 0)
  const probs  = total > 1e-7 ? pos.map(v => v / total) : [0.25, 0.25, 0.25, 0.25]
  return { fold: probs[0], call: probs[1], raise: probs[2], allIn: probs[3] }
}
