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
import { STATE_SIZE, type EncodeInput, encode, setAdvisorCache } from './encoder'
import { loadCache } from './cache'

export type ActionProbs = {
  fold:     number
  call:     number
  raise50:  number   // RAISE_0, 50% pot (slot 2)
  raise100: number   // RAISE_1, 100% pot (slot 3)
  allIn:    number   // slot 5
}

// v16c 2-size tree: 6 network output slots; RAISE_2 (slot 4) is untrained
// padding and must be masked out. Overridable via manifest.legal_slots.
export const ACTION_SIZE = 6
let _legalSlots: number[] = [1, 1, 1, 1, 0, 1]

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
        console.log(`[session] model version: ${manifest.hash ?? 'unknown'} (${manifest.iterations ?? '?'} iter, state_size=${manifest.state_size ?? '?'})`)
        if (Array.isArray(manifest.legal_slots) && manifest.legal_slots.length === ACTION_SIZE) {
          _legalSlots = manifest.legal_slots.map((x: number) => (x ? 1 : 0))
        }
        if (manifest.action_size !== undefined && manifest.action_size !== ACTION_SIZE) {
          console.warn(`[session] manifest action_size=${manifest.action_size} != build ACTION_SIZE=${ACTION_SIZE} — strategy display will be WRONG`)
        }
        // If model uses the cache-augmented 49-dim state, fetch the
        // advisor cache binary in parallel. Cache miss is non-fatal:
        // the encoder leaves advisor dims at zero, network handles it.
        if (manifest.state_size === 49 && typeof manifest.cache_path === 'string') {
          loadCache(manifest.cache_path)
            .then(c => {
              setAdvisorCache(c)
              console.log(`[session] advisor cache loaded: ${c.nEntries} entries`)
            })
            .catch(err => {
              console.warn(`[session] advisor cache load failed: ${err.message}`)
            })
        }
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
  const maskTensor  = new ort.Tensor('float32', new Float32Array(_legalSlots), [1, ACTION_SIZE])
  const output      = await session.run({ state: stateTensor, action_mask: maskTensor })
  return _toActionProbs(output[session.outputNames[0]].data as Float32Array, 0)
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

  // Legal-slot mask per row (RAISE_2 padding masked out in the 2-size tree)
  const maskData = new Float32Array(n * ACTION_SIZE)
  for (let i = 0; i < n; i++) maskData.set(_legalSlots, i * ACTION_SIZE)

  const stateTensor = new ort.Tensor('float32', stateData, [n, STATE_SIZE])
  const maskTensor  = new ort.Tensor('float32', maskData,  [n, ACTION_SIZE])
  const output      = await session.run({ state: stateTensor, action_mask: maskTensor })
  const logits      = output[session.outputNames[0]].data as Float32Array

  // Split flat output [N * 4] → N ActionProbs
  return Array.from({ length: n }, (_, i) => _toActionProbs(logits, i))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _toActionProbs(logits: Float32Array, batchIdx: number): ActionProbs {
  const offset = batchIdx * ACTION_SIZE
  // Slots: 0=fold/check, 1=call, 2=raise50, 3=raise100, 4=RAISE_2 (masked), 5=all-in
  const raw = [logits[offset], logits[offset+1], logits[offset+2],
               logits[offset+3], logits[offset+5]]
  const pos   = raw.map(v => Math.max(v, 0))
  const total = pos.reduce((a, b) => a + b, 0)
  const probs = total > 1e-7 ? pos.map(v => v / total) : [0.2, 0.2, 0.2, 0.2, 0.2]
  return { fold: probs[0], call: probs[1], raise50: probs[2],
           raise100: probs[3], allIn: probs[4] }
}
