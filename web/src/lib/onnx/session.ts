/**
 * ONNX InferenceSession singleton.
 * Lazy-loads the strategy model once and reuses it across calls.
 * onnxruntime-web runs inference entirely in-browser — no backend needed.
 */

import * as ort from 'onnxruntime-web'
import { STATE_SIZE, type EncodeInput, encode } from './encoder'

export type ActionProbs = {
  fold:  number
  call:  number
  raise: number
  allIn: number
}

// Path relative to /public — set NEXT_PUBLIC_MODEL_PATH in .env.local
// to override (e.g. Supabase Storage signed URL)
const MODEL_PATH =
  process.env.NEXT_PUBLIC_MODEL_PATH ?? '/models/strategy_net.onnx'

let _session: ort.InferenceSession | null = null
let _loading: Promise<ort.InferenceSession> | null = null

export async function getSession(): Promise<ort.InferenceSession> {
  if (_session) return _session
  if (_loading)  return _loading

  _loading = ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  }).then(s => {
    _session = s
    _loading = null
    return s
  })

  return _loading
}

/**
 * Run strategy inference for a given game state.
 * Returns softmax-normalised action probabilities after regret matching.
 */
export async function queryStrategy(input: EncodeInput): Promise<ActionProbs> {
  const session  = await getSession()
  const stateVec = encode(input)
  console.log('state nonzero:', Array.from(stateVec).filter(v => v !== 0))

  const stateTensor = new ort.Tensor('float32', stateVec, [1, STATE_SIZE])
  const maskData    = new Float32Array([1, 1, 1, 1])
  const maskTensor  = new ort.Tensor('float32', maskData, [1, 4])
  const feeds       = {
    state:       stateTensor,
    action_mask: maskTensor,
  }
  const result = await session.run(feeds)

  const logits = result[session.outputNames[0]].data as Float32Array
  console.log('raw logits:', Array.from(logits))

  // Regret matching: clamp negatives, normalise — mirrors TorchModel::forward_tensor
  const pos   = Array.from(logits.slice(0, 4)).map(v => Math.max(v, 0))
  const total = pos.reduce((a, b) => a + b, 0)
  const probs = total > 1e-7
    ? pos.map(v => v / total)
    : [0.25, 0.25, 0.25, 0.25]

  return {
    fold:  probs[0],
    call:  probs[1],
    raise: probs[2],
    allIn: probs[3],
  }
}
