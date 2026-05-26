'use client'

import { useState, useCallback, useTransition } from 'react'
import { RANKS, SUITS, makeCard, cardLabel } from '@/lib/onnx/encoder'
import { queryStrategy, type ActionProbs } from '@/lib/onnx/session'
import { STREET_NAMES, ACTION_LABELS, type Street } from '@/types'

const ACTION_COLORS: Record<keyof ActionProbs, string> = {
  fold:  '#888780',
  call:  '#1D9E75',
  raise: '#D85A30',
  allIn: '#7F77DD',
}

// ── Card picker ───────────────────────────────────────────────────────────────

interface CardPickerProps {
  label: string
  value: number | null
  excluded: number[]
  onChange: (card: number) => void
}

function CardPicker({ label, value, excluded, onChange }: CardPickerProps) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <p className="text-xs text-secondary mb-1">{label}</p>
      <button
        onClick={() => setOpen(v => !v)}
        className="card-btn"
        style={{ minWidth: 52 }}
      >
        {value !== null ? cardLabel(value) : '—'}
      </button>
      {open && (
        <div className="card-grid">
          {RANKS.slice().reverse().map(r => {
            const ri = RANKS.indexOf(r)
            return SUITS.map((s, si) => {
              const card = makeCard(ri, si)
              const disabled = excluded.includes(card)
              return (
                <button
                  key={card}
                  disabled={disabled}
                  onClick={() => { onChange(card); setOpen(false) }}
                  className={`card-cell ${disabled ? 'opacity-30' : ''}`}
                  title={cardLabel(card)}
                >
                  <span style={{ color: si === 1 || si === 2 ? '#c0392b' : 'inherit' }}>
                    {r}{s}
                  </span>
                </button>
              )
            })
          })}
        </div>
      )}
    </div>
  )
}

// ── Probability bar ───────────────────────────────────────────────────────────

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span style={{ color, fontWeight: 500 }}>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StrategyExplorer() {
  const [hole0, setHole0] = useState<number | null>(null)
  const [hole1, setHole1] = useState<number | null>(null)
  const [board, setBoard] = useState<(number | null)[]>([null, null, null, null, null])
  const [street, setStreet] = useState<Street>(0)
  const [pot, setPot]       = useState(6)
  const [toCall, setToCall] = useState(2)
  const [myStack, setMyStack] = useState(198)
  const [probs, setProbs]   = useState<ActionProbs | null>(null)
  const [error, setError]   = useState<string | null>(null)
  const [pending, startTransition] = useTransition()

  const usedCards = [hole0, hole1, ...board].filter((c): c is number => c !== null)

  const setBoard0 = useCallback((i: number) => (c: number) => {
    setBoard(prev => { const next = [...prev]; next[i] = c; return next })
  }, [])

  const visibleBoardCount = [0, 3, 4, 5][street]

  const handleQuery = () => {
    if (hole0 === null || hole1 === null) {
      setError('Valitse molemmat hole cardit')
      return
    }
    setError(null)
    startTransition(async () => {
      try {
        const result = await queryStrategy({
          holeCards: [hole0, hole1],
          boardCards: board.slice(0, visibleBoardCount).filter((c): c is number => c !== null),
          street,
          pot,
          toCall,
          myStack,
          oppStack: STARTING_STACK - (pot / 2),
          actionHistory: [],
        })
        setProbs(result)
      } catch (e) {
        setError('Mallin lataus epäonnistui — tarkista /public/models/strategy_net.onnx')
        console.error(e)
      }
    })
  }

  return (
    <div className="explorer">
      <h1 className="title">CFR Strategy Explorer</h1>

      {/* Street selector */}
      <div className="seg-row">
        {([0, 1, 2, 3] as Street[]).map(s => (
          <button
            key={s}
            onClick={() => setStreet(s)}
            className={`seg-btn ${street === s ? 'active' : ''}`}
          >
            {STREET_NAMES[s]}
          </button>
        ))}
      </div>

      {/* Card selection */}
      <div className="section">
        <p className="section-label">Hole cards</p>
        <div className="flex gap-3 flex-wrap">
          <CardPicker label="Card 1" value={hole0} excluded={[hole1, ...board].filter((c): c is number => c !== null)} onChange={setHole0} />
          <CardPicker label="Card 2" value={hole1} excluded={[hole0, ...board].filter((c): c is number => c !== null)} onChange={setHole1} />
        </div>
      </div>

      {visibleBoardCount > 0 && (
        <div className="section">
          <p className="section-label">Board ({STREET_NAMES[street].toLowerCase()})</p>
          <div className="flex gap-3 flex-wrap">
            {Array.from({ length: visibleBoardCount }).map((_, i) => (
              <CardPicker
                key={i}
                label={`Board ${i + 1}`}
                value={board[i]}
                excluded={[hole0, hole1, ...board.filter((_, j) => j !== i)].filter((c): c is number => c !== null)}
                onChange={setBoard0(i)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Betting inputs */}
      <div className="section">
        <p className="section-label">Betting state</p>
        <div className="bet-grid">
          {([
            ['Pot', pot, setPot],
            ['To call', toCall, setToCall],
            ['My stack', myStack, setMyStack],
          ] as [string, number, (v: number) => void][]).map(([label, val, set]) => (
            <label key={label} className="bet-field">
              <span className="text-xs text-secondary">{label}</span>
              <input
                type="number"
                value={val}
                min={0}
                step={1}
                onChange={e => set(Number(e.target.value))}
                className="bet-input"
              />
            </label>
          ))}
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      <button onClick={handleQuery} disabled={pending} className="query-btn">
        {pending ? 'Lasketaan…' : 'Hae strategia ↗'}
      </button>

      {/* Results */}
      {probs && (
        <div className="results">
          <p className="section-label">Action probabilities</p>
          {(Object.entries(probs) as [keyof ActionProbs, number][]).map(([action, prob]) => (
            <ProbBar
              key={action}
              label={ACTION_LABELS[action === 'fold' ? 0 : action === 'call' ? 1 : action === 'raise' ? 2 : 3]}
              value={prob}
              color={ACTION_COLORS[action]}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const STARTING_STACK = 200
