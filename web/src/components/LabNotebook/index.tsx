'use client'

import {
  BLUEPRINTS, CFV_NETS, GATE_UNIFORM, GATE_STRATIFIED,
  DECISIONS, METRIC_ERAS, type Source, type SeedDelta,
} from '@/data/lab'

// ── Source citation chip ────────────────────────────────────────────────────

function Cite({ s }: { s: Source }) {
  const label = s.kind === 'commit' ? s.ref.slice(0, 7) : s.ref
  const title = s.kind === 'commit'
    ? `commit ${s.ref}`
    : `experiment log / note: ${s.ref}`
  return <code className="cite" title={title}>{s.kind === 'commit' ? '⎇ ' : '▤ '}{label}</code>
}

// ── Horizontal bar chart (LBR by version, grouped by comparability) ─────────

function LbrChart() {
  const max = Math.max(...BLUEPRINTS.map(b => b.lbr + (b.stderr ?? 0))) * 1.05
  const barH = 30, gap = 14, padL = 210, padR = 60, w = 720
  const h = BLUEPRINTS.length * (barH + gap) + 20
  const x = (v: number) => padL + (v / max) * (w - padL - padR)
  const color = (s: string) =>
    s === 'production' ? 'var(--green)'
      : s === 'rejected' ? 'var(--red)' : 'var(--text3)'

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img"
      aria-label="LBR proxy by blueprint version">
      {[250, 500, 750, 1000, 1250].filter(g => g < max).map(g => (
        <g key={g}>
          <line x1={x(g)} y1={10} x2={x(g)} y2={h - 10}
            stroke="var(--border)" strokeDasharray="2 3" />
          <text x={x(g)} y={h - 2} className="axis" textAnchor="middle">{g}</text>
        </g>
      ))}
      {BLUEPRINTS.map((b, i) => {
        const y = 14 + i * (barH + gap)
        return (
          <g key={i}>
            <text x={padL - 8} y={y + barH / 2 - 3} className="bar-label"
              textAnchor="end">{b.version}</text>
            <text x={padL - 8} y={y + barH / 2 + 9} className="bar-sub"
              textAnchor="end">{b.tree}</text>
            <rect x={padL} y={y} width={x(b.lbr) - padL} height={barH}
              fill={color(b.status)} opacity={b.status === 'superseded' ? 0.45 : 0.85} />
            {b.stderr && (
              <line x1={x(b.lbr - b.stderr)} y1={y + barH / 2}
                x2={x(b.lbr + b.stderr)} y2={y + barH / 2}
                stroke="var(--text)" strokeWidth={1.5} />
            )}
            <text x={x(b.lbr) + 8} y={y + barH / 2 + 4} className="bar-val">
              {b.lbr}{b.stderr ? ` ±${b.stderr}` : ''}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ── CFV MAE line chart (base vs on-policy per generation) ───────────────────

function MaeChart() {
  const w = 620, h = 260, padL = 56, padB = 40, padT = 20, padR = 90
  const gens = CFV_NETS
  const maxMae = 0.15
  const x = (i: number) => padL + (i / (gens.length - 1)) * (w - padL - padR)
  const y = (m: number) => padT + (1 - m / maxMae) * (h - padT - padB)
  const path = (key: 'baseMae' | 'onPolicyMae') =>
    gens.map((g, i) => `${i ? 'L' : 'M'}${x(i)},${y(g[key])}`).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img"
      aria-label="CFV holdout MAE by network generation">
      {[0, 0.05, 0.10, 0.15].map(g => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={w - padR} y2={y(g)}
            stroke="var(--border)" strokeDasharray="2 3" />
          <text x={padL - 8} y={y(g) + 3} className="axis" textAnchor="end">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {gens.map((g, i) => (
        <text key={i} x={x(i)} y={h - 22} className="axis" textAnchor="middle">
          {g.gen}
        </text>
      ))}
      <text x={x(gens.length - 1) + 10} y={y(gens[gens.length - 1].onPolicyMae) + 3}
        className="series-label" fill="var(--red)">on-policy</text>
      <text x={x(gens.length - 1) + 10} y={y(gens[gens.length - 1].baseMae) + 3}
        className="series-label" fill="var(--green)">base</text>
      <path d={path('onPolicyMae')} fill="none" stroke="var(--red)" strokeWidth={2} />
      <path d={path('baseMae')} fill="none" stroke="var(--green)" strokeWidth={2} />
      {gens.map((g, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(g.onPolicyMae)} r={3.5} fill="var(--red)" />
          <circle cx={x(i)} cy={y(g.baseMae)} r={3.5} fill="var(--green)" />
        </g>
      ))}
      <text x={padL} y={12} className="axis">MAE (pot units) · lower is better</text>
    </svg>
  )
}

// ── Seed-variance strip: per-seed Δ with the pooled CI band ─────────────────

function SeedStrip({ deltas, mean, se, label, height = 120 }: {
  deltas: SeedDelta[]; mean: number; se: number; label: string; height?: number
}) {
  const w = 720, padL = 20, padR = 20, padT = 24, padB = 24
  const vals = deltas.map(d => d.delta)
  const lo = Math.min(-220, ...vals), hi = Math.max(140, ...vals)
  const x = (i: number) => padL + (i / (deltas.length - 1)) * (w - padL - padR)
  const y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (height - padT - padB)
  const ci = 1.96 * se

  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="chart" role="img" aria-label={label}>
      {/* zero line */}
      <line x1={padL} y1={y(0)} x2={w - padR} y2={y(0)}
        stroke="var(--text3)" strokeWidth={1} />
      <text x={w - padR} y={y(0) - 4} className="axis" textAnchor="end">Δ = 0 (no gain)</text>
      {/* pooled mean ± 95% CI band */}
      <rect x={padL} y={y(mean + ci)} width={w - padL - padR}
        height={Math.abs(y(mean - ci) - y(mean + ci))}
        fill="var(--green)" opacity={0.12} />
      <line x1={padL} y1={y(mean)} x2={w - padR} y2={y(mean)}
        stroke="var(--green)" strokeWidth={1.5} strokeDasharray="5 3" />
      <text x={padL + 4} y={y(mean) - 5} className="series-label" fill="var(--green)">
        mean {mean.toFixed(1)}
      </text>
      {/* per-seed points */}
      {deltas.map((d, i) => (
        <g key={d.seed}>
          <line x1={x(i)} y1={y(0)} x2={x(i)} y2={y(d.delta)}
            stroke={d.delta < 0 ? 'var(--green)' : 'var(--red)'} strokeWidth={1} opacity={0.5} />
          <circle cx={x(i)} cy={y(d.delta)} r={3}
            fill={d.delta < 0 ? 'var(--green)' : 'var(--red)'} />
        </g>
      ))}
    </svg>
  )
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function LabNotebook() {
  const negU = GATE_UNIFORM.deltas.filter(d => d.delta < 0).length
  return (
    <div className="lab">
      <div className="header">
        <h1 className="title">Lab Notebook</h1>
        <p className="subtitle">
          Metric evolution, seed variance, and experimental verdicts for the
          depth-limited search layer. Every number is traced to a commit or a
          committed log (hover the <code className="cite">⎇ ref</code> chips).
        </p>
      </div>

      {/* ── Methodology note ── */}
      <section className="lab-section">
        <p className="section-label">Why gates, seeds, and CIs</p>
        <div className="prose">
          <p>
            This project has a documented history of <strong>confident
            corrections invalidated by data</strong> — a distilled network that
            regressed head-to-head, richer card abstractions that only looked
            worse because of sample starvation, three independent “zero” results
            for river re-solving. So every mechanism is gated: it must beat the
            baseline on a measured, seed-replicated metric before it is kept.
          </p>
          <p>
            The load-bearing lesson is <strong>seed variance</strong>. The same
            strategy measures anywhere from 544 to 1308 mbb/decision across
            seeds. A single-seed number is one draw, not a level — so verdicts
            use pooled means with CIs, and paired (CRN) deltas for A/B, never a
            lone seed.
          </p>
        </div>
      </section>

      {/* ── Metric evolution ── */}
      <section className="lab-section">
        <p className="section-label">Metric evolution — LBR proxy by blueprint</p>
        <p className="prose">
          <span className="warn">Comparability caveat:</span> LBR is a
          lower-bound proxy, not true exploitability, and it is <em>not</em>
          comparable across action-set sizes (a wider tree gives the exploiter
          a higher max) or across metric generations. Bars are shaded by role;
          compare only within a tree/metric group. Whiskers are ±1 stderr.
        </p>
        <LbrChart />
        <div className="legend-row">
          <span><i className="sw" style={{ background: 'var(--green)' }} /> production</span>
          <span><i className="sw" style={{ background: 'var(--text3)' }} /> superseded</span>
          <span><i className="sw" style={{ background: 'var(--red)' }} /> rejected (a result, not a gap)</span>
        </div>

        <p className="section-label" style={{ marginTop: '2rem' }}>
          Metric generations
        </p>
        <div className="era-grid">
          {METRIC_ERAS.map(e => (
            <div key={e.name} className="era-card">
              <div className="era-name">{e.name} <Cite s={e.source} /></div>
              <p>{e.problem}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CFV net MAE ── */}
      <section className="lab-section">
        <p className="section-label">CFV network generations — river-value holdout MAE</p>
        <p className="prose">
          The ReBeL on-policy loop: the value net must be trained on the PBSs
          the depth-limited solver actually queries, not just blueprint
          self-play. The on-policy MAE (red) started 2.9× the base MAE and
          closed with each retraining generation.
        </p>
        <MaeChart />
        <table className="lab-table">
          <thead><tr>
            <th>net</th><th>data</th><th>base MAE</th><th>on-policy MAE</th>
            <th>note</th><th>src</th>
          </tr></thead>
          <tbody>
            {CFV_NETS.map(c => (
              <tr key={c.gen}>
                <td><strong>{c.gen}</strong></td><td>{c.data}</td>
                <td>{c.baseMae.toFixed(3)}</td><td>{c.onPolicyMae.toFixed(3)}</td>
                <td className="note">{c.note}</td><td><Cite s={c.source} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* ── Seed variance (centerpiece) ── */}
      <section className="lab-section">
        <p className="section-label">Seed variance — the gate, replicated</p>
        <p className="prose">
          The search layer’s gate: whole-strategy LBR of the depth-limited
          agent minus the v16c blueprint, CRN-paired per seed. Below Δ = 0 means
          the search agent is less exploitable. The result is <strong>replicated
          across two independent depth-sampling schemes</strong> — the
          discipline the whole page is about.
        </p>

        <div className="gate-block">
          <div className="gate-head">
            <span>{GATE_UNIFORM.method}</span>
            <span className="gate-stat">
              Δ = {GATE_UNIFORM.mean} ± {GATE_UNIFORM.se} · t = {GATE_UNIFORM.t} ·
              {' '}{negU}/{GATE_UNIFORM.deltas.length} negative <Cite s={GATE_UNIFORM.source} />
            </span>
          </div>
          <SeedStrip deltas={GATE_UNIFORM.deltas} mean={GATE_UNIFORM.mean}
            se={GATE_UNIFORM.se} label="uniform gate deltas" height={150} />
        </div>

        <div className="gate-grid">
          {(['b5', 'b4'] as const).map(net => {
            const g = GATE_STRATIFIED[net]
            return (
              <div key={net} className="gate-block">
                <div className="gate-head">
                  <span>stratified · {net} net</span>
                  <span className="gate-stat">
                    Δ = {g.mean} ± {g.se} · t = {g.t}
                  </span>
                </div>
                <SeedStrip deltas={g.deltas} mean={g.mean} se={g.se}
                  label={`stratified ${net} deltas`} />
              </div>
            )
          })}
        </div>
        <p className="prose small">
          Stratified sampling ({GATE_STRATIFIED.method}) lifts turn/river nodes
          from ~8 to ~28 per seed, tightening the paired SE. Both nets land at
          Δ ≈ −31, consistent with the uniform 20-seed −36.3 ± 16.4.
          {' '}<Cite s={GATE_STRATIFIED.source} />
        </p>
      </section>

      {/* ── Decision log ── */}
      <section className="lab-section">
        <p className="section-label">Decision log</p>
        <table className="lab-table decisions">
          <thead><tr>
            <th>verdict</th><th>measured basis</th><th>why</th><th>src</th>
          </tr></thead>
          <tbody>
            {DECISIONS.map((d, i) => (
              <tr key={i}>
                <td><strong>{d.verdict}</strong></td>
                <td className="mono-cell">{d.basis}</td>
                <td className="note">{d.justification}</td>
                <td><Cite s={d.source} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <footer className="lab-footer">
        Static page · numbers compiled at build time from{' '}
        <code className="cite">web/src/data/lab.ts</code>, each carrying a
        source reference. No runtime fetches, no server.
      </footer>
    </div>
  )
}
