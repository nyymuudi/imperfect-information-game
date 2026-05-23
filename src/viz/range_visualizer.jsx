import { useState, useMemo } from "react";

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];

// Pre-computed strategy data from CFR solver (8 buckets, 1000 iterations)
const BUCKET_MAP = {
  // Bucket 0: eq 0.32-0.38 (trash)
  "32o":0,"32s":0,"42o":0,"42s":0,"43o":0,"43s":0,"52o":0,"52s":0,"53o":0,"53s":0,
  "62o":0,"63o":0,"72o":0,"73o":0,"74o":0,"82o":0,"83o":0,"84o":0,"92o":0,"93o":0,
  "T2o":0,"T3o":0,"T4o":0,"J2o":0,
  // Bucket 1: eq 0.38-0.45 (weak)
  "54o":1,"54s":1,"63s":1,"64o":1,"64s":1,"65o":1,"65s":1,"73s":1,"74s":1,"75o":1,
  "75s":1,"76o":1,"84s":1,"85o":1,"85s":1,"86o":1,"94o":1,"94s":1,"95o":1,"95s":1,
  "96o":1,"T2s":1,"T5o":1,"J3o":1,"J4o":1,"Q2o":1,"K2o":1,
  // Bucket 2: eq 0.45-0.51 (medium-low)
  "76s":2,"86s":2,"87o":2,"87s":2,"93s":2,"96s":2,"97o":2,"97s":2,"98o":2,"98s":2,
  "T3s":2,"T4s":2,"T5s":2,"T6o":2,"T6s":2,"T7o":2,"T7s":2,"T8o":2,"J2s":2,"J3s":2,
  "J4s":2,"J5o":2,"J5s":2,"J6o":2,"J6s":2,"J7o":2,"Q2s":2,"Q3o":2,"Q3s":2,"Q4o":2,
  "Q5o":2,"K2s":2,"K3o":2,"K3s":2,"K4o":2,
  // Bucket 3: eq 0.51-0.57 (medium)
  "22":3,"33":3,"44":3,"A2o":3,"A2s":3,"A3o":3,"A3s":3,"A4o":3,"A4s":3,"A5o":3,
  "T8s":3,"T9o":3,"T9s":3,"J7s":3,"J8o":3,"J8s":3,"J9o":3,"J9s":3,"JTo":3,"JTs":3,
  "Q4s":3,"Q5s":3,"Q6o":3,"Q6s":3,"Q7o":3,"Q7s":3,"Q8o":3,"Q8s":3,"Q9o":3,"Q9s":3,
  "QTo":3,"QTs":3,"QJo":3,"QJs":3,"K4s":3,"K5o":3,"K5s":3,"K6o":3,"K6s":3,"K7o":3,
  "K8o":3,"K9o":3,"KTo":3,"KJo":3,"KQo":3,
  "A5s":3,"A6o":3,
  // Bucket 4: eq 0.58-0.64 (good)
  "55":4,"66":4,"A6s":4,"A7o":4,"A7s":4,"A8o":4,"A8s":4,"A9o":4,"A9s":4,"ATo":4,
  "ATs":4,"AJo":4,"K7s":4,"K8s":4,"K9s":4,"KTs":4,"KJs":4,"KQs":4,"AJs":4,
  "AQo":4,"AKo":4,
  // Bucket 5: eq 0.65-0.70 (strong)
  "77":5,"88":5,"AQs":5,"AKs":5,
  // Bucket 6: eq 0.73-0.76 (very strong)
  "99":6,"TT":6,
  // Bucket 7: eq 0.80-0.84 (premium)
  "JJ":7,"QQ":7,"KK":7,"AA":7,
};

const SB_OPEN = {
  0: { f: 0.80, c: 0.00, r: 0.20, a: 0.00 },
  1: { f: 0.71, c: 0.00, r: 0.29, a: 0.00 },
  2: { f: 0.00, c: 1.00, r: 0.00, a: 0.00 },
  3: { f: 0.00, c: 0.99, r: 0.01, a: 0.00 },
  4: { f: 0.00, c: 0.09, r: 0.91, a: 0.00 },
  5: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
  6: { f: 0.00, c: 0.62, r: 0.38, a: 0.00 },
  7: { f: 0.00, c: 0.45, r: 0.54, a: 0.00 },
};

const BB_VS_RAISE = {
  0: { f: 1.00, c: 0.00, r: 0.00, a: 0.00 },
  1: { f: 1.00, c: 0.00, r: 0.00, a: 0.00 },
  2: { f: 0.99, c: 0.00, r: 0.01, a: 0.00 },
  3: { f: 0.31, c: 0.63, r: 0.07, a: 0.00 },
  4: { f: 0.00, c: 1.00, r: 0.00, a: 0.00 },
  5: { f: 0.00, c: 0.73, r: 0.27, a: 0.00 },
  6: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
  7: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
};

const BB_VS_LIMP = {
  0: { f: 0.00, c: 0.70, r: 0.30, a: 0.00 },
  1: { f: 0.00, c: 0.74, r: 0.26, a: 0.00 },
  2: { f: 0.00, c: 1.00, r: 0.00, a: 0.00 },
  3: { f: 0.00, c: 0.99, r: 0.01, a: 0.00 },
  4: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
  5: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
  6: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
  7: { f: 0.00, c: 0.00, r: 1.00, a: 0.00 },
};

const SCENARIOS = {
  "SB Opening": SB_OPEN,
  "BB vs Raise": BB_VS_RAISE,
  "BB vs Limp": BB_VS_LIMP,
};

const ACTION_COLORS = {
  f: { color: "#6b7280", label: "Fold" },
  c: { color: "#3b82f6", label: "Call" },
  r: { color: "#ef4444", label: "Raise" },
  a: { color: "#f59e0b", label: "All-in" },
};

function handClass(row, col) {
  const r1 = RANKS[row], r2 = RANKS[col];
  if (row === col) return r1 + r2;
  if (row < col) return r1 + r2 + "s";
  return r2 + r1 + "o";
}

function getBlendedColor(strat) {
  const actions = Object.entries(strat).filter(([_, v]) => v > 0.01);
  if (actions.length === 0) return "#1f2937";
  if (actions.length === 1) return ACTION_COLORS[actions[0][0]].color;
  
  const colors = {
    f: [107, 114, 128],
    c: [59, 130, 246],
    r: [239, 68, 68],
    a: [245, 158, 11],
  };
  
  let rgb = [0, 0, 0];
  for (const [action, weight] of Object.entries(strat)) {
    const c = colors[action];
    rgb[0] += c[0] * weight;
    rgb[1] += c[1] * weight;
    rgb[2] += c[2] * weight;
  }
  return `rgb(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])})`;
}

function Cell({ row, col, scenario, onHover }) {
  const hc = handClass(row, col);
  const bucket = BUCKET_MAP[hc];
  const strat = bucket !== undefined ? scenario[bucket] : null;
  const bg = strat ? getBlendedColor(strat) : "#111827";
  const isPair = row === col;
  const isSuited = row < col;

  return (
    <div
      onMouseEnter={() => onHover({ hand: hc, bucket, strat })}
      onMouseLeave={() => onHover(null)}
      style={{
        width: "100%",
        aspectRatio: "1",
        backgroundColor: bg,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "10px",
        fontWeight: isPair ? "800" : "500",
        color: "#fff",
        opacity: strat ? 1 : 0.3,
        borderRadius: "2px",
        cursor: "pointer",
        transition: "transform 0.1s",
        border: isPair ? "1px solid rgba(255,255,255,0.3)" : isSuited ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(255,255,255,0.05)",
        textShadow: "0 1px 2px rgba(0,0,0,0.8)",
        letterSpacing: "-0.5px",
      }}
    >
      {hc}
    </div>
  );
}

function ActionBar({ strat }) {
  if (!strat) return null;
  return (
    <div style={{ display: "flex", height: "8px", borderRadius: "4px", overflow: "hidden", width: "100%" }}>
      {Object.entries(strat)
        .filter(([_, v]) => v > 0.005)
        .map(([action, weight]) => (
          <div
            key={action}
            style={{
              width: `${weight * 100}%`,
              backgroundColor: ACTION_COLORS[action].color,
              transition: "width 0.3s",
            }}
          />
        ))}
    </div>
  );
}

export default function RangeVisualizer() {
  const [activeScenario, setActiveScenario] = useState("SB Opening");
  const [hovered, setHovered] = useState(null);
  const scenario = SCENARIOS[activeScenario];

  const stats = useMemo(() => {
    let raise = 0, call = 0, fold = 0, total = 0;
    for (let r = 0; r < 13; r++) {
      for (let c = 0; c < 13; c++) {
        const hc = handClass(r, c);
        const b = BUCKET_MAP[hc];
        if (b === undefined) continue;
        const s = scenario[b];
        const combos = r === c ? 6 : r < c ? 4 : 12;
        raise += (s.r + s.a) * combos;
        call += s.c * combos;
        fold += s.f * combos;
        total += combos;
      }
    }
    return {
      raise: ((raise / total) * 100).toFixed(1),
      call: ((call / total) * 100).toFixed(1),
      fold: ((fold / total) * 100).toFixed(1),
    };
  }, [scenario]);

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
      background: "#0f172a",
      color: "#e2e8f0",
      padding: "24px",
      minHeight: "100vh",
      maxWidth: "720px",
      margin: "0 auto",
    }}>
      <div style={{ marginBottom: "20px" }}>
        <h1 style={{
          fontSize: "18px",
          fontWeight: "700",
          color: "#f8fafc",
          margin: "0 0 4px 0",
          letterSpacing: "-0.5px",
        }}>
          GTO Range Viewer
        </h1>
        <p style={{ fontSize: "11px", color: "#64748b", margin: 0 }}>
          CFR-solved preflop strategy · 8 equity buckets · 1,000 iterations
        </p>
      </div>

      {/* Scenario selector */}
      <div style={{ display: "flex", gap: "6px", marginBottom: "16px" }}>
        {Object.keys(SCENARIOS).map((name) => (
          <button
            key={name}
            onClick={() => setActiveScenario(name)}
            style={{
              padding: "6px 14px",
              fontSize: "11px",
              fontWeight: activeScenario === name ? "700" : "400",
              background: activeScenario === name ? "#1e40af" : "#1e293b",
              color: activeScenario === name ? "#fff" : "#94a3b8",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              transition: "all 0.2s",
              fontFamily: "inherit",
            }}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Range summary */}
      <div style={{
        display: "flex",
        gap: "16px",
        marginBottom: "12px",
        fontSize: "12px",
      }}>
        <span><span style={{ color: "#ef4444", fontWeight: "700" }}>■</span> Raise {stats.raise}%</span>
        <span><span style={{ color: "#3b82f6", fontWeight: "700" }}>■</span> Call {stats.call}%</span>
        <span><span style={{ color: "#6b7280", fontWeight: "700" }}>■</span> Fold {stats.fold}%</span>
      </div>

      {/* Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(13, 1fr)",
        gap: "2px",
        marginBottom: "16px",
      }}>
        {Array.from({ length: 13 }, (_, row) =>
          Array.from({ length: 13 }, (_, col) => (
            <Cell
              key={`${row}-${col}`}
              row={row}
              col={col}
              scenario={scenario}
              onHover={setHovered}
            />
          ))
        )}
      </div>

      {/* Hover detail */}
      <div style={{
        background: "#1e293b",
        borderRadius: "8px",
        padding: "12px 16px",
        minHeight: "64px",
        border: "1px solid #334155",
      }}>
        {hovered ? (
          <div>
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}>
              <span style={{ fontSize: "16px", fontWeight: "800", color: "#f8fafc" }}>
                {hovered.hand}
              </span>
              <span style={{ fontSize: "11px", color: "#64748b" }}>
                Bucket {hovered.bucket} · Equity {
                  hovered.bucket !== undefined
                    ? ["0.32-0.38","0.38-0.45","0.45-0.51","0.51-0.57","0.58-0.64","0.65-0.70","0.73-0.76","0.80-0.84"][hovered.bucket]
                    : "?"
                }
              </span>
            </div>
            {hovered.strat && <ActionBar strat={hovered.strat} />}
            <div style={{ display: "flex", gap: "16px", marginTop: "8px", fontSize: "12px" }}>
              {hovered.strat && Object.entries(hovered.strat)
                .filter(([_, v]) => v > 0.005)
                .map(([action, weight]) => (
                  <span key={action} style={{ color: ACTION_COLORS[action].color }}>
                    {ACTION_COLORS[action].label} {(weight * 100).toFixed(0)}%
                  </span>
                ))
              }
            </div>
          </div>
        ) : (
          <p style={{ fontSize: "12px", color: "#475569", margin: 0 }}>
            Hover over a hand to see its strategy breakdown
          </p>
        )}
      </div>

      <p style={{ fontSize: "10px", color: "#334155", marginTop: "12px", textAlign: "center" }}>
        Solved with Linear CFR · ExtensiveFormGame interface · Domain-agnostic solver
      </p>
    </div>
  );
}