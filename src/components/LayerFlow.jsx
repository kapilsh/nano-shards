import { useMemo, useRef, useState, useLayoutEffect } from 'react'
import { useStore } from '../store/store.js'
import { buildTrace } from '../lib/trace.js'
import Tensor from './Tensor.jsx'
import { Segmented } from './Fields.jsx'

const GC = { tp: 'var(--ax-tp)', dp: 'var(--ax-dp)', ep: 'var(--ax-ep)', edp: 'var(--ax-edp)' }

function RankTile({ cell, cfg, focus, onClick }) {
  const dp = Math.floor(cell.rank / cfg.tp)
  const tp = cell.rank % cfg.tp
  return (
    <button type="button" className={`tile ${cell.state || ''} ${focus ? 'focus' : ''}`} onClick={onClick}>
      <div className="tile-h">
        <span className="tile-r mono">GPU{cell.rank}</span>
        <span className="tile-c mono">dp{dp}·tp{tp}</span>
      </div>
      {cell.params?.length > 0 && (
        <div className="tile-p">
          {cell.params.map((p, i) => (
            <span key={i} className="pill mono">
              {p.n}<b>{typeof p.v === 'number' ? ` ${p.v}` : ''}</b>
            </span>
          ))}
        </div>
      )}
      {cell.tokens && (
        <div className="tile-tok">
          {cell.tokens.map((t, i) => <span key={i} className="tk mono">{t}</span>)}
          <span className="tk-arrow mono">→</span>
          {cell.target.map((t, i) => <span key={i} className="tk tgt mono">{t}</span>)}
        </div>
      )}
      {cell.tensors?.map((t, i) => <Tensor key={i} t={t} />)}
      {cell.note && <div className="tile-n">{cell.note}</div>}
    </button>
  )
}

function Edge({ s, up }) {
  return (
    <div className={`lf-edge ${up ? 'up' : ''}`}>
      <span className="tip" />
      {s.edge && <span className="lab mono">{s.edge}</span>}
    </div>
  )
}

function Step({ s, cfg, rank, setRank }) {
  const world = cfg.dp * cfg.tp
  return (
    <div className={`lf-node ${s.kind}`} style={s.kind === 'comm' ? { '--nc': GC[s.group] } : undefined}>
      <div className="lf-title">
        <span className="nm">{s.name}</span>
        {s.tag && <span className="tag">{s.tag}</span>}
      </div>
      {s.sub && <div className="lf-sub">{s.sub}</div>}
      <div className="tiles" style={{ gridTemplateColumns: `repeat(${world}, 1fr)` }}>
        {s.cells.map((c) => (
          <RankTile key={c.rank} cell={c} cfg={cfg} focus={c.rank === rank} onClick={() => setRank(c.rank)} />
        ))}
      </div>
    </div>
  )
}

/** Group the steps so the transformer layer can be drawn inside a band. */
function group(steps) {
  const out = []
  steps.forEach((s, i) => {
    if (s.block) {
      if (!out.length || out[out.length - 1].kind !== 'band') out.push({ kind: 'band', steps: [], at: i })
      out[out.length - 1].steps.push(s)
    } else out.push({ kind: 'step', step: s, at: i })
  })
  return out
}

/** Residual spans, in rows of the band grid: where each skip forks and rejoins. */
function residuals(steps) {
  const rows = zones(steps)
  const rowOf = (step) => rows.findIndex((r) => (r.kind === 'zone' ? r.steps.includes(step) : r.step === step))
  const out = []
  steps.forEach((s) => {
    for (const k of s.forks || []) {
      const end = steps.find((t) => (t.joins || []).includes(k))
      if (!end) continue
      const from = rowOf(s), to = rowOf(end)
      if (from >= 0 && to > from) out.push({ key: k, from, to })
    }
  })
  return out
}

/** Within the layer band, the feed-forward region gets a band of its own. */
function zones(steps) {
  const out = []
  for (const s of steps) {
    const last = out[out.length - 1]
    if (s.zone) {
      if (last?.kind === 'zone' && last.zone === s.zone) last.steps.push(s)
      else out.push({ kind: 'zone', zone: s.zone, steps: [s] })
    } else out.push({ kind: 'step', step: s })
  }
  return out
}


/**
 * The transformer-layer band.
 *
 * Where a residual forks, the flow genuinely splits: one branch carries on down
 * the main path, the other runs out to the gutter and rejoins further down.
 * Both branches get their own arrowhead, and the vertical connector for that
 * row is drawn here in SVG rather than by CSS so the split is one shape.
 */
function Band({ steps, up, props }) {
  const ref = useRef(null)
  const [geo, setGeo] = useState([])
  const spans = useMemo(() => residuals(steps), [steps])
  const rows = useMemo(() => zones(steps), [steps])
  const forkRows = useMemo(() => new Set(spans.map((s) => s.from)), [spans])

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const GUT = 14 // clearance between the leftmost content and the rail
    const measure = () => {
      const base = el.getBoundingClientRect()
      const box = (row, sel = '.lf-node') => {
        const n = el.querySelector(`[data-row="${row}"] ${sel}`)
        if (!n) return null
        const r = n.getBoundingClientRect()
        return { l: r.left - base.left, t: r.top - base.top, w: r.width, h: r.height }
      }
      // The rail must clear everything in the band, including the nested
      // MoE/MLP band, which reaches further left than an ordinary node.
      let minL = Infinity
      el.querySelectorAll('[data-row] > .lf-node, [data-row] > .lf-band, [data-row] .lf-band.inner').forEach((n) => {
        minL = Math.min(minL, n.getBoundingClientRect().left - base.left)
      })
      if (!Number.isFinite(minL)) minL = 0
      const next = []
      for (const sp of spans) {
        const a = box(sp.from)
        const b = box(sp.to)
        const nx = box(sp.from + 1)
        if (!a || !b) continue
        next.push({ key: sp.key, a, b, nx, gx: Math.max(6, minL - GUT) })
      }
      setGeo(next)
    }
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    el.querySelectorAll('.lf-node').forEach((n) => ro.observe(n))
    return () => ro.disconnect()
  }, [spans, rows])

  const R = 8

  return (
    <div className="lf-band" ref={ref}>
      <span className="lf-band-tag">transformer layer</span>
      <svg className="lf-res-svg" aria-hidden="true">
        <defs>
          <marker id="res-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ax-dp)" />
          </marker>
        </defs>
        {geo.map(({ key, a, b, nx, gx }) => {
          const sx = a.l + a.w / 2 // split point: bottom centre of the forking node
          const sy = a.t + a.h
          const jy = b.t + b.h / 2
          // branch 1 — straight on down the main path
          const down = nx ? `M ${sx} ${sy} L ${sx} ${nx.t - 3}` : ''
          // branch 2 — out to the gutter, down, and back into the joining node
          const side =
            `M ${sx} ${sy} L ${sx} ${sy + 9} Q ${sx} ${sy + 9 + R} ${sx - R} ${sy + 9 + R} ` +
            `L ${gx + R} ${sy + 9 + R} Q ${gx} ${sy + 9 + R} ${gx} ${sy + 9 + 2 * R} ` +
            `L ${gx} ${jy - R} Q ${gx} ${jy} ${gx + R} ${jy} L ${b.l - 3} ${jy}`
          return (
            <g key={key}>
              {down && <path className="res-line" d={down} markerEnd="url(#res-arrow)" />}
              {down && <path className={`res-flow ${up ? 'rev' : ''}`} d={down} />}
              <path className="res-line" d={side} markerEnd="url(#res-arrow)" />
              <path className={`res-flow ${up ? 'rev' : ''}`} d={side} />
              <circle className="res-dot" cx={sx} cy={sy} r="3" />
              <text className="res-lab" x={gx} y={sy + 9 + 2 * R + 13} textAnchor="middle">{key}</text>
            </g>
          )
        })}
      </svg>
      {rows.map((z, zi) =>
        z.kind === 'zone' ? (
          <div key={zi} data-row={zi} className="lf-bandwrap">
            {zi > 0 && !forkRows.has(zi - 1) && <Edge s={z.steps[0]} up={up} />}
            {forkRows.has(zi - 1) && <div className="lf-edge blank" />}
            <div className="lf-band inner">
              <span className="lf-band-tag">{z.zone}</span>
              {z.steps.map((s, i) => (
                <div key={s.id} className="lf-step">
                  {i > 0 && <Edge s={s} up={up} />}
                  <Step s={s} {...props} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div key={zi} data-row={zi} className="lf-step">
            {zi > 0 && !forkRows.has(zi - 1) && <Edge s={z.step} up={up} />}
            {forkRows.has(zi - 1) && <div className="lf-edge blank" />}
            <Step s={z.step} {...props} />
          </div>
        ),
      )}
    </div>
  )
}

function Column({ title, hint, steps, cfg, rank, setRank, up }) {
  const groups = group(steps)
  const props = { cfg, rank, setRank }
  return (
    <section className="lf-col">
      <header className="lf-colh">
        <h3>{title}</h3>
        <span className="faint">{hint}</span>
      </header>
      <div className="lf">
        {groups.map((g, gi) => {
          // The connector into a band belongs outside it, so the edge from the
          // previous stage visibly crosses the boundary instead of starting inside.
          if (g.kind === 'band')
            return (
              <div key={gi} className="lf-bandwrap">
                {g.at > 0 && <Edge s={g.steps[0]} up={up} />}
                <Band steps={g.steps} up={up} props={props} />
              </div>
            )
          return (
            <div key={gi} className="lf-step">
              {g.at > 0 && <Edge s={g.step} up={up} />}
              <Step s={g.step} {...props} />
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default function LayerFlow() {
  const { cfg, kind, data, rank, setRank, mode, setMode, model } = useStore()
  const m = model()
  const trace = useMemo(() => buildTrace(cfg, m, data), [cfg, kind, data]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h2>Layer flow</h2>
          <p className="card-sub">
            One transformer layer, step by step. Every tensor is the real tensor — hover a cell for full precision.
          </p>
        </div>
      </div>

      <div className="lf-toolbar">
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: 'forward', label: 'Forward' },
            { value: 'backward', label: 'Backward' },
          ]}
        />
      </div>

      <div className="lf-wrap">
        {mode === 'forward' ? (
          <Column title="Forward" hint="ids → loss" steps={trace.forward} cfg={cfg} rank={rank} setRank={setRank} />
        ) : (
          <Column title="Backward" hint="gradients → optimizer" steps={trace.backward} cfg={cfg} rank={rank} setRank={setRank} up />
        )}
      </div>

      <p className="lf-legend">
        <span className="k res" /> residual — skips the block, added back below
        <span className="k act" /> activations flowing forward
        <span className="k grad" /> gradients flowing back
        <span className="k partial" /> partial — still needs a collective
        <span className="k done" /> complete after a collective
        <span className="k absent" /> not resident on this rank
      </p>
    </div>
  )
}
