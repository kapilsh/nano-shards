import { useMemo } from 'react'
import { useStore } from '../store/store.js'
import { memoryFor, activationFor } from '../lib/memory.js'
import { paramList } from '../lib/model.js'

const SEG = [
  { key: 'nParams', label: 'params', color: 'var(--mem-params)' },
  { key: 'nGrads', label: 'grads', color: 'var(--mem-grads)' },
  { key: 'nOptim', label: 'optimizer', color: 'var(--mem-optim)' },
]

export default function MemoryView() {
  const { cfg, kind, model, rank } = useStore()
  const m = model()
  const rows = useMemo(() => memoryFor(cfg, m), [cfg, kind]) // eslint-disable-line react-hooks/exhaustive-deps
  const total = paramList(m).length
  const max = Math.max(...rows.map((r) => SEG.reduce((a, s) => a + r[s.key], 0)), 1)
  const act = activationFor(cfg, m)

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h2>Memory per rank</h2>
          <p className="card-sub">
            In scalars — this model has {total} parameters in total. ZeRO-1 shards the green segment, ZeRO-2 the
            orange, ZeRO-3 the blue.
          </p>
        </div>
        <div className="lens-row">
          {SEG.map((s) => (
            <span key={s.key} className="chip on" style={{ cursor: 'default', color: s.color }}>
              <span className="swatch" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>

      <div className="bars">
        {rows.map((r) => {
          const tot = SEG.reduce((a, s) => a + r[s.key], 0)
          return (
            <div key={r.rank} className={`bar ${r.rank === rank ? 'on' : ''}`}>
              <span className="bar-l mono">GPU{r.rank}</span>
              <div className="bar-t">
                {SEG.map((s) => (
                  <div key={s.key} className="bar-s" style={{ width: `${(100 * r[s.key]) / max}%`, background: s.color }}
                    title={`${s.label}: ${r[s.key]}`}>
                    {r[s.key] >= max * 0.08 ? r[s.key] : ''}
                  </div>
                ))}
              </div>
              <span className="bar-n mono num">{tot}</span>
            </div>
          )
        })}
      </div>

      <div className="mem-notes">
        <div>
          <b>Holds at rest</b>
          <span className="faint"> — {rows[0].held} of {total} parameters reach each rank after TP{cfg.ep > 1 ? ' and EP' : ''} sharding.</span>
        </div>
        {cfg.zero >= 3 && (
          <div>
            <b>Peak parameters</b>
            <span className="faint"> — {rows[0].peak} while a unit is gathered, against {rows[0].nParams} at rest.</span>
          </div>
        )}
        <div>
          <b>Activations</b>
          <span className="faint">
            {' '}— {act.full} scalars per rank stored, or {act.checkpointed} with activation checkpointing,
            recomputed one layer at a time. ZeRO never touches this column.
          </span>
        </div>
      </div>
    </div>
  )
}
