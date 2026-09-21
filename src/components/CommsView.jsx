import { useMemo } from 'react'
import { useStore } from '../store/store.js'
import { commsFor } from '../lib/comms.js'

const GC = { TP: 'var(--ax-tp)', DP: 'var(--ax-dp)', EP: 'var(--ax-ep)', 'expert-DP': 'var(--ax-edp)' }

export default function CommsView() {
  const { cfg, kind, model, data } = useStore()
  const m = model()
  const rows = useMemo(() => commsFor(cfg, m, data), [cfg, kind, data]) // eslint-disable-line react-hooks/exhaustive-deps
  const byGroup = rows.reduce((a, r) => ((a[r.group] = (a[r.group] || 0) + 1), a), {})

  return (
    <div className="card table-card">
      <div className="card-head">
        <div>
          <h2>Communication</h2>
          <p className="card-sub">Every collective in one training step, in the order it fires.</p>
        </div>
        <div className="lens-row">
          {Object.entries(byGroup).map(([g, n]) => (
            <span key={g} className="chip on" style={{ cursor: 'default', color: GC[g] }}>
              <span className="swatch" style={{ background: GC[g] }} />
              {g} ×{n}
            </span>
          ))}
        </div>
      </div>
      <div className="table-scroll">
        <table className="ptable">
          <thead>
            <tr><th>#</th><th>phase</th><th>group</th><th>op</th><th>payload</th><th className="right">groups</th><th className="right">ranks</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="faint">{i + 1}</td>
                <td>{r.phase}</td>
                <td style={{ color: GC[r.group] }}>{r.group}</td>
                <td className="mono">{r.op}</td>
                <td className="faint">{r.payload}</td>
                <td className="right mono num">{r.groups}</td>
                <td className="right mono num">{r.size}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note faint">
        TP collectives sit on the critical path of every layer, so they belong on NVLink. DP collectives fire once
        per step and overlap with backward, so they tolerate slower fabric.
      </p>
    </div>
  )
}
