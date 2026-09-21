import { useStore } from '../store/store.js'
import { coords, GROUPS, expertDP } from '../lib/mesh.js'

const C = { tp: 'var(--ax-tp)', dp: 'var(--ax-dp)', ep: 'var(--ax-ep)', edp: 'var(--ax-edp)' }

export default function MeshView() {
  const { cfg, kind, rank, lens, setRank, setLens, data } = useStore()
  const me = coords(cfg, rank)
  const keys = Object.keys(GROUPS).filter((k) => (k === 'ep' || k === 'edp' ? kind === 'moe' : true))
  const active = keys.includes(lens) ? lens : 'tp'
  const group = GROUPS[active].of(cfg, me)

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <h2>Device mesh</h2>
          <p className="card-sub">
            Rows are data-parallel replicas; columns are tensor-parallel ranks. Click a rank to inspect it.
          </p>
        </div>
        <div className="lens-row">
          {keys.map((k) => (
            <button key={k} type="button" className={`chip ${active === k ? 'on' : ''}`} onClick={() => setLens(k)}
              style={active === k ? { color: C[k], borderColor: C[k] } : undefined}>
              <span className="swatch" style={{ background: C[k] }} />
              {GROUPS[k].label}
            </button>
          ))}
        </div>
      </div>

      <div className="mesh">
        {Array.from({ length: cfg.dp }, (_, dp) => (
          <div key={dp} className="mesh-row">
            <span className="mesh-lab mono">
              dp{dp}
              {kind === 'moe' && cfg.ep > 1 && <span className="faint"> · ep{dp % cfg.ep}</span>}
            </span>
            {Array.from({ length: cfg.tp }, (_, tp) => {
              const r = dp * cfg.tp + tp
              const inG = group.includes(r)
              return (
                <button key={r} type="button"
                  className={`mcell ${inG ? 'in' : ''} ${r === rank ? 'sel' : ''}`}
                  style={{ color: C[active] }}
                  onClick={() => setRank(r)}>
                  <span className="mr mono">GPU{r}</span>
                  <span className="md mono">{data[dp]?.ids.map((v) => 'AB'[v]).join('')} · tp{tp}</span>
                </button>
              )
            })}
          </div>
        ))}
      </div>

      <div className="mesh-note" style={{ borderLeftColor: C[active] }}>
        <b>{GROUPS[active].label} group of GPU{rank}:</b> {group.map((r) => `GPU${r}`).join(', ')}
        <br />
        <span className="faint">{GROUPS[active].note}</span>
        {kind === 'moe' && active === 'edp' && expertDP(cfg) === 1 && (
          <>
            <br />
            <span className="warn-text">
              expert-DP = 1 — each expert lives on exactly one rank, so its gradients need no reduction and
              ZeRO has nothing to shard them across.
            </span>
          </>
        )}
      </div>
    </div>
  )
}
