import { useStore } from '../store/store.js'
import { Segmented, Stat } from './Fields.jsx'
import { expertDP, worldSize } from '../lib/mesh.js'
import { MAX_WORLD } from '../store/store.js'
import { V } from '../lib/model.js'

export default function ConfigPanel() {
  const { kind, cfg, data, seed, setKind, setCfg, newSeed, reset, rollToken } = useStore()
  const world = worldSize(cfg)
  const edp = expertDP(cfg)

  return (
    <aside className="panel">
      <section className="pblock">
        <h2>Model</h2>
        <Segmented
          value={kind}
          onChange={setKind}
          options={[
            { value: 'dense', label: 'Dense' },
            { value: 'moe', label: 'MoE' },
          ]}
        />
        <p className="hint">
          One transformer layer, hidden dim 1, vocab {'{A, B}'}, 2 tokens, causal. Every weight is a scalar,
          so every number on screen is the real number.
        </p>
      </section>

      <section className="pblock">
        <h2>Parallelism</h2>
        <Segmented
          label="Data parallel"
          value={cfg.dp}
          onChange={(dp) => setCfg({ dp })}
          options={[1, 2, 4].map((v) => ({
            value: v,
            label: String(v),
            title: v * cfg.tp > MAX_WORLD ? `drops TP to ${MAX_WORLD / v} — ${MAX_WORLD} ranks max` : '',
          }))}
        />
        <Segmented
          label="Tensor parallel"
          value={cfg.tp}
          onChange={(tp) => setCfg({ tp })}
          options={[1, 2].map((v) => ({
            value: v,
            label: String(v),
            title: v * cfg.dp > MAX_WORLD ? `drops DP to ${MAX_WORLD / v} — ${MAX_WORLD} ranks max` : '',
          }))}
        />
        {kind === 'moe' && (
          <Segmented label="Expert parallel" value={cfg.ep} onChange={(ep) => setCfg({ ep })} options={[1, 2]} />
        )}
        <Segmented
          label="ZeRO stage"
          value={cfg.zero}
          onChange={(zero) => setCfg({ zero })}
          options={[
            { value: 0, label: 'DDP' },
            { value: 1, label: '1' },
            { value: 2, label: '2' },
            { value: 3, label: '3' },
          ]}
        />
        <div className="stats">
          <Stat label="world" value={world} hint={`${MAX_WORLD} ranks max`} />
          <Stat label="ranks/TP" value={cfg.tp} />
          <Stat label="ranks/DP" value={cfg.dp} />
          {kind === 'moe' && (
            <Stat
              label="expert-DP"
              value={edp}
              hint={edp === 1 ? 'nothing for ZeRO to shard' : 'experts get their own reduce-scatter'}
            />
          )}
        </div>
      </section>

      <section className="pblock">
        <h2>
          Input
          <span className="hbtns">
            <button type="button" className="mini" onClick={newSeed}>new</button>
            <button type="button" className="mini" onClick={reset}>reset</button>
          </span>
        </h2>
        <p className="seedline mono">
          seed <b>{seed}</b> · vocab {V}
        </p>
        {data.map((d, i) => (
          <div key={i} className="seqrow">
            <span className="seql mono">dp{i}</span>
            <div className="toks">
              {d.ids.map((v, t) => (
                <button
                  key={t}
                  type="button"
                  className="tok mono num"
                  onClick={() => rollToken(i, t)}
                  title={`token ${v} — click to reroll`}
                >
                  {v}
                </button>
              ))}
            </div>
            <span className="seqarrow">→</span>
            <div className="toks">
              {d.tgt.map((v, t) => (
                <span key={t} className="tok tgt mono num" title={`target ${v}`}>
                  {v}
                </span>
              ))}
            </div>
          </div>
        ))}
        <p className="hint">
          Token ids drawn from a {V}-entry vocabulary. Click one to reroll it, or <b>new</b> to redraw the
          whole example. Everything downstream recomputes.
        </p>
      </section>
    </aside>
  )
}
