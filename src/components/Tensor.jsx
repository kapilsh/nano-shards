import { num } from '../lib/format.js'

/**
 * A tensor as a small table: one cell per element, with its name and shape.
 * These are genuinely tiny — a [2] activation is a 1x2 grid — so showing the
 * numbers themselves is more useful than showing a shard rectangle.
 */
export default function Tensor({ t }) {
  const cols = t.cols ? t.cols.length : t.d.length
  const shape = t.rows ? `${t.rows}×${cols}` : `1×${t.d.length}`
  return (
    <div className={`tns ${t.s || ''}`}>
      <div className="tns-h">
        <span className="tns-n mono">{t.n}</span>
        <span className="tns-s mono">{shape}</span>
      </div>
      {t.cols && (
        <div className="tns-cols" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
          {t.cols.map((c, i) => (
            <span key={i} className="tns-ch mono">{c}</span>
          ))}
        </div>
      )}
      <div className="tns-g" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
        {t.d.map((v, i) => (
          <span key={i} className="tns-c mono num" title={String(v)}>
            {t.fmt === 'int' ? `E${v}` : num(v, 3)}
          </span>
        ))}
      </div>
    </div>
  )
}
