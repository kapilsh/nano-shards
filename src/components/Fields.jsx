export function Segmented({ value, onChange, options, label }) {
  return (
    <div className="field">
      {label && <span className="field-label">{label}</span>}
      <div className="segmented" role="radiogroup" aria-label={label}>
        {options.map((o) => {
          const v = o.value ?? o
          return (
            <button
              key={v}
              type="button"
              className={v === value ? 'on' : ''}
              disabled={o.disabled}
              title={o.title}
              onClick={() => !o.disabled && onChange(v)}
            >
              {o.label ?? o}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-l">{label}</span>
      <span className="stat-v mono num">{value}</span>
      {hint && <span className="stat-h">{hint}</span>}
    </div>
  )
}
