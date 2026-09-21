import { buildTrace } from './trace.js'
import { GROUPS, coords, worldSize } from './mesh.js'

/** Roll the trace up into a per-step communication log. */
export function commsFor(cfg, model, data) {
  const t = buildTrace(cfg, model, data)
  const rows = []
  const scan = (steps, phase) => {
    for (const s of steps) {
      if (s.kind !== 'comm') continue
      const c = coords(cfg, 0)
      const members = GROUPS[s.group].of(cfg, c)
      rows.push({
        phase,
        group: GROUPS[s.group].label,
        op: s.name,
        payload: s.edge || s.sub || '',
        size: members.length,
        groups: worldSize(cfg) / members.length,
        tag: s.tag || '',
      })
    }
  }
  scan(t.forward, 'fwd')
  scan(t.backward, 'bwd')
  return rows
}
