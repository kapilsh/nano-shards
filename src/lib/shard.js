import { paramList, V } from './model.js'
import { coords, slice, worldSize, expertDP } from './mesh.js'

// Where does each parameter live, and which rank updates it?
//
// Two independent questions:
//   1. model sharding  — TP splits heads/columns/vocab; EP splits experts
//   2. ZeRO sharding   — within the DP group, who owns optimizer state
//                        (stage>=1), gradients (>=2) and parameters (>=3)

/** Does this rank hold a copy of this parameter at all? */
export function holds(cfg, p, c) {
  switch (p.shard) {
    case 'vocab':
      return slice(V, cfg.tp, c.tp).includes(p.idx)
    case 'head':
      return slice(cfg.nHeads, cfg.tp, c.tp).includes(p.idx)
    case 'col':
      return slice(cfg.nCols, cfg.tp, c.tp).includes(p.idx)
    case 'expert':
      return slice(cfg.nExperts, cfg.ep, c.ep).includes(p.expert) && slice(cfg.nCols, cfg.tp, c.tp).includes(p.idx)
    case 'replicated':
      return true
    default:
      return true
  }
}

/** Ranks that hold this parameter. */
export const holders = (cfg, p) =>
  Array.from({ length: worldSize(cfg) }, (_, r) => coords(cfg, r)).filter((c) => holds(cfg, p, c))

/**
 * ZeRO ownership. Among the ranks that hold a parameter, the ones in the same
 * DP group are redundant copies — that redundancy is what ZeRO removes.
 * Expert parameters shard over expert-DP instead, which is why expert-DP = 1
 * leaves them with nothing to shard across.
 */
export function zeroOwner(cfg, p, model) {
  const hs = holders(cfg, p)
  if (!hs.length) return null
  const isExpert = p.group === 'expert'
  const degree = isExpert ? expertDP(cfg) : cfg.dp
  if (cfg.zero === 0 || degree === 1) return { owners: hs.map((c) => c.rank), degree, sharded: false }
  // rank the holders inside their DP group and deal parameters out round-robin
  const byTp = new Map()
  for (const c of hs) {
    const k = c.tp
    if (!byTp.has(k)) byTp.set(k, [])
    byTp.get(k).push(c)
  }
  const idx = paramIndex(cfg, p, model)
  const owners = []
  for (const [, list] of byTp) {
    list.sort((a, b) => a.dp - b.dp)
    owners.push(list[idx % list.length].rank)
  }
  return { owners, degree, sharded: true }
}

/** Stable position of a parameter within its ZeRO bucket, so the deal is deterministic. */
function paramIndex(cfg, p, model) {
  const all = paramList(model).filter((q) => (q.group === 'expert') === (p.group === 'expert'))
  return all.findIndex((q) => q.fqn === p.fqn)
}

/** What each rank keeps at rest, per ZeRO stage. */
export function residency(cfg, p, c, model) {
  const has = holds(cfg, p, c)
  if (!has) return { param: false, grad: false, optim: false }
  const z = zeroOwner(cfg, p, model)
  const mine = z.owners.includes(c.rank)
  return {
    param: cfg.zero < 3 ? true : mine,
    grad: cfg.zero < 2 ? true : mine,
    optim: cfg.zero < 1 ? true : mine,
  }
}

export function placementLabel(cfg, p) {
  const bits = []
  if (p.shard === 'expert') bits.push(`EP Shard/${cfg.ep}`)
  if (p.shard === 'vocab') bits.push(`TP Shard(vocab)/${cfg.tp}`)
  if (p.shard === 'head') bits.push(`TP Shard(head)/${cfg.tp}`)
  if (p.shard === 'col') bits.push(`TP Shard(col)/${cfg.tp}`)
  if (p.shard === 'replicated') bits.push('TP Replicate')
  const deg = p.group === 'expert' ? expertDP(cfg) : cfg.dp
  if (cfg.zero > 0 && deg > 1) bits.push(`ZeRO-${cfg.zero} Shard/${deg}`)
  else bits.push(`DP Replicate/${deg}`)
  return bits.join(' → ')
}
