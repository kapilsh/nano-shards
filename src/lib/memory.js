import { paramList } from './model.js'
import { allRanks, expertDP } from './mesh.js'
import { holds } from './shard.js'

// Per-rank memory ledger, in scalars (every parameter here is one number).
// Bytes are shown too, under a configurable dtype, so the stage-by-stage
// savings read the way they would on a real model.

export const DTYPE = { bf16: 2, fp32: 4 }

function ownedBy(cfg, model, c, params) {
  return params.filter((p) => holds(cfg, p, c)).filter((p) => {
    if (cfg.zero === 0) return true
    const deg = p.group === 'expert' ? expertDP(cfg) : cfg.dp
    if (deg === 1) return true
    const peers = allRanks(cfg).filter((o) => o.tp === c.tp && holds(cfg, p, o)).sort((a, b) => a.dp - b.dp)
    const idx = params.filter((q) => (q.group === 'expert') === (p.group === 'expert')).findIndex((q) => q.fqn === p.fqn)
    return peers[idx % peers.length]?.rank === c.rank
  })
}

export function memoryFor(cfg, model) {
  const params = paramList(model)
  return allRanks(cfg).map((c) => {
    const held = params.filter((p) => holds(cfg, p, c))
    const owned = ownedBy(cfg, model, c, params)
    const nParams = cfg.zero >= 3 ? owned.length : held.length
    const nGrads = cfg.zero >= 2 ? owned.length : held.length
    const nOptim = cfg.zero >= 1 ? owned.length : held.length
    // ZeRO-3 gathers one unit at a time; peak adds the largest unit back.
    const units = groupSizes(held)
    const peak = cfg.zero >= 3 ? nParams + Math.max(0, ...units) : nParams
    return { ...c, held: held.length, nParams, nGrads, nOptim, peak, owned: owned.map((p) => p.fqn) }
  })
}

const groupSizes = (held) => {
  const m = new Map()
  for (const p of held) m.set(p.group, (m.get(p.group) || 0) + 1)
  return [...m.values()]
}

/** Activation scalars a rank keeps for one microbatch, with / without checkpointing. */
export function activationFor(cfg, model, S = 2) {
  const perHead = 4 * S // q, k, v, a
  const heads = model.heads.length / cfg.tp
  const cols = (model.kind === 'dense' ? model.mlp.length : model.experts[0].length) / cfg.tp
  const inner = heads * perHead + cols * 2 * S + 3 * S // + x, h, y
  const boundaries = 3 * S // x, h, y kept as checkpoints
  return { full: inner, checkpointed: boundaries, recomputePeak: boundaries + inner / 1 }
}
