import { val, sum, softmax, crossEntropy } from './autograd.js'
import { rng, round2 } from './rand.js'

// ---------------------------------------------------------------- presets --
//
// Weights are the ones from the worked example: they are chosen so the dense
// and MoE variants produce identical activations up to the FFN, and so that a
// 2-way head / column split reproduces the unfactored model exactly.
//   head 0 gives .5*v, head 1 gives .25*(2v) = .5*v  ->  sums to wo=1
//   mlp  0 gives .5*h, col  1 gives .25*(2h) = .5*h  ->  sums to h

/** Vocabulary size. Big enough that vocab-parallel sharding is a real split,
 *  small enough that a token id is still a readable label. */
export const V = 100
export const DEFAULT_SEED = 1337

/** Embedding and LM-head rows are drawn from the seed, so they are reproducible. */
function vocabRows(seed) {
  const r = rng(seed)
  const E = Array.from({ length: V }, () => round2(r() * 2 - 1))
  const U = Array.from({ length: V }, () => round2(r() * 2 - 1))
  return { E, U }
}

export const tokLabel = (id) => `t${id}`

export function denseModel(seed = DEFAULT_SEED) {
  const { E, U } = vocabRows(seed)
  return {
    kind: 'dense',
    E,
    heads: [
      { wq: 0, wk: 1, wv: 1, wo: 0.5 },
      { wq: 0, wk: 1, wv: 2, wo: 0.25 },
    ],
    mlp: [
      { w1: 1, w2: 0.5 },
      { w1: 2, w2: 0.25 },
    ],
    U,
  }
}

export function moeModel(seed = DEFAULT_SEED) {
  const { E, U } = vocabRows(seed)
  return {
    kind: 'moe',
    E,
    heads: [
      { wq: 0, wk: 1, wv: 1, wo: 0.5 },
      { wq: 0, wk: 1, wv: 2, wo: 0.25 },
    ],
    router: { r: [1, 0], b: [0, 1.5] },
    experts: [
      [{ w1: 1, w2: 0.5 }, { w1: 2, w2: 0.25 }], // expert 0 -> h
      [{ w1: 1, w2: 0.25 }, { w1: 1, w2: 0.25 }], // expert 1 -> .5h
    ],
    U,
  }
}

/** A seeded batch: one sequence of token ids per data-parallel replica. */
export function makeData(seed = DEFAULT_SEED, dp = 2, S = 2) {
  const r = rng(seed ^ 0x9e3779b9)
  return Array.from({ length: dp }, () => {
    const ids = Array.from({ length: S }, () => Math.floor(r() * V))
    // next-token targets, the last one wrapping so the sequence stays closed
    const tgt = ids.map((_, t) => (t + 1 < S ? ids[t + 1] : ids[0]))
    return { ids, tgt }
  })
}

export const DATA = makeData()

// ------------------------------------------------------------- parameters --

/** Flat list of every scalar parameter, with the axis it is sharded along. */
export function paramList(m) {
  const P = []
  m.E.forEach((_, v) => P.push({ fqn: `embed.E${v}`, group: 'embed', shard: 'vocab', idx: v }))
  m.heads.forEach((_, i) => {
    for (const nm of ['wq', 'wk', 'wv'])
      P.push({ fqn: `attn.h${i}.${nm}`, group: 'attn', shard: 'head', idx: i, col: true })
    P.push({ fqn: `attn.h${i}.wo`, group: 'attn', shard: 'head', idx: i, row: true })
  })
  if (m.kind === 'dense') {
    m.mlp.forEach((_, j) => {
      P.push({ fqn: `mlp.c${j}.w1`, group: 'mlp', shard: 'col', idx: j, col: true })
      P.push({ fqn: `mlp.c${j}.w2`, group: 'mlp', shard: 'col', idx: j, row: true })
    })
  } else {
    m.router.r.forEach((_, e) => P.push({ fqn: `router.r${e}`, group: 'router', shard: 'replicated' }))
    m.router.b.forEach((_, e) => P.push({ fqn: `router.b${e}`, group: 'router', shard: 'replicated' }))
    m.experts.forEach((cols, e) =>
      cols.forEach((_, j) => {
        P.push({ fqn: `experts.e${e}.c${j}.w1`, group: 'expert', shard: 'expert', expert: e, idx: j, col: true })
        P.push({ fqn: `experts.e${e}.c${j}.w2`, group: 'expert', shard: 'expert', expert: e, idx: j, row: true })
      }),
    )
  }
  m.U.forEach((_, v) => P.push({ fqn: `head.U${v}`, group: 'head', shard: 'vocab', idx: v }))
  return P
}

/** Wrap every parameter as an autograd leaf, keyed by fqn. */
export function leaves(m) {
  const L = {}
  const put = (fqn, x) => (L[fqn] = val(x, fqn))
  m.E.forEach((x, v) => put(`embed.E${v}`, x))
  m.heads.forEach((hd, i) => {
    for (const nm of ['wq', 'wk', 'wv', 'wo']) put(`attn.h${i}.${nm}`, hd[nm])
  })
  if (m.kind === 'dense') {
    m.mlp.forEach((c, j) => {
      put(`mlp.c${j}.w1`, c.w1)
      put(`mlp.c${j}.w2`, c.w2)
    })
  } else {
    m.router.r.forEach((x, e) => put(`router.r${e}`, x))
    m.router.b.forEach((x, e) => put(`router.b${e}`, x))
    m.experts.forEach((cols, e) =>
      cols.forEach((c, j) => {
        put(`experts.e${e}.c${j}.w1`, c.w1)
        put(`experts.e${e}.c${j}.w2`, c.w2)
      }),
    )
  }
  m.U.forEach((x, v) => put(`head.U${v}`, x))
  return L
}

// ---------------------------------------------------------------- forward --

/**
 * One replica's forward pass. Returns the loss plus every intermediate we want
 * to show, keeping per-head / per-column / per-expert pieces separate so the
 * sharding layer can route them to whichever rank owns them.
 */
export function forward(m, L, { ids, tgt }) {
  const S = ids.length
  const x = ids.map((v) => L[`embed.E${v}`])

  // attention, one head at a time -------------------------------------------
  const perHead = m.heads.map((_, i) => {
    const q = x.map((t) => L[`attn.h${i}.wq`].mul(t))
    const k = x.map((t) => L[`attn.h${i}.wk`].mul(t))
    const v = x.map((t) => L[`attn.h${i}.wv`].mul(t))
    const a = []
    const p = []
    for (let t = 0; t < S; t++) {
      const scores = []
      for (let u = 0; u <= t; u++) scores.push(q[t].mul(k[u])) // causal
      const pt = softmax(scores)
      p.push(pt)
      a.push(sum(pt.map((w, u) => w.mul(v[u]))))
    }
    return { q, k, v, p, a, o: a.map((at) => L[`attn.h${i}.wo`].mul(at)) }
  })
  const attn = Array.from({ length: S }, (_, t) => sum(perHead.map((hh) => hh.o[t])))
  const h = x.map((xt, t) => xt.add(attn[t]))

  // feed-forward ------------------------------------------------------------
  let ffn, perCol = null, route = null
  if (m.kind === 'dense') {
    perCol = m.mlp.map((_, j) => {
      const u = h.map((ht) => L[`mlp.c${j}.w1`].mul(ht))
      return { u, out: u.map((ut) => L[`mlp.c${j}.w2`].mul(ut.relu())) }
    })
    ffn = Array.from({ length: S }, (_, t) => sum(perCol.map((c) => c.out[t])))
  } else {
    route = h.map((ht) => {
      const lg = m.router.r.map((_, e) => L[`router.r${e}`].mul(ht).add(L[`router.b${e}`]))
      const pr = softmax(lg)
      let sel = 0
      for (let e = 1; e < lg.length; e++) if (lg[e].data > lg[sel].data) sel = e
      return { lg, pr, sel, gate: pr[sel] }
    })
    const expOut = h.map((ht, t) => {
      const e = route[t].sel
      return sum(m.experts[e].map((_, j) =>
        L[`experts.e${e}.c${j}.w2`].mul(L[`experts.e${e}.c${j}.w1`].mul(ht).relu())))
    })
    ffn = h.map((_, t) => route[t].gate.mul(expOut[t]))
    route.forEach((r, t) => (r.out = expOut[t]))
  }
  const y = h.map((ht, t) => ht.add(ffn[t]))

  // head + loss -------------------------------------------------------------
  const z = y.map((yt) => m.U.map((_, v) => yt.mul(L[`head.U${v}`])))
  const perTok = z.map((zt, t) => crossEntropy(zt, tgt[t]))
  const loss = sum(perTok).div(S)

  return { x, perHead, attn, h, perCol, route, ffn, y, z, perTok, loss }
}

/** One replica on its own: local forward + local backward, as a GPU sees it. */
export function runReplica(m, d) {
  const L = leaves(m)
  const out = forward(m, L, d)
  out.loss.backward()
  const grads = {}
  for (const k of Object.keys(L)) grads[k] = L[k].grad
  return { L, out, grads }
}

/** Replicas for a given data-parallel degree (the sample data cycles). */
export const replicasFor = (dp, data = DATA) => Array.from({ length: dp }, (_, i) => data[i % data.length])

/**
 * A full step. Each replica computes its own local gradient; the global
 * gradient is their mean, which is exactly what the DP reduction produces.
 */
export function runStep(m, data = DATA) {
  const per = data.map((d) => runReplica(m, d))
  const grads = {}
  for (const k of Object.keys(per[0].grads))
    grads[k] = per.reduce((a, r) => a + r.grads[k], 0) / per.length
  const loss = per.reduce((a, r) => a + r.out.loss.data, 0) / per.length
  return { per, reps: per.map((r) => r.out), grads, loss, L: per[0].L }
}

/** SGD with momentum; on step 1 (m = 0) this is w <- w - lr*g. */
export function sgdStep(grads, L, lr = 0.1, beta = 0.9, mom = {}) {
  const out = {}
  for (const k of Object.keys(L)) {
    const mk = beta * (mom[k] || 0) + grads[k]
    out[k] = { m: mk, w: L[k].data - lr * mk }
  }
  return out
}
