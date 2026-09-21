import { runReplica, replicasFor, paramList, V, tokLabel } from './model.js'
import { allRanks, slice, expertDP } from './mesh.js'
import { holds } from './shard.js'

// Builds the ordered steps for the layer-flow view.
//
// Cells carry *structured tensors*, not formatted strings: a tensor is its
// values plus a shape, so the UI can draw it as a small grid. Every number
// comes from the autograd run.



/** Tag a contiguous run of steps as a named sub-region (the FFN / MoE block). */
function markZone(steps, zone, names) {
  let a = -1, b = -1
  steps.forEach((s, i) => {
    if (names.some((n) => s.name.startsWith(n))) { if (a < 0) a = i; b = i }
  })
  if (a < 0) return
  for (let i = a; i <= b; i++) steps[i].zone = zone
}

/** Flag the contiguous run of steps that make up one transformer layer. */
function markBlock(steps, startsWith, endsWith) {
  const a = steps.findIndex((s) => startsWith.some((n) => s.name.startsWith(n)))
  const b = steps.map((s) => s.name).reduce((acc, n, i) => (endsWith.some((e) => n.startsWith(e)) ? i : acc), -1)
  if (a < 0 || b < a) return
  for (let i = a; i <= b; i++) steps[i].block = true
  steps[a].blockStart = true
  steps[b].blockEnd = true
}

const T = (n, d, s = {}) => ({ n, d: d.map((x) => (typeof x === 'number' ? x : x.data)), ...s })
const Tg = (n, vs, s = {}) => ({ n, d: vs.map((v) => v.grad), ...s })

export function buildTrace(cfg, model, data = replicasFor(cfg.dp)) {
  const ranks = allRanks(cfg)
  const runs = data.map((d) => runReplica(model, d))
  const isMoe = model.kind === 'moe'
  const S = data[0].ids.length

  const my = (c) => ({
    rep: runs[c.dp],
    heads: slice(cfg.nHeads, cfg.tp, c.tp),
    cols: slice(cfg.nCols, cfg.tp, c.tp),
    vocab: slice(V, cfg.tp, c.tp),
    lo: slice(V, cfg.tp, c.tp)[0],
    hi: slice(V, cfg.tp, c.tp).slice(-1)[0],
    experts: isMoe ? slice(cfg.nExperts, cfg.ep, c.ep) : [],
  })
  const cells = (fn) => ranks.map((c) => ({ rank: c.rank, ...fn(c, my(c)) }))
  const F = []
  const push = (s) => F.push({ id: `f${F.length}`, ...s })

  // =============================== FORWARD ===============================
  push({
    kind: 'io', name: 'input ids', tag: 'one sequence per DP replica',
    cells: cells((c) => ({ tokens: data[c.dp].ids.map(tokLabel), target: data[c.dp].tgt.map(tokLabel) })),
  })

  push({
    kind: 'mod', name: 'Embedding', sub: 'vocab-parallel — a rank contributes 0 for rows it does not own',
    edge: 'ids', tag: 'partial',
    cells: cells((c, m) => {
      const mine = [...new Set(data[c.dp].ids)].filter((v) => m.vocab.includes(v))
      return {
        params: mine.length
          ? mine.map((v) => ({ n: `E[${v}]`, v: model.E[v] }))
          : [{ n: `rows ${m.lo}–${m.hi}`, v: '' }],
        note: mine.length ? undefined : 'owns no row used by this sequence',
        tensors: [
          { n: 'x (partial)', cols: data[c.dp].ids.map(tokLabel), s: 'partial',
            d: data[c.dp].ids.map((v) => (m.vocab.includes(v) ? model.E[v] : 0)) },
        ],
      }
    }),
  })
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce', tag: 'TP', edge: 'partial embeddings',
      cells: cells((c, m) => ({ tensors: [T('x', m.rep.out.x, { s: 'done' })] })),
    })

  push({
    kind: 'op', name: 'q, k, v = w·x', sub: 'column-parallel: a rank owns whole attention heads', edge: 'x',
    cells: cells((c, m) => ({
      params: m.heads.flatMap((i) => [{ n: `h${i}.wv`, v: model.heads[i].wv }, { n: `h${i}.wk`, v: model.heads[i].wk }]),
      tensors: m.heads.map((i) => T(`v[h${i}]`, m.rep.out.perHead[i].v)),
    })),
  })
  push({
    kind: 'op', name: 'a = causal_softmax(q·k) · v', sub: 'token 0 attends to itself only', edge: 'q, k, v',
    cells: cells((c, m) => ({ tensors: m.heads.map((i) => T(`a[h${i}]`, m.rep.out.perHead[i].a)) })),
  })
  push({
    kind: 'op', name: 'o = Σ wo·a over my heads', sub: 'row-parallel — every rank holds a partial sum',
    edge: 'a', tag: 'partial',
    cells: cells((c, m) => ({
      params: m.heads.map((i) => ({ n: `h${i}.wo`, v: model.heads[i].wo })),
      tensors: [T('o (partial)', m.rep.out.x.map((_, t) => m.heads.reduce((s, i) => s + m.rep.out.perHead[i].o[t].data, 0)), { s: 'partial' })],
    })),
  })
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce', tag: 'TP', edge: 'partial outputs',
      cells: cells((c, m) => ({ tensors: [T('attn', m.rep.out.attn, { s: 'done' })] })),
    })
  push({
    kind: 'mod', name: 'h = x + attn', sub: 'residual — replicated across the TP group', edge: 'attn',
    cells: cells((c, m) => ({ tensors: [T('h', m.rep.out.h)] })),
  })

  if (!isMoe) {
    push({
      kind: 'op', name: 'w2 · relu(w1·h)', sub: 'column-parallel up, row-parallel down', edge: 'h', tag: 'partial',
      cells: cells((c, m) => ({
        params: m.cols.flatMap((j) => [{ n: `c${j}.w1`, v: model.mlp[j].w1 }, { n: `c${j}.w2`, v: model.mlp[j].w2 }]),
        tensors: [T('mlp (partial)', m.rep.out.h.map((_, t) => m.cols.reduce((s, j) => s + m.rep.out.perCol[j].out[t].data, 0)), { s: 'partial' })],
      })),
    })
    if (cfg.tp > 1)
      push({
        kind: 'comm', group: 'tp', name: 'all-reduce', tag: 'TP', edge: 'partial outputs',
        cells: cells((c, m) => ({ tensors: [T('mlp', m.rep.out.ffn, { s: 'done' })] })),
      })
    push({ kind: 'mod', name: 'y = h + mlp', edge: 'mlp', cells: cells((c, m) => ({ tensors: [T('y', m.rep.out.y)] })) })
  } else {
    push({
      kind: 'mod', name: 'router', sub: 'logit_e = r_e·h + b_e — replicated, so every rank routes identically',
      edge: 'h', tag: 'top-1',
      cells: cells((c, m) => ({
        params: model.router.r.map((v, e) => ({ n: `r${e}`, v })),
        tensors: [
          T('gate', m.rep.out.route.map((r) => r.gate)),
          { n: 'expert', d: m.rep.out.route.map((r) => r.sel), fmt: 'int', s: 'pick' },
        ],
      })),
    })
    push({
      kind: 'comm', group: 'ep', name: 'dispatch all-to-all', tag: 'EP · tokens to their expert', edge: 'routed tokens',
      cells: cells((c, m) => {
        const mine = []
        runs.forEach((r, dp) => r.out.route.forEach((rt, t) => {
          if (m.experts.includes(rt.sel)) mine.push({ lbl: `t${dp * S + t}`, v: r.out.h[t].data })
        }))
        return mine.length
          ? { tensors: [{ n: `h @ E${m.experts.join(',')}`, d: mine.map((x) => x.v), cols: mine.map((x) => x.lbl), s: 'done' }] }
          : { state: 'absent', note: 'hosts no expert this step' }
      }),
    })
    push({
      kind: 'op', name: 'expert forward', sub: 'TP still splits the columns inside each expert', edge: 'my tokens', tag: 'partial',
      cells: cells((c, m) => {
        const out = []
        runs.forEach((r, dp) => r.out.route.forEach((rt, t) => {
          if (!m.experts.includes(rt.sel)) return
          const hv = r.out.h[t].data
          out.push({ lbl: `t${dp * S + t}`, v: m.cols.reduce((s, j) => s + model.experts[rt.sel][j].w2 * Math.max(0, model.experts[rt.sel][j].w1 * hv), 0) })
        }))
        return out.length
          ? {
              params: m.experts.flatMap((e) => m.cols.map((j) => ({ n: `E${e}c${j}.w1`, v: model.experts[e][j].w1 }))),
              tensors: [{ n: 'out (partial)', d: out.map((x) => x.v), cols: out.map((x) => x.lbl), s: 'partial' }],
            }
          : { state: 'absent', note: 'no expert here' }
      }),
    })
    if (cfg.tp > 1)
      push({
        kind: 'comm', group: 'tp', name: 'all-reduce', tag: 'TP · inside the expert', edge: 'partial expert out',
        cells: cells((c, m) => {
          const out = []
          runs.forEach((r, dp) => r.out.route.forEach((rt, t) => {
            if (m.experts.includes(rt.sel)) out.push({ lbl: `t${dp * S + t}`, v: r.out.route[t].out.data })
          }))
          return out.length ? { tensors: [{ n: 'out', d: out.map((x) => x.v), cols: out.map((x) => x.lbl), s: 'done' }] } : { state: 'absent' }
        }),
      })
    push({
      kind: 'comm', group: 'ep', name: 'combine all-to-all', tag: 'EP · back to the token owner', edge: 'expert outputs',
      cells: cells((c, m) => ({ tensors: [T('out', m.rep.out.route.map((r) => r.out), { s: 'done' })] })),
    })
    push({
      kind: 'mod', name: 'y = h + gate·out', sub: 'the gate is the only path by which the router learns', edge: 'out',
      cells: cells((c, m) => ({ tensors: [T('y', m.rep.out.y)] })),
    })
  }

  push({
    kind: 'mod', name: 'logits z = y·U', sub: 'vocab-parallel — each rank owns a slice of the logit columns',
    edge: 'y', tag: 'partial',
    cells: cells((c, m) => {
      const S2 = m.rep.out.z.length
      const mx = [], se = [], tg = []
      for (let t = 0; t < S2; t++) {
        const vals = m.vocab.map((v) => m.rep.out.z[t][v].data)
        const mm = Math.max(...vals)
        mx.push(mm)
        se.push(vals.reduce((a, x) => a + Math.exp(x - mm), 0))
        const tv = data[c.dp].tgt[t]
        tg.push(m.vocab.includes(tv) ? m.rep.out.z[t][tv].data : NaN)
      }
      const cols = data[c.dp].ids.map((_, t) => `tok${t}`)
      return {
        params: [{ n: `cols ${m.lo}–${m.hi}`, v: '' }],
        tensors: [
          { n: 'local max', d: mx, cols, s: 'partial' },
          { n: 'local Σexp', d: se, cols, s: 'partial' },
          { n: 'z[target]', d: tg, cols, s: 'partial' },
        ],
      }
    }),
  })
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce ×3', tag: 'TP · max, Σexp, target',
      sub: 'three scalars per token — never the logits themselves', edge: 'local logits',
      cells: cells((c, m) => ({ tensors: [T('loss/token', m.rep.out.perTok, { s: 'done' })] })),
    })
  push({
    kind: 'io', name: 'L', tag: 'local loss', edge: 'cross-entropy',
    cells: cells((c, m) => ({ tensors: [{ n: 'L', d: [m.rep.out.loss.data], s: 'done' }] })),
  })

  const B = backward(cfg, model, runs, cells, isMoe, data)
  markBlock(F, ['q, k, v'], ['y = h'])
  markBlock(B, ['all-reduce', 'MLP backward', 'MoE split'], ['Attention backward'])
  // residual paths: x forks before attention and rejoins at h; h forks before
  // the FFN and rejoins at y. Both are inside the layer band.
  const mark = (steps, key, forkAt, joinAt) => {
    const f = steps.findIndex((s) => s.name.startsWith(forkAt))
    const j = steps.findIndex((s) => s.name.startsWith(joinAt))
    if (f >= 0) (steps[f].forks ||= []).push(key)
    if (j >= 0) (steps[j].joins ||= []).push(key)
  }
  mark(F, 'x', 'q, k, v', 'h = x + attn')
  mark(F, 'h', 'h = x + attn', 'y = h')
  const zone = isMoe ? 'MoE' : 'MLP'
  if (isMoe) {
    markZone(F, zone, ['router', 'dispatch all-to-all', 'expert forward', 'combine all-to-all'])
    markZone(B, zone, ['MoE split', 'router backward', 'dispatch-grads', 'expert backward', 'combine-grads'])
  } else {
    markZone(F, zone, ['w2 · relu'])
    markZone(B, zone, ['MLP backward'])
  }
  return { forward: F, backward: B, runs, data }
}

// =============================== BACKWARD ===============================
function backward(cfg, model, runs, cells, isMoe, data) {
  const B = []
  const push = (s) => B.push({ id: `b${B.length}`, ...s })
  const params = paramList(model)

  push({
    kind: 'io', name: 'L', tag: 'seed — dL/dL = 1',
    sub: 'backward starts where forward ended: the scalar loss, seeded with a gradient of one',
    cells: cells((c, m) => ({
      tensors: [
        { n: 'L', d: [m.rep.out.loss.data], s: 'done' },
        { n: 'dL', d: [m.rep.out.loss.grad], s: 'done' },
      ],
    })),
  })
  push({
    kind: 'op', name: 'cross-entropy backward', sub: 'dz = (softmax(z) − onehot) / S', edge: 'dL',
    cells: cells((c, m) => ({
      tensors: [
        { n: 'dz[target]', cols: data[c.dp].tgt.map(tokLabel), s: 'partial',
          d: data[c.dp].tgt.map((v, t) => (m.vocab.includes(v) ? m.rep.out.z[t][v].grad : NaN)) },
        Tg('dL/dloss_tok', m.rep.out.perTok),
      ],
    })),
  })
  push({
    kind: 'mod', name: 'LM head backward', sub: 'dU = Σ dz·y — each rank owns its own column, no collective', edge: 'dz',
    cells: cells((c, m) => {
      const t = [...new Set(data[c.dp].tgt)].filter((v) => m.vocab.includes(v))
      return {
        tensors: [
          t.length
            ? { n: 'dU[target]', d: t.map((v) => m.rep.L[`head.U${v}`].grad), cols: t.map(tokLabel) }
            : { n: `dU  Σ|·| over cols ${m.lo}–${m.hi}`,
                d: [m.vocab.reduce((a, v) => a + Math.abs(m.rep.L[`head.U${v}`].grad), 0)] },
        ],
      }
    }),
  })
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce', tag: 'TP · dy', edge: 'partial dy',
      cells: cells((c, m) => ({ tensors: [Tg('dy', m.rep.out.y, { s: 'done' })] })),
    })

  if (!isMoe) {
    push({
      kind: 'mod', name: 'MLP backward', sub: 'dw2 = Σ dy·relu(u),  dw1 = Σ du·h', edge: 'dy',
      cells: cells((c, m) => ({
        tensors: [{ n: 'dw1', d: m.cols.map((j) => m.rep.L[`mlp.c${j}.w1`].grad), cols: m.cols.map((j) => `c${j}`) },
                  { n: 'dw2', d: m.cols.map((j) => m.rep.L[`mlp.c${j}.w2`].grad), cols: m.cols.map((j) => `c${j}`) }],
      })),
    })
  } else {
    push({
      kind: 'mod', name: 'MoE split', sub: 'd_out = dy·gate   ·   d_gate = dy·out', edge: 'dy', tag: 'two paths',
      cells: cells((c, m) => ({
        tensors: [T('d_out', m.rep.out.route.map((r, t) => m.rep.out.y[t].grad * r.gate.data)),
                  T('d_gate', m.rep.out.route.map((r, t) => m.rep.out.y[t].grad * r.out.data))],
      })),
    })
    push({
      kind: 'mod', name: 'router backward', sub: 'trained only through the gate scalar', edge: 'd_gate',
      cells: cells((c, m) => ({
        tensors: [{ n: 'dr', d: model.router.r.map((_, e) => m.rep.L[`router.r${e}`].grad), cols: model.router.r.map((_, e) => `e${e}`) },
                  { n: 'db', d: model.router.b.map((_, e) => m.rep.L[`router.b${e}`].grad), cols: model.router.b.map((_, e) => `e${e}`) }],
      })),
    })
    push({
      kind: 'comm', group: 'ep', name: 'dispatch-grads all-to-all', tag: 'EP', edge: 'd_out',
      cells: cells((c, m) => (m.experts.length ? { note: `receives d_out for E${m.experts.join(',')}`, state: 'hot' } : { state: 'absent', note: 'sends only' })),
    })
    push({
      kind: 'mod', name: 'expert backward', tag: '×1/DP applied locally',
      sub: 'already full-batch — the dispatch all-to-all was the reduction', edge: 'd_out',
      cells: cells((c, m) => (m.experts.length ? {
        tensors: m.experts.map((e) => ({
          n: `dw2[E${e}]`, cols: m.cols.map((j) => `c${j}`),
          d: m.cols.map((j) => avg(runs, `experts.e${e}.c${j}.w2`)),
        })),
      } : { state: 'absent' })),
    })
    push({
      kind: 'comm', group: 'ep', name: 'combine-grads all-to-all', tag: 'EP', edge: 'dh_exp',
      cells: cells(() => ({ note: 'dh_exp returns to the token owner', state: 'hot' })),
    })
  }
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce + residual', tag: 'TP · dh', edge: 'partial dh',
      cells: cells((c, m) => ({ tensors: [Tg('dh', m.rep.out.h, { s: 'done' })] })),
    })
  push({
    kind: 'mod', name: 'Attention backward', sub: 'dwo = Σ dh·a,  dwv = Σ dv·x', edge: 'dh',
    cells: cells((c, m) => ({
      tensors: [{ n: 'dwo', d: m.heads.map((i) => m.rep.L[`attn.h${i}.wo`].grad), cols: m.heads.map((i) => `h${i}`) },
                { n: 'dwv', d: m.heads.map((i) => m.rep.L[`attn.h${i}.wv`].grad), cols: m.heads.map((i) => `h${i}`) }],
    })),
  })
  if (cfg.tp > 1)
    push({
      kind: 'comm', group: 'tp', name: 'all-reduce + residual', tag: 'TP · dx', edge: 'partial dx',
      cells: cells((c, m) => ({ tensors: [Tg('dx', m.rep.out.x, { s: 'done' })] })),
    })
  push({
    kind: 'mod', name: 'Embedding backward', sub: 'vocab-parallel — scatter into my own rows, no collective', edge: 'dx',
    cells: cells((c, m) => {
      const used = [...new Set(data[c.dp].ids)].filter((v) => m.vocab.includes(v))
      return {
        tensors: [
          used.length
            ? { n: 'dE (rows used)', d: used.map((v) => m.rep.L[`embed.E${v}`].grad), cols: used.map(tokLabel) }
            : { n: `dE  rows ${m.lo}–${m.hi}`, d: [0] },
        ],
        state: used.length ? undefined : 'hot',
        note: used.length
          ? `${V - used.length} of ${V} rows get exactly 0`
          : 'every row this rank owns gets exactly 0 — the sequence never used one',
      }
    }),
  })

  if (cfg.dp > 1 && cfg.zero > 0)
    push({
      kind: 'comm', group: 'dp', name: cfg.zero >= 2 ? 'reduce-scatter (bucketed)' : 'reduce-scatter',
      tag: `DP · ZeRO-${cfg.zero}`, edge: 'local gradients',
      sub: cfg.zero >= 2 ? 'each bucket is reduced as backward finishes it, then freed everywhere else' : 'averaged gradients, one slice per rank',
      cells: cells((c) => shardSummary(cfg, model, c, params, runs, 'grad')),
    })
  else if (cfg.dp > 1)
    push({
      kind: 'comm', group: 'dp', name: 'all-reduce', tag: 'DP · DDP', edge: 'local gradients',
      cells: cells(() => ({ note: 'every rank receives the full averaged gradient' })),
    })

  push({
    kind: 'opt', name: 'optimizer step', tag: cfg.zero > 0 ? 'owned shard only' : 'full, redundantly',
    sub: 'SGD + momentum, lr = 0.1 — on step 1 momentum equals the gradient', edge: 'averaged gradient',
    cells: cells((c) => shardSummary(cfg, model, c, params, runs, 'weight')),
  })
  if (cfg.dp > 1 && cfg.zero > 0 && cfg.zero < 3)
    push({
      kind: 'comm', group: 'dp', name: 'all-gather', tag: 'DP · before the next forward', edge: 'updated shards',
      cells: cells(() => ({ note: 'full parameter buffer restored on every rank' })),
    })
  else if (cfg.zero === 3)
    push({
      kind: 'opt', name: 'no all-gather', tag: 'ZeRO-3', edge: 'updated shards',
      sub: 'parameters stay sharded; the next forward gathers each unit right before it is used',
      cells: cells(() => ({ note: 'shards stay put' })),
    })

  return B
}


/**
 * A rank's ZeRO shard, summarised. Listing every owned parameter is useless
 * once the vocabulary is real, so the big vocab-sharded groups are reported as
 * counts and the small groups — the ones you can actually read — as values.
 */
function shardSummary(cfg, model, c, params, runs, mode) {
  const mine = owned(cfg, model, c, params)
  const groups = new Map()
  for (const f of mine) {
    const g = params.find((p) => p.fqn === f).group
    if (!groups.has(g)) groups.set(g, [])
    groups.get(g).push(f)
  }
  const pills = [...groups].map(([g, fs]) => ({ n: g, v: `×${fs.length}` }))
  const tensors = []
  for (const [g, fs] of groups) {
    if (fs.length > 5) continue
    tensors.push({
      n: mode === 'grad' ? `avg grad · ${g}` : `updated · ${g}`,
      cols: fs.map(short),
      d: fs.map((f) => (mode === 'grad' ? avg(runs, f) : runs[0].L[f].data - 0.1 * avg(runs, f))),
      s: 'done',
    })
  }
  const big = [...groups].filter(([, fs]) => fs.length > 5).map(([g, fs]) => `${g} ×${fs.length}`)
  return {
    params: pills,
    tensors,
    note: `${mine.length} of ${params.length} parameters` + (big.length ? ` · ${big.join(', ')} not shown` : ''),
  }
}

const avg = (runs, fqn) => runs.reduce((a, r) => a + r.L[fqn].grad, 0) / runs.length
const short = (f) => f.split('.').slice(-2).join('.')

function owned(cfg, model, c, params) {
  return params.filter((p) => holds(cfg, p, c)).filter((p) => {
    if (cfg.zero === 0) return true
    const deg = p.group === 'expert' ? expertDP(cfg) : cfg.dp
    if (deg === 1) return true
    const peers = allRanks(cfg).filter((o) => o.tp === c.tp && holds(cfg, p, o)).sort((a, b) => a.dp - b.dp)
    const idx = params.filter((q) => (q.group === 'expert') === (p.group === 'expert')).findIndex((q) => q.fqn === p.fqn)
    return peers[idx % peers.length]?.rank === c.rank
  }).map((p) => p.fqn)
}
