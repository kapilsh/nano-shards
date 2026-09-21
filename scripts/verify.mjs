// Central finite differences against the autograd tape.
// If this passes, every gradient the app displays is the real gradient.
import { denseModel, moeModel, leaves, forward, runStep, DATA } from '../src/lib/model.js'

const EPS = 1e-6
const TOL = 2e-5

function lossOf(m) {
  const L = leaves(m)
  const reps = DATA.map((d) => forward(m, L, d))
  return reps.reduce((a, r) => a + r.loss.data, 0) / reps.length
}

function bump(m, fqn, d) {
  const c = structuredClone(m)
  const [g, ...rest] = fqn.split('.')
  if (g === 'embed') c.E[+rest[0].slice(1)] += d
  else if (g === 'head') c.U[+rest[0].slice(1)] += d
  else if (g === 'attn') c.heads[+rest[0].slice(1)][rest[1]] += d
  else if (g === 'mlp') c.mlp[+rest[0].slice(1)][rest[1]] += d
  else if (g === 'router') c.router[rest[0][0]][+rest[0].slice(1)] += d
  else if (g === 'experts') c.experts[+rest[0].slice(1)][+rest[1].slice(1)][rest[2]] += d
  else throw new Error('unknown param ' + fqn)
  return c
}

let fail = 0, n = 0
for (const [name, make] of [['dense', denseModel], ['moe', moeModel]]) {
  const m = make()
  const { grads } = runStep(m)
  console.log(`\n== ${name} ==`)
  // Only a couple of the 100 vocab rows are touched by the batch; check every
  // non-vocab parameter plus a sample of the vocab rows, including used ones.
  const used = new Set(DATA.flatMap((d) => [...d.ids, ...d.tgt]))
  const keep = (fqn) => {
    const i = fqn.match(/^(embed\.E|head\.U)(\d+)$/)
    return !i || used.has(+i[2]) || +i[2] % 17 === 0
  }
  for (const fqn of Object.keys(grads).filter(keep)) {
    const fd = (lossOf(bump(m, fqn, EPS)) - lossOf(bump(m, fqn, -EPS))) / (2 * EPS)
    const err = Math.abs(fd - grads[fqn])
    n++
    const ok = err < TOL
    if (!ok) fail++
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${fqn.padEnd(22)} autograd ${grads[fqn].toFixed(6).padStart(10)}   fd ${fd.toFixed(6).padStart(10)}`,
    )
  }
}
console.log(`\n${n - fail}/${n} gradients match finite differences (tol ${TOL})`)
console.log('(vocab rows: all rows used by the batch, plus every 17th as a spot check)')
process.exit(fail ? 1 : 0)
