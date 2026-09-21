// Reverse-mode autograd over scalars.
//
// The whole point of nano-shards is that every number on screen is the real
// number, not a transcription. Hidden dim is 1, so a scalar tape is enough to
// run the model for real and get exact gradients with no hand-derivation.

export class V {
  constructor(data, children = [], op = '', label = '') {
    this.data = data
    this.grad = 0
    this._back = () => {}
    this._prev = children
    this.op = op
    this.label = label
  }

  static of(x, label = '') {
    return x instanceof V ? x : new V(x, [], 'const', label)
  }

  add(o) {
    o = V.of(o)
    const out = new V(this.data + o.data, [this, o], '+')
    out._back = () => {
      this.grad += out.grad
      o.grad += out.grad
    }
    return out
  }

  mul(o) {
    o = V.of(o)
    const out = new V(this.data * o.data, [this, o], '*')
    out._back = () => {
      this.grad += o.data * out.grad
      o.grad += this.data * out.grad
    }
    return out
  }

  sub(o) {
    return this.add(V.of(o).neg())
  }

  neg() {
    return this.mul(-1)
  }

  div(o) {
    o = V.of(o)
    const out = new V(this.data / o.data, [this, o], '/')
    out._back = () => {
      this.grad += out.grad / o.data
      o.grad += (-this.data / (o.data * o.data)) * out.grad
    }
    return out
  }

  exp() {
    const out = new V(Math.exp(this.data), [this], 'exp')
    out._back = () => {
      this.grad += out.data * out.grad
    }
    return out
  }

  log() {
    const out = new V(Math.log(this.data), [this], 'log')
    out._back = () => {
      this.grad += out.grad / this.data
    }
    return out
  }

  relu() {
    const out = new V(this.data > 0 ? this.data : 0, [this], 'relu')
    out._back = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad
    }
    return out
  }

  named(label) {
    this.label = label
    return this
  }

  backward() {
    const topo = []
    const seen = new Set()
    const build = (n) => {
      if (seen.has(n)) return
      seen.add(n)
      for (const c of n._prev) build(c)
      topo.push(n)
    }
    build(this)
    this.grad = 1
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._back()
    return this
  }
}

export const val = (x, label) => new V(x, [], 'leaf', label)
export const sum = (xs) => xs.reduce((a, b) => a.add(b), val(0))

export function softmax(xs) {
  const m = Math.max(...xs.map((x) => x.data))
  const e = xs.map((x) => x.sub(m).exp())
  const s = sum(e)
  return e.map((x) => x.div(s))
}

// Cross-entropy against a hard target, computed the numerically stable way
// (max, log-sum-exp, target logit) — the same three reductions a vocab-parallel
// loss has to all-reduce.
export function crossEntropy(logits, target) {
  const m = Math.max(...logits.map((l) => l.data))
  const lse = sum(logits.map((l) => l.sub(m).exp())).log().add(m)
  return lse.sub(logits[target])
}
