export function num(x, d = 3) {
  if (x === 0) return '0'
  if (!Number.isFinite(x)) return String(x)
  let s = x.toFixed(d)
  if (/\./.test(s)) s = s.replace(/0+$/, '').replace(/\.$/, '')
  return s.replace(/^(-?)0\./, '$1.')
}
export const arr = (xs, d = 3) => '[' + xs.map((x) => num(x, d)).join(', ') + ']'
export const vals = (vs, d = 3) => arr(vs.map((v) => (typeof v === 'number' ? v : v.data)), d)
export const grads = (vs, d = 3) => arr(vs.map((v) => v.grad), d)
export const bytes = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`)
