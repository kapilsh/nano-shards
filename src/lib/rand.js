// mulberry32 — small, fast, and seeded, so a given seed always reproduces the
// same model and the same batch. "New" just picks a different seed.
export function rng(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const round2 = (x) => Math.round(x * 100) / 100
