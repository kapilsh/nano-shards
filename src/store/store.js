import { create } from 'zustand'
import { denseModel, moeModel, makeData, DEFAULT_SEED, V } from '../lib/model.js'

export const MAX_WORLD = 4

const base = { dp: 2, tp: 2, ep: 2, zero: 1, ckpt: false, nHeads: 2, nCols: 2, nExperts: 2 }

export const useStore = create((set, get) => ({
  kind: 'moe',
  cfg: base,
  seed: DEFAULT_SEED,
  data: makeData(DEFAULT_SEED, base.dp),
  view: 'flow',
  mode: 'forward',
  rank: 0,
  lens: 'tp',
  model: () => (get().kind === 'moe' ? moeModel(get().seed) : denseModel(get().seed)),

  setKind: (kind) =>
    set((s) => ({ kind, cfg: { ...s.cfg, ep: kind === 'moe' ? s.cfg.ep : 1 } })),
  setCfg: (patch) =>
    set((s) => {
      const cfg = { ...s.cfg, ...patch }
      // The app tops out at MAX_WORLD ranks: more columns than that stop being
      // readable, and every distinction worth showing is reachable within it.
      if ('dp' in patch && cfg.dp * cfg.tp > MAX_WORLD) cfg.tp = MAX_WORLD / cfg.dp
      if ('tp' in patch && cfg.dp * cfg.tp > MAX_WORLD) cfg.dp = MAX_WORLD / cfg.tp
      if (cfg.ep > cfg.dp) cfg.ep = cfg.dp
      if (s.kind !== 'moe') cfg.ep = 1
      const world = cfg.dp * cfg.tp
      return { cfg, rank: Math.min(s.rank, world - 1), data: resize(s.data, cfg.dp, s.seed) }
    }),
  setView: (view) => set({ view }),
  setMode: (mode) => set({ mode }),
  setRank: (rank) => set({ rank }),
  setLens: (lens) => set({ lens }),
  // A new seed redraws both the batch and the embedding / LM-head rows, so the
  // whole example changes together and stays reproducible.
  newSeed: () =>
    set((s) => {
      const seed = (Math.random() * 1e9) | 0
      return { seed, data: makeData(seed, s.cfg.dp) }
    }),
  reset: () => set((s) => ({ seed: DEFAULT_SEED, data: makeData(DEFAULT_SEED, s.cfg.dp) })),
  rollToken: (i, t) =>
    set((s) => {
      const data = s.data.map((d, k) => {
        if (k !== i) return d
        const ids = d.ids.map((x, j) => (j === t ? Math.floor(Math.random() * V) : x))
        return { ids, tgt: ids.map((_, j) => (j + 1 < ids.length ? ids[j + 1] : ids[0])) }
      })
      return { data }
    }),
}))

const resize = (data, dp, seed) =>
  Array.from({ length: dp }, (_, i) => data[i] ?? makeData(seed + i, 1)[0])
