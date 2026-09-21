// Rank layout and process groups.
//
// world = dp * tp, laid out row-major: rank = dpIdx * tp + tpIdx.
// Every collective in the app names the group it runs on, and every group is
// derived here so the mesh view and the trace can never disagree.

export function worldSize(cfg) {
  return cfg.dp * cfg.tp
}

export function coords(cfg, rank) {
  const dp = Math.floor(rank / cfg.tp)
  const tp = rank % cfg.tp
  return { rank, dp, tp, ep: dp % cfg.ep, edp: Math.floor(dp / cfg.ep) }
}

export const allRanks = (cfg) => Array.from({ length: worldSize(cfg) }, (_, r) => coords(cfg, r))

/** Ranks sharing everything but the named axis. */
export const GROUPS = {
  tp: {
    label: 'TP',
    note: 'same tokens, different weight shards — fires inside every layer',
    of: (cfg, c) => allRanks(cfg).filter((o) => o.dp === c.dp).map((o) => o.rank),
  },
  dp: {
    label: 'DP',
    note: 'same weight shard, different tokens — where ZeRO lives',
    of: (cfg, c) => allRanks(cfg).filter((o) => o.tp === c.tp).map((o) => o.rank),
  },
  ep: {
    label: 'EP',
    note: 'MoE only — all-to-all moves tokens to the rank hosting their expert',
    of: (cfg, c) => allRanks(cfg).filter((o) => o.tp === c.tp).map((o) => o.rank),
  },
  edp: {
    label: 'expert-DP',
    note: 'ranks holding a replica of the same expert',
    of: (cfg, c) =>
      allRanks(cfg)
        .filter((o) => o.tp === c.tp && o.ep === c.ep)
        .map((o) => o.rank),
  },
}

/** expert-DP degree: how many ranks hold a copy of each expert. */
export const expertDP = (cfg) => cfg.dp / cfg.ep

/** Contiguous split of n items over `parts`, returning the slice for `idx`. */
export function slice(n, parts, idx) {
  const per = n / parts
  const lo = Math.round(idx * per)
  const hi = Math.round((idx + 1) * per)
  return Array.from({ length: hi - lo }, (_, i) => lo + i)
}

export const owns = (n, parts, idx, i) => slice(n, parts, idx).includes(i)
