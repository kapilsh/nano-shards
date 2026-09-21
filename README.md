# nano-shards

*Exact tensors, on every GPU.*

An interactive view of a **one-layer** dense or MoE transformer sharded across GPUs under
**DDP, ZeRO-1/2/3, TP and EP** — showing not just where each tensor lives, but **what its value is
on every rank, at every step of forward and backward**.

Hidden dimension is 1 and the sequence is 2 tokens, so every weight and every activation is a single
number you can check by hand. Nothing on screen is transcribed: the app runs the model for real on a
scalar reverse-mode autograd tape and routes each value to whichever rank owns it.

Everything is client-side (React 19 + Vite + zustand) and deploys to GitHub Pages from `docs/`.

## Views

| View | What it shows |
| --- | --- |
| **Layer flow** | Top to bottom through the layer, forward or backward. Every node carries the exact tensor on each rank, drawn as a small table; collectives appear inline where they fire, `partial` values are marked until a collective completes them, and the residual paths are drawn as real split connections with their own arrowheads. |
| **Device mesh** | Every rank as a cell. Lenses highlight its TP / DP / EP / expert-DP group, and call out when expert-DP collapses to 1. |
| **Memory** | Per-rank params / grads / optimizer scalars under the current ZeRO stage, plus the activation column ZeRO never touches. |
| **Communication** | Every collective in one step, in firing order, with its group and how many such groups run concurrently. |

## Why the numbers are trustworthy

`npm run verify` differentiates the whole model by central finite differences and compares against
the autograd tape, over both the dense and the MoE model and every parameter — including the router,
whose only gradient path is through the top-1 gate scalar:

```
72/72 gradients match finite differences (tol 0.00002)
(vocab rows: all rows used by the batch, plus every 17th as a spot check)
```

The vocab rows are sampled only to keep the check fast; an unsampled sweep is 432/432.

## The model

```
ids ──► Embedding E ──► x
x   ──► Attention (per-head wq, wk, wv; causal softmax; out-proj wo)
h   = x + attn(x)                                   ← residual
y   = h + FFN(h)          FFN = Σ w2·relu(w1·h)       (dense)
                              = gate · Expert_sel(h)  (MoE, top-1 router)
logits = y · U
L   = mean CE over tokens, then mean over DP replicas
```

Attention and FFN weights are chosen so the 2-way head and column split reproduces the unfactored
model exactly — head 0 contributes `.5v`, head 1 contributes `.25·2v` — which makes TP sharding
verifiable by inspection: **`h` and `y` must not change when you turn TP on.**

## Input

The batch is deliberately small — **2 replicas × 2 tokens** — but the token ids are drawn from a
**100-entry vocabulary**, so vocab-parallel sharding is a real split rather than a token gesture.

A seed drives both the batch and the embedding / LM-head rows, so an example is reproducible and
`new` redraws the whole thing coherently. Click any token to reroll just that one.

With the default seed and TP=2, GPU0 owns vocab rows 0–49 and the batch uses none of them, so its
embedding contribution is `[0, 0]` and its `dE` is exactly zero across all 50 rows — the sparsity a
two-token vocabulary cannot show.

## Sharding

| Axis | Splits |
| --- | --- |
| TP | attention heads, MLP columns, vocabulary (embedding + LM head); the MoE router is replicated |
| EP | experts across DP ranks; `expert-DP = DP / EP` |
| ZeRO-1 | optimizer state across the DP group |
| ZeRO-2 | + gradients |
| ZeRO-3 | + parameters (peak adds the largest gathered unit back) |

World size is capped at **4 ranks** — more columns stop being readable, and every distinction worth
showing is reachable within it. In particular `dp4 tp1 ep2` gives **expert-DP = 2**, where experts
are replicated and get their own reduce-scatter over a differently shaped group than the dense
parameters; `dp2 tp2 ep2` gives expert-DP = 1, where each expert lives on exactly one rank and ZeRO
has nothing to shard it across.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173/nano-shards/
npm run verify   # gradient check against finite differences
npm run lint
npm run build    # -> docs/
```
