# nano-shards

*Exact tensors, on every GPU.*

An interactive view of a **one-layer** dense or MoE transformer sharded across GPUs under
**DDP, ZeRO-1/2/3, TP and EP** — showing not just where each tensor lives, but **what its value is
on every rank, at every step of forward and backward**.

Hidden dimension is 1, the vocabulary is `{A, B}` and the sequence is 2 tokens, so every weight and
every activation is a single number you can check by hand. Nothing on screen is transcribed: the app
runs the model for real on a scalar autograd tape and derives the sharding from it.

Everything is client-side (React 19 + Vite + zustand) and deploys to GitHub Pages from `docs/`.

## Views

| View | What it shows |
| --- | --- |
| **Layer flow** | Top-to-bottom through the layer, forward or backward. Every node carries the exact tensor value on each rank, with collectives inline where they fire and a `partial` marker on values that still need one. |
| **Device mesh** | Every rank as a cell. Lenses highlight its TP / DP / EP / expert-DP group, and call out when expert-DP collapses to 1. |
| **Memory** | Per-rank params / grads / optimizer scalars under the current ZeRO stage, plus the activation column ZeRO never touches. |
| **Communication** | Every collective in one step, in firing order, with its group and how many such groups run concurrently. |

## Why the numbers are trustworthy

`npm run verify` differentiates the whole model by central finite differences and compares against the
autograd tape:

```
40/40 gradients match finite differences (tol 0.00002)
```

That runs over both the dense and the MoE model, every parameter, including the router — whose only
gradient path is through the top-1 gate scalar.

## The model

```
ids ──► Embedding E ──► x
x   ──► Attention (per-head wq, wk, wv; causal softmax; out-proj wo)
h   = x + attn(x)
y   = h + FFN(h)          FFN = Σ w2·relu(w1·h)        (dense)
                              = gate · Expert_sel(h)   (MoE, top-1 router)
logits = y · U
L   = mean CE over tokens, then mean over DP replicas
```

Weights are chosen so the 2-way head and column split reproduces the unfactored model exactly
(head 0 gives `.5v`, head 1 gives `.25·2v`), which makes TP sharding verifiable by inspection:
`h` and `y` must not change when you turn TP on.

## Sharding

| Axis | Splits |
| --- | --- |
| TP | attention heads, MLP columns, vocabulary (embedding + LM head); router is replicated |
| EP | experts across DP ranks; `expert-DP = DP / EP` |
| ZeRO-1 | optimizer state across the DP group |
| ZeRO-2 | + gradients |
| ZeRO-3 | + parameters (peak adds the largest gathered unit back) |

## Develop

```bash
npm install
npm run dev      # http://localhost:5173/nano-shards/
npm run verify   # gradient check
npm run lint
npm run build    # -> docs/
```
