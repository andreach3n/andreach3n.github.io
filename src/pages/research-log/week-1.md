---
layout: ../../layouts/ResearchLogLayout.astro
title: training linear maps
week: 1
date: 2026-06-07
blurb: built the flipped-tuned-lens pipeline — cache gemma-2-2b activations, train token→activation linear maps, and map out where they fail.
---

The project starts with one object: a linear map $P$ that predicts a layer's residual-stream activation directly from the static token embedding — a "flipped" tuned lens. The [tuned lens](https://arxiv.org/abs/2303.08112) ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112)) trains an affine probe from each layer's hidden state *to the output logits*; here the direction is reversed: embedding → hidden state. If a big chunk of the residual stream is just a linear image of token identity, then that part carries no in-context computation, and the interesting signal is exactly the residual $r = h - P[\text{token}]$ that the map *can't* explain.

## experiment: training the linear maps

**motivation:** before asking anything about the residual, I need the maps themselves — one per layer, trained well enough that the reconstruction error is meaningful and not just an artifact of a bad fit.

**setup:** `extract.py` caches residual-stream activations from gemma-2-2b (layers 1, 5, 9, 13, 17, 21, 25) on streamed text, saving in chunks of ~100k tokens so memory doesn't blow up. `train.py` trains a linear map from the token embedding to each layer's activation. A lot of the week was making this actually run: fixing device placement, freeing RAM between chunks, and computing train loss the same way as test loss so the curves are comparable.

**results:** swept learning rates and settled on $\text{lr} = 0.01$; extended training to all the target layers. The maps train stably, and by end of week there was a working pipeline from raw text → cached activations → fitted per-layer maps. Most bugs this week were plumbing, not science (wrong indexing in the eval loop, `float` casts, `d_vocab` mixups).

## experiment: where does the map fail?

**motivation:** the whole premise is that reconstruction *error* is the interesting signal. So the first real question is: which tokens and positions does the linear lens fail on?

**setup:** `analyze_err.py` decomposes the reconstruction error **by token** and **by position in the sequence**; `heatmap.py` makes per-layer error heatmaps, in normalized and unnormalized versions, over a set of test sentences.

**results:** error is far from uniform — it varies systematically by token and by position, which is exactly what you'd hope if some tokens are "mostly static" (well predicted from the embedding) and others are heavily rewritten by context. The normalized/unnormalized comparison mattered: raw error partly tracks activation norm, so normalization is needed before reading anything off the heatmaps.

![reconstruction error by token](/research-log/week1-error-by-token.png)

![reconstruction error by position in the text](/research-log/week1-error-by-position.png)

![per-layer error heatmap](/research-log/week1-layer-heatmap.png)

## next week

- zoom into one concrete concept instead of aggregate error: days of the week (started `days.py` at the very end of this week)
- PCA the day-of-week representations — is there structure in what the model adds on top of the token embedding?
- keep an eye on which layer to focus on going forward
