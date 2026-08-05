---
layout: ../../layouts/ResearchLogLayout.astro
title: residual saes and trivial features
week: 6
date: 2026-07-12
blurb: "the central experiment — does an SAE trained on h − P[token] have fewer trivial features? answer: no, but finding that out honestly was the work."
---

This was the week the project found its spine. The question: SAE dictionaries are cluttered with "trivial" single-token detectors that arguably reflect tokenization rather than computation. If my linear map $P$ captures the token-static part of layer 13, then an SAE trained on the residual $r = h - P[\text{token}]$ should spend its capacity on *contextual* structure instead — the same motivation as Tokenized SAEs ([Dooms & Wilhelm 2025](https://arxiv.org/abs/2502.17332)), which peel token reconstruction off with per-token biases.

## experiment: matched full vs residual SAEs

**setup:** `extract.py` cached ~50M tokens of gemma-2-2b layer-13 activations + token ids from OpenWebText. Trained two SAELens **BatchTopK** SAEs, matched at 16k features and $k = 64$: one on the full activation $h$, one on the residual $r$. A sanity check on the way in: the map explains **55%** of $\operatorname{Var}(h)$ at layer 13, not the ~66% an earlier estimate suggested (removing the BOS token was the suspect; the training logs settle it at 0.554).

**results (reconstruction accounting, `eval_fvu.py`):** the full SAE gets FVU 0.1426 on $h$; the residual SAE gets FVU 0.32 on $r$ — but the fair comparison is the *composite* system (map + residual SAE) on raw $h$: **0.1445 vs 0.1426**. Offloading the token-predictable part buys **zero** reconstruction at matched dictionary/k. Residual variance is ~2.3× harder to reconstruct per unit variance — the token-predictable structure is precisely the *easy* part for a sparse code.

## experiment: measuring triviality (and getting fooled twice)

**setup:** `eval_trivial.py` streams 4M tokens and computes per-feature triviality stats — modal-word fraction, distinct words in top activations, entropy-based effective #words — plus a neuronpedia-style `dashboard.py` for eyeballing features side by side, and a matched-pairs "churn" study comparing full↔resid feature sets. Metrics computed three ways: at **peak** activations (top-K), **uniformly** sampled, and **activation-weighted**.

**results:** the first pass said ~16% of full-SAE features are single-token and that the residual SAE differed — both wrong, for instructive reasons:

- **peak-sampling bias:** features look single-token at their strongest activations but are broad in typical firing. Measured across the whole activation range, trivial features are **~1%**, not ~16% — the peak metric inflates triviality ~16×.
- **firing-frequency confound:** rare features are intrinsically more trivial, and the residual dictionary simply skews rarer (19% vs 11% of features below ~632 firings). The aggregate "resid is more trivial" effect, and a ~22% feature churn, both vanish when compared within frequency bins.

**Net result: NULL — residualization does not reduce triviality.** Also worth keeping: "single-token" ≠ "useless" (a "health" feature that predicts "insurance" is token-anchored and still doing something). The durable output is the methodology: always measure activation-weighted, always compare within frequency bins.

![frequency × unique-word-count heatmap, full / resid / difference](/research-log/week6-freq-vs-words-heatmap.png)

![activation-weighted unique-word distribution — the principled metric](/research-log/week6-unique-words-weighted.png)

## next week

- ideas for rescuing the effect: an earlier layer (L1, where $R^2 \approx 0.85$), a nonlinear map instead of linear, or rethinking whether "fewer single-token features" is the right objective at all
- mentor suggestion to try: don't freeze the map — train it *jointly* with the SAE, end-to-end on $h$
