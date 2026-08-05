---
layout: ../../layouts/ResearchLogLayout.astro
title: day-of-week geometry
week: 2
date: 2026-06-14
blurb: PCA on day-of-week activations in gemma-2-2b — class means reveal a ring, so I started fitting explicit circular probes.
---

## experiment: PCA on days of the week

**motivation:** [Gurnee & Tegmark 2023](https://arxiv.org/abs/2310.02207) show LLMs hold linear representations of space and time, and [Engels et al. 2024](https://arxiv.org/abs/2405.14860) show that days of the week are the canonical case where the representation is *not* a line but a circle — an actual ring used for modular arithmetic. Days are a clean, closed, 7-element concept: an ideal test bed for asking what the geometry of activations looks like, and eventually whether that structure lives in the raw activation $h$ or in the in-context residual $r = h - P[\text{token}]$.

**setup:** `days.py` / `days2.py` collect gemma-2-2b activations at day-of-week tokens in context (targeting ~200 hits per day across the target layers); `pca.py` projects them onto principal components. Variants: PC1–PC2 vs PC2–PC3 views, plots with every sample vs **class means only**, and arrows connecting consecutive days to make the ordering visible.

**results:** the full point clouds are noisy — individual occurrences of "Monday" scatter widely. But plotting class means only strips the within-class noise, and a ring-like arrangement of the seven days shows up, with consecutive days adjacent. Choice of PC pair matters (the circle doesn't live in the top two components at every layer). Getting here took some iteration on normalization and rescaling; the means-only plots were the trick that made the structure legible.

![day-of-week tokens projected onto PC1–PC2](/research-log/week2-days-pca-pc12.png)

![class means only — the ring becomes visible](/research-log/week2-days-pca-means.png)

## experiment: first circular probes (started)

**motivation:** a PCA picture is suggestive, not a claim. Following the multi-dimensional-feature framing of [Engels et al. 2024](https://arxiv.org/abs/2405.14860), the honest test is to *fit* a circle: find a 2D subspace where days lie at fixed angles.

**setup:** started `circular_probe.py` at the end of the week — decomposing activations into $h$, $\hat{h} = P[\text{token}]$, and the residual $r = h - \hat{h}$, and fitting a probe to each to see where the circular structure actually lives. First version uses least squares; produced initial MSE comparisons across the three targets.

**results:** just getting started — the decomposition runs and the first MSE plots exist, but no conclusions yet. Main output this week was the harness.

## next week

- finish the circular probes: least-squares vs gradient-descent fits, proper train/test split, real text (OpenWebText) instead of templates
- compare probe quality on $h$ vs $\hat{h}$ vs $r$ — is the ring in the token-static part or the contextual part?
