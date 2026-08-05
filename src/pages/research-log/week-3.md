---
layout: ../../layouts/ResearchLogLayout.astro
title: circular probes and the geometry of truth
week: 3
date: 2026-06-21
blurb: fit explicit circular probes to the day representations, then pivoted to truth probes — train on cities, test on neg_cities.
---

## experiment: circular probes on real text

**motivation:** continuing from last week — testing whether the day-of-week ring is genuinely recoverable as a low-dimensional structure, per [Engels et al. 2024](https://arxiv.org/abs/2405.14860), rather than an artifact of eyeballing PCA plots.

**setup:** `circular_probe.py` / `circular_probe2.py` fit the probe two ways: closed-form least squares, and gradient descent with early stopping (settled on that after trying a fixed 1000 epochs). Switched the data source to OpenWebText so days appear in natural contexts, with a proper train/test split. Also made "PCA inside the probe subspace" plots — project all activations into the fitted probe plane and look at the day structure there, normalized and not.

**results:** the probe subspace shows the day structure much more cleanly than raw PCA — the ring is recoverable by an explicit fit, not just visible in means. Normalization again changed the picture enough that I kept both versions. This closed the days thread for now: the structure is real, and the machinery (fit a probe, look inside its subspace) carries over to the next concept.

![circular probe fit on OpenWebText activations](/research-log/week3-circular-probe.png)

![PCA inside the fitted probe subspace](/research-log/week3-probe-subspace.png)

## experiment: truth probes, train on cities → test on neg_cities

**motivation:** truth is the next, subtler contextual concept. [Marks & Tegmark 2023](https://arxiv.org/abs/2310.06824) (*The Geometry of Truth*) show a linear "truth" direction exists in LLM activations but is easily confounded — a probe can score high while keying on a surface correlate — and their negation/cross-dataset transfer test is the standard way to check whether a probe found *truth* or a shortcut. I wrote up notes on the paper (`geometry_of_truth_notes.md`), including mass-mean probing and their misalignment-from-correlational-inconsistency story, before reproducing the setup.

**setup:** `probe_truth.py` trains logistic-regression truth probes on gemma-2-2b activations from the paper's `cities` dataset (statements like "The city of X is in country Y"), across layers 1–25. Variants tried through the week: probes on mean-over-tokens vs last-token activations, three separately-trained probes vs one, regularization at $C = 0.1$, and reporting train accuracy alongside test. `probe_cross_dataset.py` then does the real test: **train on `cities`, evaluate on `neg_cities`** (the same facts, negated).

**results:** in-distribution accuracy is high at middle-to-late layers, but that's the easy part. Cross-dataset transfer to `neg_cities` is where things get interesting/messy — accuracy drops substantially, consistent with the paper's warning that probes latch onto features that don't survive negation. Regularization and the mean-vs-last-token choice both move the numbers, which itself is a sign the probe is not reading one clean direction. This set up the confound stress-test as the obvious next step.

![cross-dataset truth-probe accuracy by layer](/research-log/week3-truth-cross-dataset.png)

## next week

- build deliberately **confounded** train/test splits of cities/neg_cities and grid out what the probe actually keys on
- probe not just $h$ but the decomposition — embedding part vs residual part — to ask *where* the truth signal lives
