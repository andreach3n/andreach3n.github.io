---
layout: ../../layouts/ResearchLogLayout.astro
title: k-sparse probing and random transformers
week: 9
date: 2026-08-02
blurb: reproduced the Temporal-SAE probing protocol on the skip-embed SAEs, then opened a new thread — can skip-embed SAEs tell a trained transformer from a random one?
---

New naming that stuck this week: the family is now the **skip-embed SAEs** — `resid` (pretrained skip), `hybrid` (residual encoder), `outbias` (raw encoder) — vs the plain top-k `full`.

## experiment: k-sparse probing (Temporal-SAE protocol)

**motivation:** the judge said the map makes features *broader* but not more *abstract* — leaving open whether the decomposition makes them more **useful**. Sparse probing ([Gurnee et al. 2023](https://arxiv.org/abs/2305.01610)) measures how much task-relevant information a small set of features carries, and the [Temporal SAEs paper](https://arxiv.org/abs/2511.05541) turns it into a clean benchmark — its **Figure 3** compares k-sparse probe accuracy on semantic vs syntactic tasks, which is exactly the axis I care about. Reproducing that protocol on my SAEs asks: does moving token-static structure into the map free up features that better encode concepts?

**setup:** three iterations. `probe_sparse.py` (first pass) → `probe_v2.py` (GPU one-vs-rest logistic probes, standardized features, a dense probe as ceiling) → `probe_paper.py` (a faithful replication of the paper's feature-selection + single-multiclass-fit method, in their metric). Datasets: **MMLU** (57 subjects) and **FineFineWeb** (the paper's 10 web domains), tasks split semantic vs syntactic.

**results:** the interesting signal so far: **skip-embed encoders lift semantic probe accuracy** over the plain SAE. One bug mattered en route, traced with Nathan: the dense "ceiling" probe was underfitting because features went in unscaled — fixed with a StandardScaler plus subsampling, which made dense a real ceiling again. A 40k-example subsample check confirmed the numbers are stable at reduced n.

![k-sparse probe accuracy by SAE variant](/research-log/week9-ksparse-bars.png)

![semantic vs syntactic probing on MMLU, 57 subjects](/research-log/week9-probe-mmlu.png)

![paper-metric replication on FineFineWeb, 10 domains](/research-log/week9-probe-finefineweb.png)

![40k-subsample stability check on the MMLU probes](/research-log/week9-probe-mmlu-sub40000.png)

## experiment: trained vs random — tier 0

**motivation:** a mentor-endorsed reframe from the Sunday call. [Heap et al. 2025](https://arxiv.org/abs/2501.17727) show that automated interpretability metrics **do not distinguish** SAEs trained on a real Pythia from ones trained on a randomly-initialized network — trained and randomized arms overlap on auto-interp and AUROC, and only their noise-embedding control drops to chance. My hypothesis: that's because SAE dictionaries are dominated by *token-identity* structure, which survives randomization completely (the embedding table is intact; only the computation is destroyed). Skip-embed factors token identity out — so the trained/random distinction should reappear. That reframes the project's nulls: from "skip-embed doesn't make features more abstract" to "skip-embed makes SAE evaluation *sensitive to whether the model learned anything*."

**setup:** before training any SAEs, the cheap tier-0 check: `context_var.py` builds a full per-token lookup table (which strictly upper-bounds any linear map) and measures how much of $h_{13}$ it explains, on a trained gemma-2-2b vs a re-randomized one (every weight resampled from a Gaussian matched to that matrix's own mean/std — Heap et al.'s recipe), 20M tokens per arm. The randomization itself was verified three ways (288 re-initialized tensors — the exact count for gemma-2-2b including its double norms and tied embedding; healthy activation scale with no massive-activation dims; and next-token loss 22.6 — *above* uniform-chance 12.45, which is correct: a random network is confidently wrong, not uniform).

**results:** **NULL, and it falsifies the cheap shortcut.** Lookup-table $R^2$: trained **0.274** vs random **0.288** — the random model is marginally *more* contextual, and the contextual variance fraction is $\rho \approx 0.71$ for both. A random gemma's layer 13 is as contextual as a trained one by raw variance decomposition: random attention mixes context diffusely, generating plenty of context-dependent variance — just unstructured. Variance measures how much contextual signal exists; the hypothesis is about how it's *organized*. So there's no 1-GPU-hour proxy — the question has to be answered with actual SAEs (FVU against a matched-covariance Gaussian null, probe-curve shapes). Arguably a cleaner statement than Heap et al.'s: trained-vs-random is invisible not just to auto-interp scores but to the variance decomposition itself.

![per-token lookup R² on layer 13 — trained vs random](/research-log/week9-context-var.png)

(Also this week: started this research log, and hit the one real implementation hazard — the HF push paths are hardcoded, so a random-arm run would silently overwrite the trained checkpoints; fixed by namespacing repos via the `HF_REPO` env var.)

## next week

- tier 1: train `full` + `resid` SAEs on both arms at matched 20M tokens, and build the matched-covariance **Gaussian null** — raw FVU on a random model is meaningless without knowing what an SAE scores on pure correlated noise
- port `eval_fvu.py` to the on-the-fly + HF pipeline first
- then tier 2: the k-sparse probe curve *shape* on both arms
