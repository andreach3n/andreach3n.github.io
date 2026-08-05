---
layout: ../../layouts/ResearchLogLayout.astro
title: validating the judge
week: 8
date: 2026-07-26
blurb: a written rubric, a blinded human-vs-judge agreement study, one real judge bug found and fixed — then rebuilt the whole pipeline around HuggingFace and retrained at 200M tokens.
---

## experiment: a written rubric and a human validation set

**motivation:** an LLM judge is only evidence if it agrees with humans — [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) ([Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) validates its auto-interp scores against human labels for exactly this reason. Before trusting a *null* on abstractness across three SAEs, the rubric needed to be explicit and the judge needed to be checked against me.

**setup:** wrote the rubric down properly (`rubric_written.md`): a coherence gate (1–5) that *gates* an abstractness ladder (0–5: exact spelling → one word → topic → role/frame → abstract idea). `annotate.py` gave me a blinded, SAE-balanced sample of features to hand-label (`human_val.json`), compared axis-by-axis against the judge (`judge_val.json`).

**results:** agreement is strong on coherence, breadth, and the "no pattern" floor; weakest on abstractness — the axis the conclusion rests on, of course. Digging into disagreements found a real judge bug: it under-rated abstractness on **subword-split** tokens, reading `Cr|umble` as unrelated fragments. Added a "read the whole word/construction" rule to the rubric, then re-validated on a **held-out seed** so the fix isn't graded on the features it was tuned on. With the revised rubric, re-ran the full 2832-feature batch (`judge_ratings.json`). The week-7 conclusions survive: breadth is the axis that moves; the abstractness null stands, now with a validated instrument behind it.

![breadth (effective #words) across SAE variants — the axis that moves](/research-log/week8-judge-breadth.png)

## experiment: how far does the map reach across depth?

**setup:** measured the linear map's $R^2$ at every layer of gemma-2-2b — the embedding-inherited fraction of the residual stream as a function of depth.

**results:** the token-static fraction is large early (around $R^2 \approx 0.85$ at layer 1) and decays with depth, with layer 13 sitting near 0.55. This chart is the quantitative backdrop for the whole project — it says where a token→activation map can matter at all, and it flags early layers as the place the residualization effect should be strongest if it exists anywhere.

![linear-map R² by layer](/research-log/week8-r2-by-layer.png)

## experiment: portable pipeline + 200M retrain

**motivation:** not science — a prerequisite. Every result so far lived on one region-locked Runpod network volume that regularly had no GPUs available. And the SAE budgets had drifted: full/resid were trained at 50M tokens, hybrid/outbias at 20M — the specific head-to-heads were matched, but the set wasn't.

**setup:** rebuilt the pipeline to be disposable-pod-friendly: `activations.py` generates activations **on the fly** (200M tokens of cached activations would be ~920 GB — not worth storing), `fit_map.py` refits the map, `hf_io.py` pushes/pulls checkpoints to a private HuggingFace repo (`andreayhchen/gemma2-2b-linearmap-saes`), and `setup.sh` does a one-shot fresh-pod install. Then retrained **all four variants at a matched 200M tokens each**.

**results:** the new full SAE converges to FVU **0.135**, consistent with the old ~0.143 baseline at 50M — migration validated. Re-ran the pooled dictionary-triviality plot on the 200M set; the picture holds. Remaining debt: the eval scripts still read the old cache-dir layout and need the pull-from-HF treatment. (One expensive lesson from this stretch, recorded for posterity: `git pull` on the box silently not carrying local edits cost hours — verify the file on the box before trusting a run.)

![pooled dictionary triviality on the 200M-token SAEs](/research-log/week8-dict-200M.png)

## next week

- move from "are features trivial?" to "do features *encode useful concepts*?" — reproduce the k-sparse probing protocol from the Temporal SAEs paper (Figure 3) on my SAE variants
- MMLU subjects and FineFineWeb domains as semantic tasks vs syntactic ones
