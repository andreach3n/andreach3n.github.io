---
layout: ../../layouts/ResearchLogLayout.astro
title: hybrid maps and an llm judge
week: 7
date: 2026-07-19
blurb: jointly training the map with the SAE turns the null partly positive — better reconstruction and a less trivial dictionary — but an LLM judge says the features get broader, not more abstract.
---

The SAE zoo this week, for reference: **full** = SAE on $h$; **resid** = SAE on $r = h - P[\text{token}]$ with a frozen map; **hybrid** = map trained *jointly* with the SAE, end-to-end on $h$; **outbias** = encoder sees full $h$, map only added at the output.

## experiment: hybrid (joint) training

**motivation:** Nathan's fix for last week's null — a frozen, pre-fit map is a blunt instrument. Letting the map train jointly with the SAE lets the two coordinate, which is essentially the learned-per-token-bias move of Tokenized SAEs ([Dooms & Wilhelm 2025](https://arxiv.org/abs/2502.17332)).

**setup:** `MODE="hybrid"` in `train_sae_res.py` — map warm-started from the greedy fit with `requires_grad=True`, encoder still sees $h - P$, loss is the SAE's residual MSE (algebraically identical to minimizing $h$-reconstruction, so no degenerate shortcut).

**results:** two wins at matched dictionary/k. FVU on $h$: **0.1426 (full) → 0.1376 (hybrid)**, while the frozen map was a wash (0.1445) — joint training is the active ingredient. And the dictionary de-trivializes: peak single-word fraction **15.7% → 9.2%**, and this one *holds within frequency bins*, so it's a real per-feature effect, not the composition artifact that killed the resid comparison. Eyeballing with `dashboard.py`'s `compare_concept`: for health/insurance/election/district, full always uses a single-token detector where hybrid uses a broader coherent concept (election → electoral politics, district → federal district *court*). Honest caveats: it's word → **topic**, not word → abstraction; the trivial detection *moved into the dense map* rather than disappearing; and purely-lexical tokens get absorbed by the map rather than broadened.

## experiment: reconstruction-vs-sparsity sweep

**setup:** trained full + hybrid at $k \in \{16, 32, 64, 128, 256\}$ (20M tokens each, seeded); `eval_sweep.py` + `plot_sweep.py`.

**results:** hybrid sits below full at **every** k — not a k=64 fluke. Rel. FVU reduction ~4% (7% at k=16); read horizontally, hybrid needs **~10–15% fewer active features** for equal reconstruction, up to ~27% at k=16 — the map helps most when the sparse budget is tightest.

![reconstruction vs sparsity — hybrid below full at every k](/research-log/week7-recon-vs-sparsity.png)

## experiment: what did joint training do to the map?

**setup:** `map_compare.py` compares the greedy vs jointly-trained map: standalone $R^2$, per-token cosine, norm ratio.

**results:** the jointly-trained map is a *worse* standalone predictor ($R^2$ 0.551 → 0.496) but a better teammate: it shrinks its predictions to ~85% norm and rotates ~24° (mean cosine 0.914) — it *under-commits*, leaving the SAE an easier residual to sparse-code.

![greedy vs jointly-trained map: R², cosine, norm ratio](/research-log/week7-map-comparison.png)

## experiment: outbias ablation — input or target?

**motivation:** hybrid changes two things at once (what the encoder sees, and what gets reconstructed). The outbias mode isolates them: encoder sees full $h$, the map only shapes the reconstruction target.

**setup:** `MODE="outbias"` with a custom native-form MSE plus the reused AuxK loss — without AuxK it collapses to ~13k dead features (and the dead-mask only populates at step ~200, so short runs *falsely* look collapsed).

**results:** peak single-word: full 15.7% / hybrid 9.2% / **outbias 7.7%** — the variant whose encoder *sees* the token is the least trivial. So the de-trivialization comes from the reconstruction **target** ($h - P$), not the encoder input. Reconstruction: outbias 0.1440 ≈ hybrid 0.1432, both beat full (0.1491 on the clean matched eval) — a cleaner dictionary at roughly the same reconstruction.

## experiment: LLM judge — broader, or actually more abstract?

**motivation:** word-count triviality only measures lexical spread — it can't separate a genuinely abstract feature from noise that fires on many words. Auto-interp with an LLM judge is the standard instrument: OpenAI's neuron-explanation recipe ([Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) and Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) ([Templeton et al. 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) — pointed here at *abstractness* instead of specificity.

**setup:** `build_judge_features.py` samples 944 features per SAE (full/hybrid/outbias), **frequency-matched by construction** via stratified equal-N per half-OOM frequency bin, and builds blinded peak + typical example blocks. `judge_features.py` runs an OpenAI batch judge (strict JSON schema; axes: breadth, coherence, abstractness × {peak, typical}; ~$9 for 2832 features). `analyze_judge.py` joins and plots with z-tests.

**results:** breadth **up, significant** (hybrid +0.31, outbias +0.20 at peak) — matches the word-count story. Abstractness **flat** (null; outbias even significantly *less* abstract in typical firing). Coherence **down** (−0.25 to −0.50). And level-5 abstraction is ~empty for *all three* SAEs — layer-13 features top out at topic/frame, never sycophancy-like abstraction. So the map buys **broader lexical spread, not richer meaning** — the added breadth is even slightly incoherent. An honest complication to the hybrid/outbias positive framing.

![judge axes: breadth / coherence / abstractness by SAE](/research-log/week7-judge-axes.png)

![abstractness distribution — flat across variants](/research-log/week7-judge-abstractness.png)

## next week

- the abstractness null is only trustworthy if the judge is — write an explicit rubric, hand-label a blinded validation set, and check human-judge agreement axis by axis
- robustness: abstractness within frequency bins, judge test-retest
