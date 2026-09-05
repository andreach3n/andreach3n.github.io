---
layout: ../../layouts/ResearchLogLayout.astro
title: two bugs in a converter
week: 12
date: 2026-08-23
blurb: "delphi reversed my result at proper sample size, which would have meant my whole pipeline was wrong. it traced to two bugs in thirty lines of my own checkpoint converter — and the broken SAE had been manufacturing interpretable features out of a random transformer."
---

## experiment: delphi at proper $n$ — and my result reverses

**motivation:** last week's delphi run agreed with me, but on 100 requested latents it scored only 35 (trained) and 23 (random). I already distrust small-$n$ auto-interp; the honest thing is to rerun at a sample size where per-latent noise cannot carry the result, and to cluster by *latent* rather than pooling individual judge decisions, since ~100 correlated decisions per latent is not 100 independent observations.

**setup:** `--max_latents 500 --n_tokens 30000000`, everything else unchanged. That gave 195 (trained) and 161 (random) scored latents versus 35/23 before.

**results:** per-latent, clustered by latent:

| scorer | trained | random | gap |
|---|---|---|---|
| detection | 0.699 ± 0.012 | 0.704 ± 0.017 | **−0.005 ± 0.021** ($z = -0.24$) |
| fuzz | 0.639 ± 0.009 | 0.638 ± 0.013 | **+0.001 ± 0.016** ($z = 0.08$) |

![delphi at n=500 — no gap](/research-log/week12-delphi-n500.png)

**No gap. [Heap et al.](https://arxiv.org/abs/2501.17727) replicates inside delphi.** The trained arm barely moved (0.684 → 0.699); the random arm went 0.609 → **0.704** once measured on 161 latents instead of 23. Last week's "+0.075, survives their pipeline" was small-$n$ noise, and I should not have written it down as a result.

Which puts the disagreement somewhere uncomfortable: same models, same SAEs, same architecture, same layer. My pipeline says 0.754 vs 0.632 at $z > 10$; delphi says nothing. **Possibility 1 — that my metric implementation is wrong — is back, and it is now the live hypothesis.** My mentor had dismissed that branch earlier, but that judgement predates this evidence.

What still separates the arms inside delphi is *response bias*, not accuracy: fuzz TPR/TNR is 0.694/0.583 trained versus **0.797/0.478** random, with TNR below chance. The judge over-accepts on random features and the two errors cancel inside balanced accuracy.

## experiment: AUROC inside delphi — killing my own best excuse

**motivation:** the most comfortable explanation for the previous section is that delphi reports class-balanced *accuracy*, which folds response bias into the score. If the judge is systematically over-accepting on the random arm, accuracy could hide a real difference that a threshold-free statistic would reveal. AUROC — the exact Mann-Whitney statistic on the judge's probabilities — is bias-free by construction, so it tests that excuse directly.

**setup:** this needed logprobs, and OpenRouter cannot supply them for this judge — not an API limitation but a *provider* one: only two providers serve Llama-3.1-70B-Instruct there, DeepInfra returns no logprobs at all and CoreWeave returns only the final token's. So I served the model locally with vLLM and passed `--log_probs`. 198 trained / 161 random latents.

**results:** detection **−0.008 ± 0.023**, fuzz **+0.006 ± 0.024**. Balanced accuracy on the same latents also shows no gap.

![AUROC inside delphi — still no gap](/research-log/week12-delphi-auroc-broken.png)

**So AUROC was not the explanation, and the hypothesis is falsified.** The bias-free metric agrees with the biased one.

But comparing cell by cell against my own pipeline — both AUROC, both on TopK SAEs at 100M — pointed at exactly where the problem lives:

| cell | mine (fuzz / det) | delphi (fuzz / det) |
|---|---|---|
| trained / full | 0.754 / 0.718 | 0.787 / 0.737 |
| random / full | 0.632 / 0.579 | **0.781 / 0.745** |

**The trained arms roughly agree. The random arms diverge enormously.** The two pipelines are not disagreeing about interpretability in general — they disagree specifically about how interpretable a *random* transformer's features are, and delphi scores them nearly as high as trained ones. Whatever the cause is, it lives in the random arm.

## experiment: diffing the caches — two bugs in my own converter

**motivation:** to isolate the random arm, I wrote delphi's latent cache myself from `sae_lens`' own `encode`, bypassing my sparsify converter entirely, then let delphi explain and score it unmodified. The first thing to do with a reimplementation is check it against the original.

**results:** my cache and delphi's archived one agreed on **15% of firings**. Not a rounding difference — a different computation.

It was not the activations (TransformerLens vs HuggingFace cosine 0.999681, both giving the same 0.15) and not the checkpoint (sparsify weights matched my fold to 1e-6). It was `convert_sae_to_sparsify.py`, which had two bugs:

1. **`b_dec` applied twice.** sparsify's `SparseCoder.encode` already does `x = x - self.b_dec`; my converter *also* folded `−b_dec @ W_enc` into the encoder bias. A constant offset across all 73,728 pre-activations changes which 32 win the top-k.
2. **A missing decoder-norm fold.** `sae_lens`' encode returns pre-activations scaled by $\lVert W_{\text{dec},j} \rVert$, and the decoder here is *not* unit-norm (1.3–1.7), so omitting the fold re-ranks latents.

**why it was never caught, which is the transferable part:** the converter verified `h @ encoder.weight.T + encoder.bias` **by hand** — a restatement of its own assumption — instead of calling `sc.encode()`. It passed at 1e-6 while delphi computed something with 0.15 Jaccard against the truth. Verify against the library's *own* call, and on the *selected latents*, not the value vector.

The replacement route (`write_delphi_cache.py`) validated to **Jaccard 1.000000** against delphi's archived cache when told to emulate sparsify's exact computation, which proves the format, indexing and sharding are right and isolates the defect to the conversion.

**every delphi number before this point is invalid** — the +0.075, the n=500 "no gap", and the AUROC "no gap" above. Not affected: my own GPT-judge 2×2, all the FVU and Gaussian-null work, `context_var.py`, the probes, and the SAE training itself.

## experiment: what the broken SAE actually was, and what it did

**motivation:** "the converter was wrong" is a bug report. "The converter changed the answer, and here is the mechanism" is a result. Two measurements make the difference.

**setup:** (a) `fvu_converted_sae.py` — reconstruction quality of the converted SAE as delphi actually ran it, against the true SAE, on 512k tokens; (b) `ablation_delphi.py` — a 2×2 over {broken conversion, true SAE} × {BOS-prefixed, BOS-free}, since delphi's activation regime differs from mine and I needed to know which defect mattered.

**results, (a):** in delphi's own regime the trained arm's converted SAE reconstructs to FVU **3.951** — nearly **4× worse than predicting the mean**. Top-32 set overlap with the true SAE is **0.0000**: not one token in 512k. Its "features" were not features of any trained SAE.

The damage is asymmetric — the conversion wrecked the *trained* dictionary (0.278 → 3.951) and barely touched the *random* one (0.638 → 0.683). Tempting to call that the mechanism, but it is not: the trained arm's AUROC barely moved (0.787 → 0.781) while the random arm's fell 0.781 → 0.632. Reconstruction damage and score inflation are different things.

**results, (b):** random arm, plain SAE, fuzz AUROC:

| | no BOS | BOS |
|---|---|---|
| broken conversion | **0.779** (n=160) | 0.764 (n=167) |
| true SAE | 0.573 (n=328) | **0.632** (n=336) |

The conversion effect is +0.132 to +0.206; the BOS effect is **−0.059, the opposite direction**. The BOS mismatch made random features look *less* interpretable and was partially *masking* how much the conversion inflated them.

![firing distribution — the broken conversion collapses the dictionary](/research-log/week12-firing-distribution.png)

**the mechanism, measured:** the broken conversion *halves* the surviving latents (160 vs 328) and drops firings in latents 0–499 from 0.138 to 0.098 per token. With k=32 fixed, that is the dictionary collapsing into fewer, much more active latents — exactly what a constant bias offset predicts. And TNR *rises* under the broken SAE (detection 0.62 vs 0.39): a near-always-on latent has crisp, describable surface behaviour that a judge can characterise *and* correctly reject non-examples for. **The mis-conversion manufactured genuinely describable features out of a random transformer.** That is how it produced a false replication.

## experiment: the corrected 2×2

**setup:** local Llama-3.1-70B-AWQ with `--log_probs`, caches from `write_delphi_cache.py`, n = 401/336/429/445 latents.

**results:** AUROC per latent, clustered:

| cell | detection | fuzz |
|---|---|---|
| trained / full | 0.732 ± 0.008 | 0.781 ± 0.008 |
| random / full | 0.610 ± 0.009 | 0.632 ± 0.009 |
| trained / resid | 0.697 ± 0.008 | 0.754 ± 0.007 |
| random / resid | 0.566 ± 0.005 | 0.554 ± 0.006 |

![corrected delphi 2×2](/research-log/week12-delphi-2x2-corrected.png)

Gaps: plain **+0.149** fuzz ($z = 12.0$), skip-embed **+0.200** ($z = 20.7$). **Heap et al. does not replicate inside delphi either** — the earlier "it replicates" was the defects.

**The two pipelines now agree cell by cell**, to within ~0.03 everywhere. The old divergence was *entirely* in the random arms, where the mis-converted SAE inflated interpretability by 0.15–0.19. **"The disagreement is the metric" is dead**, and possibility 1 with it.

Skip-embed's advantage gets a third instrument: difference-of-differences **+0.051 ± 0.016** ($z = 3.24$) on fuzz AUROC, close to the +0.042 my own pipeline gave. But detection is +0.011 ($z = 0.69$), **not significant** — the effect is fuzz-specific and I should not claim it for detection.

**caveats:** still $n = 1$ random seed. And `--max_latents 500` is `torch.arange(500)`, not a frequency-matched draw — trained/`resid` latents 0–499 fire at 0.55/token against 0.217 uniform, so that cell's sample is unusually active.

## experiment: opening the pythia arm — and the real headline

**motivation:** every result so far is one model family. Heap et al.'s suite is Pythia, so the standing "gemma is not pythia" caveat can only be killed by rerunning on their model. This time I trained **natively in sparsify** — delphi loads sparsify checkpoints directly, so the entire conversion bug class that ate this week is *structurally absent*.

**setup:** pythia-1b, layer 8 of 16, R=64 (d_sae 131,072), k=32, 100M tokens, openwebtext. Randomization verified first across 20 gates, including a Pythia-specific trap: **`tie_word_embeddings` is `False` on Pythia but `True` on Gemma**, so `embed_in` and `embed_out` are separate tensors and *both* must be re-drawn. A Gemma-recipe copy-paste would silently leave a trained unembedding inside a "random" model.

An aside worth recording: **the paper's dataset no longer exists.** Both `RedPajama-Data-1T-Sample` and `SlimPajama-627B` 404 even with a valid token, and the surviving parent repo is a loading *script* that `datasets>=3` refuses to run — so the example command in sparsify's own README no longer executes. I moved to openwebtext, which is also closer to Pythia's training distribution than RedPajama was.

**results:** per-latent fuzz AUROC at LR 1e-3 — trained **0.626** (n=397) versus random **0.808** (n=444). A gap of **−0.183** ($z = -17$): a *reversal*, opposite in sign to gemma's +0.149. The LR 2e-3 cell gives a gap of ≈0, but its trained arm returned only n=192 against the random arm's 410 — a selection artifact, not a replication.

![pythia-1b layer 8, per-latent AUROC](/research-log/week12-pythia-auroc.png)

**the mechanism, and the actual headline.** Per-latent AUROC against firing count: trained Spearman **+0.617** — its bottom seven deciles sit at 0.50–0.59, i.e. *chance*, and only the top decile reaches 0.884 — while random is **−0.012**, flat at ~0.81 in every single decile. Median firings: 459 trained versus 2,482 random.

![the gap resolved by firing frequency](/research-log/week12-pythia-freq-gap.png)

**So the naive mean was comparing two different frequency mixtures.** The headline of this arm is not "replicates" or "doesn't replicate" — it is that the aggregate hides a sign change, and any comparison that does not condition on firing rate is reporting a composition effect.

**caveats:** one seed; one layer; and the LR 2e-3 cell should not be quoted at all until the selection issue is handled properly.

## next week

- train skip-embed SAEs natively on pythia via the same hookpoint trick, and get the $\Delta\Delta$ — does gemma's +0.051 carry over?
- frequency-matched (directly standardized) gaps as the *primary* statistic, with the marginal number demoted to secondary
- work out why the trained arm's low-frequency latents score at chance — is that real, or is it delphi's example construction?
- the selection problem needs a principled fix: `arange(500)` plus a survival threshold is not a defensible sampling frame when survival differs by arm
