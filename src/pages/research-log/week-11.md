---
layout: ../../layouts/ResearchLogLayout.astro
title: scaling to the paper's recipe
week: 11
date: 2026-08-16
blurb: "retrained everything at the paper's scale to rule out undertraining. it worked — but every cell rose and the random one rose most, so the effect I'm arguing against gets stronger with better SAEs."
---

Last week ended with a plan: random seeds 1 and 2, then `rand_nonembed`. Neither happened. My mentor's triage reframed the question — my non-replication of [Heap et al. 2025](https://arxiv.org/abs/2501.17727) could be (1) a broken metric implementation, (2) undertrained SAEs, or (3) a genuine non-replication — and possibility 2 is the one an outside reader raises first, because my 20M-token / 16k-latent SAEs are much smaller than the paper's 100M / R=64. Seeds test whether my number is stable; scale tests whether it exists at all. Scale went first.

## experiment: is the learning rate hiding the effect?

**motivation:** possibility 2 has two knobs, tokens and learning rate, and LR is far cheaper to sweep. There is also a specific mechanism worth killing: if the random arm's dictionary were mostly *dead*, auto-interp would be sampling from a handful of surviving latents and would score low for a reason that has nothing to do with interpretability. [Gao et al. 2024](https://arxiv.org/abs/2406.04093) introduce AuxK precisely because top-k SAEs lose latents during training, so "how many latents are alive" is a real failure mode, not a hypothetical.

**setup:** `eval_lr_sweep.py` on the random arm, `full` mode, k=32, d_sae=73,728, 20M tokens each, evaluated on 2M held-out tokens (stream seed 7). Three rates: 1e-4, 4e-4 (the project default), 1e-3. The 4e-4 row comes free as the 20M milestone of the 100M run, which is fair because training uses a *constant* LR with no scheduler.

**results:**

| LR | FVU (centred) | L0 | alive | alive % |
|---|---|---|---|---|
| 1e-4 | 0.5988 | 32.0 | 48,036 | 65.2% |
| **4e-4** | **0.5699** | 32.0 | **73,725** | **100.0%** |
| 1e-3 | 0.6313 | 32.0 | 73,250 | 99.4% |

The project default wins on every metric, so there is nothing to promote to 100M and the LR branch is closed. But the load-bearing number is **alive % = 100 at 4e-4**: the dictionary is already fully alive at 20M, which kills the dead-dictionary mechanism outright. What survives of possibility 2 is only "alive but *under-refined*" — a latent can fire constantly and still be an uninterpretable superposition — and that can only be settled by auto-interp on the milestone checkpoints.

This also resolved an earlier scare: ~93% dead latents at step 300 is normal cold-start at this dictionary size, AuxK clears it completely, and the loss *jump* at step ~201 is AuxK switching on rather than divergence.

## experiment: the 100M fleet, and a lesson about which FVU you quote

**setup:** six runs on one 6×L40S pod — {trained, random} × {`full`, `resid`} at 100M tokens, plus the LR arm. Frozen milestone checkpoints every 20M so auto-interp could be plotted *against training tokens*, which is the actual decision rule: if the random arm is still climbing at 100M, undertraining was the story; if it is flat from 20M, weight shifts to possibility 3. One process per GPU via `CUDA_VISIBLE_DEVICES`, not DDP — I want six different models, not one model sharded six ways.

**results:** all four ran to step 24,400 and pushed cleanly. Held-out on 2,000,308 fresh tokens, k=32, d_sae=73,728:

| | full (centred) | resid-on-$r$ | composite-on-$h$ |
|---|---|---|---|
| trained | 0.2663 | 0.3593 | 0.2686 |
| random | 0.5372 | 0.6912 | 0.5350 |

A determinism check fell out for free: $\operatorname{Var}(r)/\operatorname{Var}(h)$ on the trained arm is **0.4411 flat — identical to the 20M eval's 0.4411**, same model, same map, same stream seed.

**one correction, and it is the useful part of this section.** Reading the final *training batch*, the random arm's `full` FVU looked like 0.379 — a big improvement over the old 0.4701, which I briefly took as "a bigger SAE finds much more structure in a random transformer." Held-out says **0.4725**. No improvement at all. Never compare a training-batch FVU to a held-out one.

**an unplanned discriminator.** The train-vs-held-out gap is wildly asymmetric: trained/full 0.153 → 0.157 (+0.004), random/full 0.379 → 0.4725 (**+0.094**). A 73,728-latent SAE appears to *memorize* a random transformer's activations while genuinely generalizing on a trained one. If that survives a proper full-train-set pass rather than a single batch, "generalization gap" is a cheap trained-vs-random discriminator that needs no auto-interp at all.

## experiment: refitting the nulls, and the framing that did not survive

**motivation:** last week's headline was "% below own matched-Gaussian null," and those nulls were fit at k=64 with a 16k dictionary. They do **not** transfer to k=32 / 73,728 — the raw FVU difference (0.5013 here vs 0.4701 recorded at the old config) is a *sparsity* difference, not a finding. "% below null" is a property of the *(model, SAE)* pair, not of the model, so every cross-arm percentage had to be recomputed before it could be quoted.

**results:** full table at k=32 / R=32 / 100M, % below own null, centred:

| arm | full | resid |
|---|---|---|
| trained | **53.3%** (0.2663 vs 0.5703) | **42.0%** (0.3593 vs 0.6198) |
| random | **25.4%** (0.5372 vs 0.7204) | **6.6%** (0.6912 vs 0.7399) |

A useful validity check: flat and centred conventions now agree *exactly*, as they must when the null matches both $\mu$ and $\Sigma$.

**Both random cells rose** — 14.5% → 25.4% on `full`, and −3.2% → **+6.6%** on `resid`. So a better-trained SAE extracts *more* genuine non-Gaussian structure from a random transformer, and **the Heap et al. effect strengthens with better SAEs.** That is the opposite of what I wanted, and it kills last week's cleanest sentence: the categorical "a matter of *kind*" framing (zero vs 37.3%) does not survive better training. The 20M "zero" was partly an artifact of an undertrained, small SAE.

**what does survive, and is the better claim: the asymmetry of what skip-embed removes.** Random goes 25.4% → 6.6%, i.e. skip-embed removes ~74% of the exploitable structure. Trained goes 53.3% → 42.0%, removing only ~21%. Same subtraction of $P[\text{tok}]$, opposite effect depending on whether the model learned anything — and it needs no cell to be exactly zero. Better still, the *trained* arm's removal fraction is essentially config-independent (20.8% at 20M/k64/16k vs 21.2% here), which is evidence it measures a real property of trained gemma's layer 13 rather than an SAE artifact. The random arm's equivalent is not stable (~100% → 74%), which is itself informative.

## experiment: auto-interp at the paper's scale

**motivation:** this is the measurement that actually decides possibility 2 versus 3. Auto-interp in the [Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) lineage — explain a latent from its top activations, then score whether a held-out judge can use that explanation to pick activating examples — is the metric Heap et al. report as failing to separate the arms.

**setup:** full 2×2, 520 latents per cell sampled frequency-stratified, ~1,955 explanations and ~3,887 scoring calls through the OpenAI batch API, about $20.

**results:** fuzz AUROC (20M/k64/16k numbers in parentheses):

| cell | fuzz | detection | (old fuzz) |
|---|---|---|---|
| trained / full | **0.754** | 0.718 | (0.716) |
| trained / resid | **0.724** | 0.695 | (0.679) |
| random / full | **0.632** | 0.579 | (0.556) |
| random / resid | **0.560** | 0.535 | (0.520) |

![auto-interp AUROC across the 2×2 at 100M tokens](/research-log/week11-autointerp-100M.png)

1. **Heap et al. still does not replicate.** Trained and random are clearly separated (0.754 vs 0.632) after 5× the tokens, 4.5× the dictionary, and a verified-optimal LR. **Undertraining is not the explanation.**
2. **But possibility 2 had real substance.** Every cell rose and random/full rose most (+0.076). In above-chance terms it went from 26% of trained/full to **52%** — it doubled — and the raw gap narrowed 0.160 → 0.122. I should not claim the separation is scale-stable.
3. **Skip-embed's separation is scale-stable where the plain SAE's is not.** The `resid` gap went 0.159 → **0.164** while `full` eroded 0.160 → 0.122, losing a quarter of its discriminating power. That is the strongest argument yet for skip-embed as a *diagnostic*.
4. A free pre-scoring diagnostic, worth quoting on its own: the rate at which the explainer refuses and says "no clear pattern" — trained/full 17.1%, trained/resid 15.4%, random/full **46.0%**, random/resid **54.9%**. Skip-embed moves the two arms in *opposite* directions, widening separation from 28.9 to 39.5 points.

Three independent instruments now agree that skip-embed holds or widens the separation where the plain SAE erodes: FVU-vs-Gaussian-null (27.9 → 35.4 points), explainer null-rate (28.9 → 39.5), and auto-interp AUROC (0.122 vs 0.164).

## experiment: running the paper's own pipeline

**motivation:** if my pipeline and the paper's disagree, one of us is wrong and it matters which. The honest move is to run *their* code — EleutherAI's `delphi`, with its default Llama-3.1-70B-Instruct judge — on *my* SAEs, changing as little as possible.

**setup:** SAEs retrained with `SAE_ARCH=topk` (per-token TopK, the paper's architecture and the only rule sparsify can express), layer 13, 100M tokens, openwebtext, `--max_latents 100`.

**results:** class-balanced accuracy, trained vs random: detection 0.683 vs 0.608, fuzz 0.669 vs 0.587. Gaps of roughly 6σ treating decisions as independent, 3–4σ once clustered within latents.

![delphi on the topk SAEs, first run](/research-log/week11-delphi-first.png)

So the non-replication is not an artifact of my reimplementation or my judge model. And the confusion matrices show a mechanism: on the **random** arm the Llama judge scores TPR 0.772 but TNR **0.445 — below chance**. It is not discriminating, it is *agreeing*; its apparent accuracy on random-model features is response bias. On the trained arm it is calibrated (0.729 / 0.636). That is direct evidence for the idea that a *weaker explainer inflates random-model scores*: my GPT judge declined to describe 46% of random features, while Llama describes them and then validates its own descriptions.

**caveats:** delphi reports balanced *accuracy*, not AUROC (`Logits not available` — I should try `--log_probs`), so these are not directly comparable to my 0.754/0.718. It also drew 100 latents by its top-firing default from pools of 7,000 (trained) and 4,600 (random), which is not my frequency-stratified design. And $n = 100$ latents is small — an earlier version of this same comparison at $n = 35/23$ is the reason I now distrust small-$n$ delphi numbers.

## next week

- rerun delphi at proper $n$ — `--max_latents 500`, more scoring tokens, and per-latent AUROC clustered by latent rather than pooled decisions
- get `--log_probs` working so delphi reports AUROC, making the two pipelines directly comparable
- recount the "vacuous explanation" rate properly on the n=500 explanations; the current 56.5% vs 31.4% is $z = 1.9$ and was classified by crude substring matching
- still owed from two weeks ago: random seeds 1 and 2, and `rand_nonembed`
