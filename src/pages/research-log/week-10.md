---
layout: ../../layouts/ResearchLogLayout.astro
title: the gaussian null
week: 10
date: 2026-08-09
blurb: first positive result of the thread — after skip-embed, a random transformer's residual is pure noise to an SAE, while a trained one beats its Gaussian floor by 37%.
---

## experiment: tier 1 — SAEs on both arms, against a Gaussian null

**motivation:** last week's tier-0 check showed the trained/random distinction is invisible to variance decomposition, so it has to be settled at the SAE level. But raw FVU on a random model is confounded: a trained residual stream is anisotropic (easy to sparse-code) and a random one is closer to isotropic (intrinsically hard), regardless of any learned features. Following the sanity-check logic of [Heap et al. 2025](https://arxiv.org/abs/2501.17727) and the random-baseline results in [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) (where random-direction SAEs nearly match trained ones on interpretability *and* sparse probing), every FVU needs its own **matched-covariance Gaussian null**: what would this SAE score on Gaussian noise with the same covariance as this data?

**setup:** ported `eval_fvu.py` to the on-the-fly activations + HuggingFace pipeline (with a dual FVU convention, centred and uncentred, to keep numbers comparable across scripts), and wrote `gauss_null.py`, which fits/evaluates an SAE on covariance-matched Gaussian samples. Retrained all four cells at **matched 20M tokens** — {`full`, `resid`} × {trained, random seed 0} — in per-arm HF repos. Port validated first: the trained arm reproduces the pre-existing numbers (FVU 0.1496 on $h$, 0.3337 on $r$, $\operatorname{Var}(r)/\operatorname{Var}(h) = 0.441$).

**results:** the raw FVU table actually points the *wrong* way — full degrades 3.14× under randomization vs resid's 2.01×, which would say skip-embed is a worse discriminator. The Gaussian null dissolves the confound (~0.12 of the raw gap was pure isotropy). The statistic that matters is **% below own null**:

| arm | full (plain top-k) | resid (skip-embed) |
|---|---|---|
| trained | **+47.1%** (0.254 vs 0.479) | **+37.3%** (0.334 vs 0.532) |
| random | **+14.5%** (0.534 vs 0.625) | **−3.2%** (0.669 vs 0.649) |

![SAE FVU vs matched-Gaussian null, all four cells](/research-log/week10-gauss-null.png)

Reading it:

- **After removing the token-static part, a random transformer's residual is indistinguishable from correlated Gaussian noise to an SAE** (−3.2% ≈ 0; real activations plausibly have heavier-than-Gaussian tails, so slightly negative is expected). A trained one beats its floor by 37%.
- **The key cell is random/full = 14.5%.** A plain top-k SAE finds genuine non-Gaussian structure in a randomly-initialized transformer — this is quantitatively *why* SAEs "look interpretable" on random nets in Heap et al. The skip-embed SAE on the same model finds zero, and the two differ by exactly one thing (whether $P[\text{tok}]$ was subtracted). So for a random transformer, **all exploitable structure is token-static** — a direct measurement of the mechanism, not an inference.
- Separation (trained − random): full 32.6 points vs resid 40.5 points. But the qualitative framing is stronger than the margin: the plain SAE gives "a matter of degree" (14.5% vs 47.1%), skip-embed gives "a matter of kind" (zero vs 37.3%).
- Also worth quoting anywhere FVU numbers appear: an SAE reconstructs *pure Gaussian noise* to FVU ~0.53 at $k = 64$. Raw FVU is meaningless without that floor — and here the nulls **reversed** the conclusion the raw table suggested. Never quote the raw numbers alone.

**caveats:** $n = 1$ random seed (the caveat that matters — the contrast between the two random cells is 17.7 points, so it should hold, but seeds 1 and 2 come first); reconstruction only, 20M tokens; hybrid/outbias untested on the random arm.

## next week

- random seeds 1 and 2 — nothing gets quoted until the key cell replicates
- `rand_nonembed` (randomize everything *except* the embedding): separates learned representation from learned computation; prediction — plain SAE still finds structure, skip-embed still finds none
- tier 2: `probe_paper.py` on both arms — the k-sparse probe curve *shape*, where "are the features useful" lives
- hybrid / outbias on the random arm
