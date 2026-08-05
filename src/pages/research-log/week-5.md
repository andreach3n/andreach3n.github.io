---
layout: ../../layouts/ResearchLogLayout.astro
title: sae training infrastructure
week: 5
date: 2026-07-05
blurb: SAEs on GPT-2's embedding matrix — Louvain stability checks, JumpReLU sparsity sweeps, convergence at 150M–600M samples.
---

This week the hypergraph thread turned into SAE-training infrastructure: to get English "atoms of discourse" I have to train the sparse code myself, and before trusting *any* interpretability readout from an SAE, the SAE itself has to be validated.

## experiment: community stability and controls

**motivation:** last week's Louvain communities looked interpretable, but community detection is notoriously unstable — before building on the clusters I wanted to know how much they change under seeds, resolution, and a control.

**setup:** `stability.py` compares community partitions across runs; `louvain_control.py` runs the same pipeline on a control graph; swept the Louvain resolution parameter; `crime_community.py` follows one semantically coherent community (crime-related tokens) across settings as a concrete tracer.

**results:** the coarse structure (punctuation, numbers, a few big topical blocks) is robust; fine-grained community boundaries move around between runs. The per-dimension hyperedges remain the most stable, crispest objects — which reinforced reading atoms off dimensions rather than leaning on the global partition.

## experiment: TopK / JumpReLU SAEs on GPT-2 embeddings

**motivation:** the English version of the atoms requires a trained sparse code over GPT-2's embedding matrix. SAEs are famously sensitive to training choices, so I built on the two sparsity-control lines that made them reliable — **JumpReLU** ([Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435)) and **TopK** ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)) — and ran the validation sweeps *before* looking at features.

**setup:** `SAEs/gpt2_sae.py` trains a TopK SAE on GPT-2's embedding matrix $W_E$ (50257 × 768, globally rescaled so the mean embedding has norm $\sqrt{d}$), using the `dictionary_learning` trainers. `jumprelu_sweep.py` sweeps the JumpReLU sparsity coefficient (12 values) and target $L_0$; trained seed-replicate SAEs for stability; convergence testing at increasing sample budgets — 150M → 300M → 600M — with the coefficient pinned at 1.0.

**results:** the sparsity–reconstruction frontier behaves as it should (more sparsity, worse reconstruction, smooth tradeoff), and the 12-coefficient sweep at 300M samples gives a usable picture of where on the frontier to sit. Seeds agree reasonably well at the frontier level. The main deliverable is negative-space: nothing exotic happened, which is what you want from infrastructure — I now have SAE training I can trust.

![JumpReLU sparsity sweep, 12 coefficients, 300M samples](/research-log/week5-jumprelu-sweep.png)

## experiment: effective rank of the GPT-2 embedding

**motivation:** a quick sanity check on the object I'm sparse-coding — if the embedding matrix were effectively very low-rank, a sparse dictionary over it would be a strange object.

**setup:** `SAEs/effective_rank.py` computes singular values of $W_E$ (raw and mean-centered), the stable rank $\sum_i \sigma_i^2 / \sigma_1^2$, and the entropy-based effective rank $\exp(-\sum_i p_i \log p_i)$ with $p_i = \sigma_i^2 / \sum_j \sigma_j^2$.

**results:** the embedding is not degenerate — effective rank is a healthy fraction of 768 once centered (centering matters; the mean embedding direction dominates the raw spectrum). Fine to proceed.

## next week

- move from embeddings to **activations**: extract a large activation dataset from gemma-2-2b and train SAEs on it
- the real question the linear map has been waiting for: does an SAE trained on the residual $r = h - P[\text{token}]$ learn fewer trivial single-token features than one trained on $h$?
