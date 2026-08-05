---
layout: ../../layouts/ResearchLogLayout.astro
title: confounded probes and embedding hypergraphs
week: 4
date: 2026-06-28
blurb: a 7×7 confounded truth-probe grid, then a pivot — reading "atoms of discourse" straight off a weight-sparse transformer's embeddings.
---

## experiment: the confounded probe grid

**motivation:** the core lesson of [Marks & Tegmark 2023](https://arxiv.org/abs/2310.06824) is that a truth probe can score high while keying on a spurious correlate. The way to expose that is to *build the confound in on purpose*: construct train/test splits of `cities`/`neg_cities` where truth and the confound are aligned in training and anti-aligned at test.

**setup:** `probe_grid.py` runs a **7×7 grid** of probes over the confounded dataset variants across layers/representations (embedding-predicted part, residual, and full activation), so each cell answers "trained under this confounding, tested under that one." Built the confounded cities/neg_cities variants earlier in the week.

**results:** the grid makes the shortcut behavior visible: cells where train and test confounding agree look great, and transfer cells fall off — the probe is at least partly reading the correlate, not truth. Comparing embedding / residual / full representations connects this back to the project's spine: the interesting question is whether the *contextual* (residual) part carries the transferable signal. This thread got parked here — partly because the next idea took over, partly because the probes-on-decompositions question comes back much later with better tools.

![confounded probe grid on full activations](/research-log/week4-probe-grid-confounded.png)

## experiment: weight-sparse embedding hypergraph

**motivation:** a hard pivot, from a conversation with my mentor. [Arora et al. 2018](https://arxiv.org/abs/1601.03764) ("atoms of discourse") argue word embeddings are sparse superpositions of a small number of semantic atoms — normally you'd need to *train* a sparse code (an SAE) to find them. But OpenAI's weight-sparse transformers ([Gao et al. 2025](https://arxiv.org/abs/2511.13653)) have embedding matrices that are *already* sparse — each token has ~16 nonzero dimensions — so the atoms can be read straight off the weights, no SAE needed.

**setup:** cloned `circuit_sparsity` and built `SAEs/weight_sparse.py`: treat each embedding dimension as a **hyperedge** over the tokens that use it, i.e. a `{dim: tokens}` hypergraph; project to a token–token graph $A = M M^\top$ (with $M$ the binary sparsity mask); filter hub dimensions (>50 tokens) and byte-fragment/undecodable tokens; run **Louvain community detection**. Getting the model loaded was its own fight — `load_model()` and the HF path are broken by config version-skew, so I read the embedding straight out of the checkpoint's state dict. One catch discovered along the way: the released models are *Python-code* models with a 2k-token `tinypython` BPE, not English.

**results:** at Louvain resolution 2.0 the communities are genuinely interpretable — delimiters/punctuation, numbers, path/maze/search/route words, graph-adjacency vocabulary — though code-flavored because of the tokenizer. The crispest signal is per-dimension: single hyperedges like dim 888 = {node, vertex, Dijkstra} read as atoms directly, while community detection gives a coarser global map. Filtering mattered a lot: restricting to full alphabetic words of length ≥ 4 and dropping hub dimensions is what turned noise into legible clusters (printed to `communities.txt`, with a 30–300-size filtered version).

![Louvain communities over the embedding hypergraph](/research-log/week4-louvain-communities.png)

## next week

- how stable are these communities? rerun across seeds/resolutions and compare
- the English version: the weight-sparse models are code-only, so train a TopK SAE on GPT-2's ~50k English token embeddings and build the same hypergraph from the learned code
