# Implementation Plan

Based on the paper: "Информационно-разведочный поиск на основе метода количественной оценки концептуальной новизны научных публикаций"

## Paper Summary

The paper proposes a method to find breakthrough scientific publications by quantifying their **conceptual novelty**. Key insight: papers are represented as configurations of a finite set of **base ideas**, and novelty is measured by how rare + coherent that configuration is.

## Pipeline Architecture

```
Papers corpus W = {w_1, ..., w_N}
        |
[1. SEGMENTATION] — split each paper into semantic segments
        |  (sentence embeddings → local max gap → segments c_1...c_M)
        |
[2. IDEA EXTRACTION] — LLM extracts one idea per segment
        |  (LLM(segment, full_paper_context, prompt) → idea formulation i_j)
        |
[3. IDEA CLUSTERING] — build base idea set I = {I_1, ..., I_K}
        |  (incremental: new idea → if dist < δ merge, else create new I_K+1)
        |  (cluster representative updated via LLM summarization)
        |
[4. IDEA SPECTRUM] — map each paper to K-dim vector
        |  s_i(w) = max_{j in segments(w)} sim(embedding(I_i), embedding(i_j))
        |
[5. NOVELTY METRICS] — compute per-paper scores
        |  D_M(w) — Mahalanobis distance from field center
        |  R(w)   — rarity: -Σ s_i(w) log(p_i)
        |  C(w)   — coherence: Σ s_i(w)²
        |  ρ(w)   — local density: 1/mean_kNN_dist
        |  B_pot(w) = R(w) · C(w) · 1/ρ(w)   [Breakthrough Potential Index]
        |
[6. RANKING + VALIDATION]
        |  Retrospective: do known breakthroughs rank high?
        |  Ablation: which components matter?
```

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Segmentation | `src/novelty_search/pipeline/segmentation.py` | Implemented |
| Idea Extraction | `src/novelty_search/pipeline/idea_extraction.py` | Implemented |
| Idea Clustering | `src/novelty_search/pipeline/idea_clustering.py` | Implemented |
| Novelty Metrics | `src/novelty_search/metrics/novelty.py` | Implemented |
| Embeddings | `src/novelty_search/utils/embeddings.py` | Implemented |
| Data Collection | `scripts/collect_*.py` | **Done** (513K papers) |
| End-to-end Pipeline | — | **Not yet wired** |
| Experiments | — | **Not started** |
| Evaluation | — | **Not started** |

## Data Readiness

| Dataset | Papers | Years | Full Text | Citations |
|---------|--------|-------|-----------|-----------|
| v3 arXiv (cs.CL/LG/CV/AI) | 442,507 | 2016-2026 | 99.9% | Yes |
| ACL All-NLP (merged) | 71,117 | 1952-2026 | 96.8% | 93.1% verified |
| **Total** | **513,624** | **1952-2026** | **99.5%** | **Yes** |

## Remaining Work

### Phase 1: Wire End-to-End Pipeline
1. **Main runner script** — takes a corpus slice, runs full pipeline
2. **Temporal ordering** — process papers chronologically (critical for retrospective eval)
3. **Idea space persistence** — save/load base idea set between runs
4. **Batch LLM processing** — idea extraction at scale (cost estimation needed)

### Phase 2: Retrospective Experiment
1. Pick a focused domain (e.g., cs.CL 2016-2022)
2. Define breakthrough set B (papers with 500+ citations by 2026)
3. For each paper w_t, compute B_pot using only papers before t
4. Evaluate: Precision@k, nDCG@k, average rank of B papers

### Phase 3: Ablation Study
Remove one component at a time, measure quality drop:
- M1: No segmentation (treat full text as one segment)
- M2: No LLM extraction (use segment text directly as "idea")
- M3: No clustering (each idea is its own base idea)
- M4: Only D_M (no R, C, ρ)
- M5: Only R (no D_M, C, ρ)
- M6: No coherence C
- M7: No density ρ

### Key Decisions Needed
1. **LLM choice** — GPT-4o-mini (cheap, fast) vs Claude Haiku vs local model
2. **Embedding model** — all-MiniLM-L6-v2 (current) vs larger model
3. **Corpus scope for first experiment** — full 513K or focused subset?
4. **Cost estimation** — LLM calls for 500K papers × ~5 segments each = ~2.5M calls
5. **δ threshold tuning** — currently 0.3, may need grid search

### Cost Estimates (rough)
- **GPT-4o-mini**: ~$0.15/1M input tokens, ~$0.60/1M output
- Average paper: ~5 segments × ~500 tokens input + ~50 tokens output = 2,750 tokens
- 500K papers: ~1.375B input tokens + ~125M output tokens
- **Total: ~$280** (input $206 + output $75)
- Could start with 10K-paper pilot for ~$5
