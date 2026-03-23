# Experiment 001: Embeddings Subset — End-to-End Pipeline

## Summary

First end-to-end run of the novelty search pipeline on 5,234 embedding-related ACL papers.

**Key Finding:** Paper-level k-NN novelty (mean idea embedding distance to prior papers) achieves **Spearman r=0.333** with log-citations. The original BPI formula from the paper (R·C/ρ) shows *negative* correlation (r=-0.179) — it rewards obscure papers, not breakthroughs.

## Pipeline

```
5,234 ACL papers (embedding topic)
    → Simple paragraph chunking (13,522 chunks)
    → gpt-5.4-nano idea extraction (62,759 ideas, ~$6.38)
    → text-embedding-3-small (62,750 unique idea embeddings)
    → K-Means clustering (K=100, silhouette=0.014)
    → Temporal metrics (year-by-year, comparing to prior years)
    → Validation (golden set + citation buckets)
```

## Metric Comparison

| Metric | Spearman r | Golden HIGH % | Description |
|--------|-----------|---------------|-------------|
| **p_nn_novelty** | **+0.333** | **81.6%** | Mean kNN distance in paper embedding space |
| p_cos_dist | +0.263 | 72.6% | Cosine distance from prior centroid |
| p_nn_novelty_3y | +0.321 | 82.0% | kNN novelty, 3-year window |
| BPI_original_znorm | +0.086 | 67.8% | Year-normalized original BPI |
| BPI_original | **-0.179** | 60.5% | Original paper formula R·C/ρ |

## Citation Bucket Analysis (p_nn_novelty)

| Bucket | N | Mean Novelty | Median Novelty |
|--------|---|-------------|---------------|
| 0 (zero) | 719 | 0.238 | 0.230 |
| 1-4 (noise) | 1,248 | 0.268 | 0.266 |
| 5-19 (minor) | 1,730 | 0.278 | 0.274 |
| 20-99 (decent) | 1,106 | 0.289 | 0.284 |
| 100-499 (strong) | 350 | 0.299 | 0.294 |
| 500+ (breakthrough) | 58 | 0.298 | 0.300 |

Clear monotonic increase: more cited papers are more novel.

## Golden Set (p_nn_novelty rankings)

| Paper | Citations | Percentile |
|-------|-----------|-----------|
| CNN for Sentence Classification (Kim 2014) | 14,058 | **97.0%** |
| Learning Word Vectors for Sentiment (Maas 2011) | 5,869 | **91.6%** |
| Word Representations: Simple & General (Turian 2010) | 2,340 | **90.1%** |
| Learning Phrase Representations / GRU (Cho 2014) | 25,776 | **87.0%** |
| Linguistic Regularities (Mikolov 2013) | 3,887 | 73.0% |
| GloVe (Pennington 2014) | 34,143 | 51.1% |
| Universal Sentence Encoder (2018) | 987 | 29.7% |
| Flair / Contextual String Embeddings (2018) | 1,362 | 11.8% |

## Why Original BPI Fails

The formula BPI = R · C · (1/ρ) rewards:
- R (rarity): unusual idea combinations → **niche papers**
- C (coherence): focused on few ideas → **narrow scope**
- 1/ρ (isolation): far from other papers → **obscure topics**

This describes niche, isolated papers — NOT breakthroughs.

## What Works Better

Paper-level kNN novelty directly measures: "how different is this paper from the nearest prior papers in embedding space?" This captures:
- Novel combinations of concepts (high distance)
- New methods entering the field (far from prior work)
- Cross-pollination from other fields (different embedding region)

## Cost

See [COST_LOG.md](COST_LOG.md) for detailed cost tracking.

| Step | Estimated Cost | Time | Notes |
|------|---------------|------|-------|
| Extraction Run 1 (gpt-4.1-nano) | **~$8-9** | 5.5 min | WASTED — wrong model fallback |
| Extraction Run 2 (gpt-5.4-nano) | ~$8-9 | 5.5 min | Actual experiment |
| Idea embedding | ~$0.02 | 1 min | text-embedding-3-small |
| Clustering + metrics | $0 | 3 min | Local compute only |
| **TOTAL SPENT** | **~$17-18** | | |
| **Useful cost** | **~$8-9** | ~10 min | Excluding wasted run |
| **Budget remaining** | **~$7** | | of $25 |

**Lesson**: Always test model params with a single API call before running full extraction.

## Next Steps

1. **Try K=200-300 clusters** — finer-grained spectra might improve spectrum-based metrics
2. **Abstract-level embeddings** — embed abstracts directly, skip idea extraction entirely
3. **Combine signals** — p_nn_novelty + year-normalized BPI might complement each other
4. **Scale to full arXiv dataset** (442K papers) with the best metric
5. **Investigate GloVe/Flair ranking** — why do some breakthroughs rank low?

## Files

```
experiments/exp001_embeddings/
├── data/
│   ├── subset.parquet            # 5,234 papers
│   ├── ideas.jsonl               # 62,759 extracted ideas
│   ├── idea_embeddings.npy       # (62,750, 1536) embeddings
│   ├── idea_metadata.json        # idea → paper mapping
│   ├── base_idea_embeddings.npy  # (100, 1536) cluster centroids
│   ├── cluster_labels.json       # cluster assignments
│   ├── cluster_descriptions.json # representative ideas per cluster
│   ├── spectra.npy               # (5,232, 100) idea spectra
│   ├── metrics.parquet           # original metrics
│   ├── metrics_v2.parquet        # 8 formula variants
│   └── metrics_v3.parquet        # + paper-level novelty
├── reports/
│   ├── 01_subset_stats.md
│   ├── 02_extraction_report.md
│   ├── 03_clustering_report.md
│   ├── 04_metrics_report.md
│   ├── 04b_metric_iteration.md
│   ├── 04c_paper_novelty.md
│   └── 05_validation_report.md
├── README.md
└── COST_LOG.md
```

## Reproducibility

```bash
cd scripts/pipeline
python step1_subset.py          # Select embeddings papers
python step2_extract_ideas.py   # LLM extraction (~$6)
python step3_cluster.py         # Embed + cluster
python step4_metrics.py         # Spectrum-based metrics
python step4b_iterate_metrics.py  # Formula comparison
python step4c_paper_novelty.py  # Paper-level novelty (best)
python step5_validate.py        # Golden set + citations
```
