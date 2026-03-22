# Data Collection: Methodology & Decisions

## Data Source

**Primary**: [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/)
- Free, no API key required (1 req/sec rate limit; with key: higher)
- 214M+ papers indexed
- Provides: title, abstract, citation count, influential citation count, arXiv ID, venue, year, fields of study
- Merges preprint + published versions automatically (arXiv paper and its journal version share one citation count)

**Secondary** (for full text): [arXiv](https://arxiv.org/)
- Free, LaTeX source available for most CS papers
- Used only when we need full document text beyond abstract
- Not all papers have arXiv versions

## How Papers Are Selected

### Search Strategy

Semantic Scholar's bulk search endpoint (`/paper/search/bulk`) matches keywords against **title + abstract**. It does NOT support arXiv category filtering (e.g., `cs.CL`) directly. We approximate this with:

1. **Targeted keyword queries** — each designed to capture a sub-area of ML/NLP/LLM research
2. **Field-of-study filter** — `fieldsOfStudy=Computer Science` (Semantic Scholar's own classifier)
3. **Year range filter** — per query, to focus on the relevant era for each topic
4. **Deduplication** — papers appearing in multiple queries are kept once (by `paperId`)

### Query Table

| Query | Year Range | Rationale |
|---|---|---|
| `transformer attention mechanism` | 2017-2021 | Core transformer era |
| `BERT language model pretraining` | 2018-2022 | Pretrained LM wave |
| `word embeddings representation learning` | 2013-2019 | Pre-transformer NLP |
| `neural machine translation sequence to sequence` | 2014-2020 | Seq2seq era |
| `text classification sentiment analysis deep learning` | 2015-2020 | Applied NLP |
| `deep learning convolutional neural network image` | 2014-2020 | Core ML/CV |
| `generative adversarial network GAN` | 2014-2020 | Generative models |
| `reinforcement learning policy gradient` | 2015-2020 | RL |
| `graph neural network` | 2017-2021 | GNN wave |
| `large language model GPT generation` | 2019-2023 | LLM era |
| `prompt engineering in-context learning` | 2020-2023 | Prompt-based methods |
| `instruction tuning RLHF alignment` | 2021-2023 | Alignment era |

Each query returns up to 200 papers (API default page). Results are deduplicated across queries.

### What Fields We Fetch Per Paper

| Field | Why |
|---|---|
| `title` | Display, search |
| `abstract` | Primary text for pipeline (segmentation + idea extraction) |
| `year` | Temporal ordering for retrospective experiment |
| `venue` | Quality signal (NeurIPS vs. predatory journal) |
| `publicationDate` | Precise temporal ordering |
| `citationCount` | **Primary ground truth proxy for "impact"** |
| `influentialCitationCount` | Citations where the paper was heavily used (not just mentioned) — better quality signal |
| `externalIds` | arXiv ID for full text download; DOI for linking |
| `s2FieldsOfStudy` | Verify CS papers; detect interdisciplinary work |
| `openAccessPdf` | Link to free PDF if no LaTeX source |

## Current Dataset: `papers_ml_nlp` (v1)

### Stats

- **1,664 papers** (deduplicated across 12 queries)
- **100% have abstracts** (filtered)
- **58% have arXiv IDs** (965 papers)
- **Year range**: 2014-2023
- **Avg abstract length**: 1,309 chars

### Citation Distribution

```
Bucket        Count    %
0-10:             0    0.0%   ← PROBLEM: no low-citation papers
10-20:          389   23.4%
20-50:          542   32.6%
50-100:         275   16.5%
100-500:        365   21.9%
500-1k:          45    2.7%
1k-5k:           38    2.3%
5k+:             10    0.6%
```

### Known Issues (v1)

1. **BIAS: No low-citation papers.** We used `min_citation_count=10`, so 0-citation papers (majority of all published papers) are excluded. For BPI validation, we need the full distribution — breakthrough papers should rank ABOVE ordinary/low-citation papers.

2. **Search bias.** Bulk search returns papers sorted by relevance, not randomly. Papers with popular keywords in title/abstract are overrepresented. Niche/novel papers (the ones BPI should find!) may be underrepresented.

3. **Survivorship bias.** We're only seeing papers that Semantic Scholar indexed. Workshop papers, rejected submissions, and withdrawn preprints are mostly absent.

4. **Temporal coverage is uneven.** 2019-2020 dominate (47.5% of dataset). Early years (2014-2015) are sparse.

5. **No full text yet.** Only abstracts collected. Full LaTeX text download is implemented but not run at scale.

## Planned Fix: v2 Collection

### Strategy Change

For BPI validation, we need a **corpus that resembles real-world distribution**:
- Many low/zero-citation papers (the "background")
- Some medium-citation papers (solid but not breakthrough)
- Few high-citation papers (the "breakthroughs" we want BPI to detect)

**New approach**: Pick a narrow time window + venue/topic and collect ALL papers, not just keyword matches.

Options:
1. **arXiv cs.CL, one year** — download the full listing of all papers in cs.CL for 2016 or 2017, then enrich with citation counts from Semantic Scholar
2. **Conference proceedings** — all papers from ACL/EMNLP/NeurIPS for a given year (Semantic Scholar supports venue search)
3. **Semantic Scholar bulk download** — use their dataset API for a full dump of a field

### v2 Result (COLLECTED)

| Property | Value |
|---|---|
| Scope | ML/NLP/DL papers, 2016-2018 |
| Source for paper IDs + citations | Semantic Scholar bulk search |
| Source for text | Abstracts from SS; full LaTeX from arXiv on demand |
| Total papers | **2,458** (with abstracts) |
| With arXiv IDs | 1,367 (55.6%) |
| Citation range | **0 to 169,082** (full distribution) |

### v2 Collection Method: 3-tier strategy

The key insight: a single query with no citation filter gives you mostly mid-range papers. To get the full distribution, we use **3 tiers**:

**Tier 1 — Background (no citation filter)**: 16 NLP/ML topic queries (machine translation, text classification, NER, QA, sentiment analysis, language model, word embeddings, relation extraction, text generation, semantic parsing, reading comprehension, dialogue systems, neural network training, RNN, attention mechanism, transfer learning). Each returns 200 papers. Purpose: capture ordinary papers with 0-10 citations.

**Tier 2 — Mid-impact (min_citation_count=50)**: 8 broader queries (NLP deep learning, representation learning, seq2seq, CNN text, GAN, RL, GNN, VAE). Purpose: fill the 50-500 citation range with solid papers.

**Tier 3 — High-impact (min_citation_count=500)**: 5 broad queries (deep learning, neural network, attention transformer, generative model, NLP). Purpose: ensure known breakthroughs are included.

All queries filtered to `fieldsOfStudy=Computer Science`, year range `2016-2018`. Deduplicated by Semantic Scholar `paperId`.

### v2 Citation Distribution

```
Bucket        Count    %     Description
    0:          104    4.2%  Zero citations — unpopular/niche
  1-4:          254   10.3%  Barely cited
  5-9:          184    7.5%  Low impact
10-19:          164    6.7%  Below average
20-49:          179    7.3%  Average
50-99:          444   18.1%  Solid papers
100-499:        568   23.1%  Well-known
500-999:        308   12.5%  Highly cited
1k-4999:        231    9.4%  Very influential
5000+:           22    0.9%  Breakthroughs
```

542 papers with < 10 citations (background), 561 papers with >= 500 citations (breakthrough candidates). This is the distribution needed for BPI validation.

### v2 Top Papers

| Citations | Year | Title |
|---|---|---|
| 169,082 | 2017 | Attention is All you Need |
| 33,932 | 2016 | Semi-Supervised Classification with Graph Convolutional Networks |
| 25,075 | 2017 | Graph Attention Networks |
| 19,372 | 2016 | TensorFlow: A system for large-scale machine learning |
| 10,615 | 2017 | Improved Training of Wasserstein GANs |
| 10,581 | 2016 | Enriching Word Vectors with Subword Information |
| 10,065 | 2016 | Improved Techniques for Training GANs |
| 9,383 | 2018 | How Powerful are Graph Neural Networks? |
| 9,143 | 2016 | SQuAD: 100,000+ Questions for Machine Comprehension of Text |

## How Citations Are Used as Ground Truth

Citations are a **proxy**, not perfect truth. Known limitations:
- **Matthew effect**: famous authors get more citations regardless of quality
- **Review papers** get high citations but aren't breakthroughs
- **Negative citations**: a paper can be cited because it's wrong
- **Field size**: a mediocre ML paper gets more citations than a great topology paper

Mitigations:
- Use `influentialCitationCount` (Semantic Scholar's filtered metric) alongside raw count
- Consider using **citation percentile within year** rather than absolute count
- Optionally filter out survey/review papers (by title keywords)
- The retrospective design (measure citations 5+ years after publication) reduces recency bias

## v3 Dataset (CURRENT — Collected March 2026)

### Scope

**arXiv-based**: All papers from cs.CL, cs.LG, cs.CV, cs.AI (2016-2026)

| Property | Value |
|---|---|
| Total papers | **442,507** |
| With SS citation data | 439,827 (99.4%) |
| With full text | 440,909 (99.64%) |
| Year range | 2016-2026 |
| Categories | cs.CL, cs.LG, cs.CV, cs.AI |

**Full-text sources**: HTML (ar5iv) 404,519 / PDF 29,025 / LaTeX 7,365

### v3 Collection Method

1. **arXiv API** — exhaustive per-category per-quarter collection using `submittedDate` ranges
   - Quarter splitting to stay under arXiv's 10K pagination limit
   - `submittedDate` format: `YYYYMMDDHHMM` (NO `T` separator)
   - Caches: `data/processed/v3/arxiv_ids_{category}_{year}[_Q{n}].json`
2. **Semantic Scholar batch API** — `/paper/batch` with 450 IDs/request for citation enrichment
   - 99.4% match rate
3. **Full-text collection** — async multi-strategy pipeline:
   - ar5iv HTML (primary, ~91% coverage, fast)
   - arXiv PDF via PyMuPDF (fallback, ~7% additional)
   - arXiv LaTeX source (supplementary, ~2%)
   - PDF `v1` suffix fix for older papers returning 404

### Citation Distribution (v3)

| Bucket | Count | % |
|---|---|---|
| 0 | 70,434 | 15.9% |
| 1-4 | 107,521 | 24.3% |
| 5-19 | 112,388 | 25.4% |
| 20-99 | 86,789 | 19.6% |
| 100-499 | 47,132 | 10.7% |
| 500-999 | 8,918 | 2.0% |
| 1000+ | 7,325 | 1.7% |

Mean: 41.9, Median: 5.0, Max: 169,119 (Attention Is All You Need)

## Embeddings-focused Supplement (ACL Anthology)

### Purpose

Targeted collection to capture the full **embeddings revolution timeline**:

```
static embeddings (Word2Vec, GloVe, FastText)
  → contextual embeddings (ELMo, Flair)
    → transformer representations (BERT, GPT, RoBERTa)
      → sentence embeddings (Sentence-BERT, SimCSE)
        → retrieval embeddings (DPR, ColBERT, E5, BGE)
```

### Search Strategy

**Keywords** (matched against title + abstract):
- Static: `word embedding`, `word2vec`, `glove`, `fasttext`, `distributional semantics`, `static embedding`
- Contextual: `contextual embedding`, `contextualized representation`, `ELMo`, `deep contextualized`
- Transformer: `BERT embedding`, `transformer representation`, `pre-trained language model`
- Sentence: `sentence embedding`, `sentence-BERT`, `SimCSE`, `contrastive sentence`, `semantic textual similarity`
- Retrieval: `text embedding`, `dense retrieval`, `bi-encoder`, `dual-encoder`, `ColBERT`, `embedding model`

### Sources

| Source | Papers | Coverage |
|---|---|---|
| **HuggingFace ACL Anthology** (`ACL-OCL/acl-anthology-corpus` parquet) | ~5,838 | Pre-2022, ACL/EMNLP/NAACL/COLING/TACL + all venues with keyword match |
| **`acl-anthology` Python package** | ~1,832 | 2023-2025 ACL/EMNLP/NAACL/COLING/TACL/Findings |
| **Semantic Scholar search** (gap-filling) | ~445 | Papers from non-ACL venues not in arXiv |
| **Pre-2016 & conference-only landmarks** | 10 | Word2Vec, GloVe, Flair, attention mechanism |

After deduplication against v3 dataset: **~3,904 new papers** (embeddings-focused only)

## Full ACL Anthology Collection (All NLP)

### Purpose

Fills the gap of conference-only NLP papers not posted to arXiv. Covers ALL topics, not just embeddings.

### Sources

| Source | Papers | Notes |
|---|---|---|
| **HuggingFace ACL Anthology** parquet (pre-2022) | ~59,408 | Full text included via GROBID extraction |
| **`acl-anthology` package** (2023-2025) | ~8,658 | PDFs from aclanthology.org |
| After self-dedup | **~67,213** | |

### Coverage

- **Citation verification**: 63,399 verified via SS CorpusId (94.3%)
- **Year range**: 1963-2026
- Venues: ACL, EMNLP, NAACL, EACL, COLING, TACL, Findings, LREC, workshops
- Script: `scripts/collect_acl_all.py`
- Output: `data/processed/v3/acl_all.parquet`
- Full text: `data/processed/v3/acl_all_fulltext/`

### Landmark Papers Explicitly Included

| Paper | Year | arXiv/Venue | Citations |
|---|---|---|---|
| Word2Vec (Efficient Estimation) | 2013 | 1301.3781 | 33,862 |
| Word2Vec (Skip-gram) | 2013 | 1310.4546 | 34,937 |
| GloVe | 2014 | EMNLP (D14-1162) | 34,142 |
| Skip-Thought Vectors | 2015 | 1506.06726 | 2,473 |
| Bahdanau Attention | 2014 | 1409.0473 | 29,045 |
| Show, Attend and Tell | 2015 | 1502.03044 | 10,678 |
| Flair | 2018 | COLING (C18-1139) | 1,362 |

### Full Text

- HuggingFace corpus provides GROBID-extracted text for ~5,796 papers
- arXiv papers fetched via ar5iv HTML / PDF / LaTeX pipeline
- ACL-only papers fetched via ACL Anthology PDF download + PyMuPDF extraction

## File Locations

```
data/processed/v3/
├── dataset_v3.parquet           # v3 main dataset (442,507 papers)
├── dataset_v3.csv               # CSV version
├── arxiv_ids_all.parquet        # All arXiv IDs with metadata
├── ss_partial.json              # Raw SS API responses
├── fulltext_progress.json       # Full-text collection progress tracker
├── fulltext/                    # ~440K individual text files
├── embeddings_acl.parquet       # Embeddings supplement (~3,904 papers)
├── embeddings_acl_fulltext/     # ACL embeddings full texts
├── acl_all.parquet              # Full ACL Anthology (~67,213 papers, all NLP)
├── acl_all_fulltext/            # ACL all-NLP full texts
├── acl_all_step1_hf.json        # Cached: HF ACL Anthology (all NLP)
├── acl_all_step2_recent.json    # Cached: Recent ACL 2023+ (all NLP)
├── embeddings_acl_step1_hf.json        # Cached: HF ACL Anthology (embeddings)
├── embeddings_acl_step2_landmarks.json # Cached: Pre-2016 landmarks
├── embeddings_acl_step3_recent.json    # Cached: Recent ACL (2023+, embeddings)
├── embeddings_acl_step4_ss.json        # Cached: SS gap-filling
└── arxiv_ids_{cat}_{year}[_Q{n}].json  # Per-category-year caches

data/processed/
├── papers_v2_full.csv      # v2 dataset (2,458 papers) — superseded by v3
├── papers_v2_full.json     # Same, richer format
├── papers_ml_nlp.csv       # v1 dataset (1,664 papers) — superseded
└── papers_ml_nlp.json      # Same, richer format

scripts/
├── collect_v3_comprehensive.py         # v3 arXiv + SS collection
├── collect_fulltext.py                 # v3 full-text pipeline (async)
├── collect_embeddings_acl.py           # Embeddings ACL supplement
├── collect_acl_all.py                  # Full ACL Anthology (all NLP)
├── fill_embeddings_gaps.py             # Gap-fill fulltext + citations
├── collect_v2_full_distribution.py     # v2 collection (3-tier strategy)
├── collect_ml_nlp.py                   # v1 collection script
├── smoke_test_api.py                   # API verification
└── audit_dataset.py                    # Distribution analysis

src/novelty_search/data/
├── semantic_scholar.py     # SS API client
├── arxiv_text.py           # arXiv LaTeX downloader + stripper
└── collector.py            # High-level collection logic

docs/
└── data_collection.md      # THIS FILE
```
