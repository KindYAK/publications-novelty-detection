"""Pipeline configuration — paths, models, hyperparameters."""
import os
import sys
from pathlib import Path

# === Resolve project root & load .env ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# === Paths ===
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "v3"
ACL_PARQUET = DATA_DIR / "acl_all.parquet"
ACL_FULLTEXT = DATA_DIR / "acl_all_fulltext"

EXP_DIR = PROJECT_ROOT / "experiments" / "exp001_embeddings"
EXP_DATA = EXP_DIR / "data"
EXP_REPORTS = EXP_DIR / "reports"
for _d in [EXP_DATA, EXP_REPORTS]:
    _d.mkdir(parents=True, exist_ok=True)

# === Models ===
LLM_MODEL = "gpt-5.4-nano"
LLM_FALLBACK = "gpt-4.1-nano"
LLM_REASONING = "low"  # low / medium / high
EMBEDDING_MODEL = "text-embedding-3-small"

# === Chunking ===
CHUNK_MAX_TOKENS = 3000
CHUNK_OVERLAP_SENTS = 3
MIN_CHUNK_TOKENS = 80  # skip tiny chunks (references, headers)

# === Extraction ===
EXTRACTION_CONCURRENCY = 300
MAX_RETRIES = 3

# === Embedding ===
EMBEDDING_BATCH = 500
EMBEDDING_CONCURRENCY = 10

# === Clustering (search grids) ===
AGGLOM_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
HDBSCAN_MIN_SIZES = [5, 10, 20, 50, 100]
KMEANS_KS = [50, 100, 200, 500, 1000]

# === Metrics ===
K_NEIGHBORS = 10
MIN_PRIOR_PAPERS = 30  # min papers from prior years for metric computation
COV_REG = 1e-4  # covariance regularization

# === Subset keywords (embeddings / representation learning) ===
SUBSET_KEYWORDS = [
    "embedding", "word2vec", "glove", "word vector",
    "word representation", "distributed representation",
    "sentence embedding", "document embedding",
    "contextual embedding", "contextualized representation",
    "representation learning", "dense retrieval",
    "fasttext", "skip-gram", "cbow", "negative sampling",
    "subword embedding", "character embedding",
    "elmo", "sentence-bert", "sbert",
    "text representation", "semantic representation",
    "vector representation", "neural embedding",
    "pretrained representation", "pre-trained representation",
]

# === Golden set: known breakthroughs (title substring → expected tier) ===
# Papers that are IN the ACL subset and represent genuine embedding innovations
GOLDEN_HIGH = [
    "GloVe: Global Vectors",  # GloVe — Pennington 2014, 34K cit
    "Linguistic Regularities in Continuous Space Word Representations",  # Mikolov 2013, 3.9K cit
    "Learning Word Vectors for Sentiment Analysis",  # Maas 2011, 5.9K cit
    "Convolutional Neural Networks for Sentence Classification",  # Kim 2014, 14K cit
    "Learning Phrase Representations using",  # Cho 2014 (GRU/Enc-Dec), 25K cit
    "Word Representations: A Simple and General Method",  # Turian 2010, 2.3K cit
]
GOLDEN_MEDIUM = [
    "Contextual String Embeddings for Sequence Labeling",  # Flair, 1.4K cit
    "Universal Sentence Encoder",  # USE, 987 cit
    "Improved Distributional Similarity with Lessons Learned",  # Levy 2015, 1.4K cit
    "Dependency-Based Word Embeddings",  # Levy & Goldberg 2014, 1.2K cit
    "word2vec Explained",  # Goldberg & Levy, 1.7K cit
    "Improving Word Representations via Global Context",  # Huang 2012, 1.3K cit
]
# Papers that USE embeddings but aren't embedding innovations → should NOT rank highly
GOLDEN_LOW = [
    "Named Entity Recognition with Bidirectional",  # Application paper, uses embeddings
    "Relation Classification via Convolutional Deep Neural Network",  # Application
    "Knowledge Graph Embedding via Dynamic Mapping Matrix",  # KG, not text embedding innovation
]

# === LLM prompt ===
SYSTEM_PROMPT = "You extract scientific ideas from paper excerpts. Output JSON only."

USER_PROMPT_TEMPLATE = """Extract novel ideas from this scientific paper excerpt.
- Return JSON: {{"ideas": ["idea1", "idea2", ...]}}
- 0-5 ideas, each 5-20 words, specific
- Focus on novel methods, techniques, models, findings
- Skip background, related work, references

Title: {title}

{chunk}"""
