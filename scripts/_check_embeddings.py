import pandas as pd
import json
import re
from pathlib import Path

df = pd.read_parquet('data/processed/v3/dataset_v3.parquet')

# Search titles and abstracts for embedding-related keywords
keywords = [
    'contextual embedding', 'contextualized embedding',
    'contextualized word representation', 'contextual word representation',
    'BERT embedding', 'ELMo', 'sentence embedding', 'sentence representation',
    'contrastive.*sentence', 'word2vec', 'word embedding',
    'GloVe', 'fastText', 'transformer representation',
    'retrieval embedding', 'dense retrieval', 'dense passage',
    'SimCSE', 'SentenceBERT', 'Sentence-BERT', 'InferSent',
    'universal sentence encoder', 'skip-thought',
    'static embedding', 'pre-trained embedding',
    'BERT', 'RoBERTa', 'XLNet', 'ALBERT', 'DeBERTa',
    'contrastive learning.*embedding', 'embedding.*contrastive',
    'E5 ', 'BGE ', 'GTE ', 'Instructor embedding',
    'text embedding', 'learned representation',
]

title_col = 'title'
abstract_col = 'abstract'

results = {}
for kw in keywords:
    pattern = re.compile(kw, re.IGNORECASE)
    title_match = df[title_col].fillna('').apply(lambda x: bool(pattern.search(x)))
    abstract_match = df[abstract_col].fillna('').apply(lambda x: bool(pattern.search(x)))
    combined = title_match | abstract_match
    count = combined.sum()
    if count > 0:
        results[kw] = count

print("=== Keyword coverage in existing dataset ===")
for kw, count in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {kw}: {count:,} papers")

print(f"\nTotal unique papers: {len(df):,}")
print(f"\nColumns: {list(df.columns)}")

print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")

landmarks = [
    'Attention Is All You Need',
    'BERT: Pre-training of Deep',
    'ELMo',
    'Sentence-BERT',
    'SimCSE',
    'Dense Passage Retrieval',
    'Distributed Representations of Words',
    'GloVe: Global Vectors',
    'Universal Sentence Encoder',
    'InferSent',
    'Contrastive Learning of',
    'E5: Text Embeddings',
    'Improving Text Embeddings with Large Language',
    'BGE M3',
    'Instructor',
    'ColBERT',
    'Matryoshka Representation',
]

print("\n=== Landmark papers check ===")
for title_part in landmarks:
    matches = df[df['title'].fillna('').str.contains(title_part, case=False)]
    if len(matches) > 0:
        for _, row in matches.head(3).iterrows():
            cites = row.get('citationCount', 'N/A')
            print(f"  FOUND: {row['title'][:80]}... (citations: {cites})")
    else:
        print(f"  MISSING: '{title_part}'")
