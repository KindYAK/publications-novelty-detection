# Clustering Comparison Report

- Total unique ideas: 62750
- Embedding dims: 1536
- PCA: 256 dims (79.3% variance)

| Method | Param | K | Silhouette | Max | Median | Min | Time |
|--------|-------|---|-----------|-----|--------|-----|------|
| kmeans | k=50 | 50 | 0.0161 | 2254 | 1201 | 725 | 4.8s |
| kmeans | k=100 | 100 | 0.0140 | 1135 | 611 | 227 | 7.2s |
| kmeans | k=200 | 200 | 0.0096 | 560 | 313 | 89 | 10.9s |
| kmeans | k=500 | 500 | -0.0050 | 352 | 121 | 1 | 44.3s |
| kmeans | k=1000 | 1000 | -0.0239 | 285 | 49 | 1 | 91.8s |

## Selected: **kmeans(k=100)**
- Clusters: 100
- Silhouette: 0.0140

## Sample Cluster Descriptions (top 30 by size)

- **[40]** (n=1135): Design targets efficient multilingual transfer without language-specific training data.
- **[95]** (n=1004): Compare explicit ~100 morphosyntactic/syntactic features vs BERT embeddings
- **[32]** (n=964): Introduce margin-aware contrastive loss separating positive/negative pairs and controlling intra-group similarity.
- **[82]** (n=958): Composite embedding method whose best-performing choice depends on dataset agreement level.
- **[22]** (n=949): Use locally trained CBOW Word2Vec embeddings without external pretrained vectors for context overlap.
- **[67]** (n=889): Relation-aware entity encoding focusing query-specific subgraph features
- **[24]** (n=886): Combined loss LCE+LfCE+LKL+LSCL showing improved clustering under low known-intent ratios.
- **[46]** (n=876): Perform error analysis on the best-performing embedding model and assess domain-specific versus generalized embeddings.
- **[34]** (n=835): Show large gains on the 'ambiguous' label versus BERT/SVM, exceeding 3% F1 improvement.
- **[33]** (n=829): Jointly models cue-conditioned negation/speculation scope with syntax-informed POS/DEP/PATH features.
- **[44]** (n=828): Propose multilabel LRC formulation to mitigate polysemy-related ambiguity in predictions.
- **[48]** (n=821): SimAlign performs high-quality word alignment without parallel training using static/contextual embeddings.
- **[79]** (n=821): Define heuristic imbalance coefficient via CMDS-g cluster removal to simulate realistic skewed fine-tuning data.
- **[21]** (n=817): Plug-and-play Contrastive Prompting steers LLM sentence embeddings at inference-time
- **[94]** (n=806): Introduces relation-sensitive graph attention combining word-pair and edge-description embeddings.
- **[47]** (n=799): Apply low-rank approximations of covariance representations, using 3–4× mean-vector size.
- **[36]** (n=791): Use semantic construction operators to decouple grammar from multiple semantic formalisms
- **[12]** (n=788): Uses discourse-dimension classifiers (causal explanations, counterfactuals, dissonance/consonance) with nearby-message pairing to interpret improvements.
- **[49]** (n=787): Custom Audio Encoder uses contrastive learning to align singing,speech,text prompt representations.
- **[54]** (n=762): Embedding evaluation via averaging DiffCSE similarities across first 16 ranking outputs
- **[80]** (n=761): Test whether improved structural predictability from refined models better explains that-omission.
- **[71]** (n=759): Subtask E: sentiment-strength regression using expanded term lexicons plus word embedding vectors
- **[25]** (n=756): Jointly learn embeddings for words and dependency-context structural features without separating parameters.
- **[56]** (n=756): WordNet synset ID replacement for content words to reduce OOV impact, with POS-filtered intra-category sense selection.
- **[75]** (n=754): Adds self-attention after cross-sentence interaction to inject global context into attention-aware representations.
- **[58]** (n=746): Multi-stage training: distant supervision on pairs, rule-based noisy long-context, then LLM-synthesized fine-tuning
- **[5]** (n=742): Instruction-finetuned “one embedder” approach for task-agnostic embedding transfer.
- **[15]** (n=736): Final training optimizes supervised CE jointly with weighted (λ) hierarchical contrastive losses.
- **[51]** (n=733): Retrofit Exchange with stochastic merging to escape local maxima during iterative AMI optimization.
- **[10]** (n=720): Sentence similarity implemented using three published metrics plus neighbor-distribution comparison measures.
