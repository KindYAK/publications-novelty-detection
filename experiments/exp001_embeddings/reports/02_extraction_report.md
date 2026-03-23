# Idea Extraction Report

## Config
- Model: gpt-5.4-nano
- Concurrency: 300
- Chunk size: 3000 tokens
- Test mode: full

## Results
- Papers processed: 5,234
- Chunks processed: 13,522
- Total ideas: 62,759
- Unique ideas: 62,751
- Papers with ideas: 5,232
- Avg ideas/chunk: 4.6
- Errors: 193

## Cost
- Tokens in: 36,680,900
- Tokens out: 1,468,552
- Elapsed: 332s

## Sample Ideas
- UGs modify Conceptual Graphs to model valency-based linguistic predicates without reification
- Define Unit Graphs over support (T,C,M) with hierarchical unit types, circumstantial symbols, identifiers
- Introduce actant-slot signatures with obligatory/prohibited/optional slots to constrain graph structure
- Provide two formal semantics for UGs: closure/homomorphism semantics and model-theoretic semantics
- Create deep-semantic MTT representation where specialization follows actantial structure hierarchy, not meaning hierarchy
- Model relation extraction as analogical similarity between entity-pair mention sets, not relation classification.
- Build millions of analogy training pairs via KB triple matching with web corpora using distant supervision.
- Use hierarchical siamese network with attention at word and mention levels to encode multi-instance mention sets.
- Introduce one-shot relation type task using analogy embeddings from a single unseen example per relation type.
- Pretrained analogy embeddings transferred into convolutional RE model, outperforming state-of-the-art on benchmark datasets.
- Bayesian probabilistic tensor factorization fuses multiple word-similarity matrices as tensor slices.
- Learns a single latent word vector plus per-perspective linear transformations to reconstruct each slice.
- Uses missing-entry indicators I_kij to model unknown relatedness without forcing zero entries.
- Single-stage fusion recreates lexicon-based synonym/antonym signals using distributional perspectives via regularized transformations.
- Evaluates recreated synonym/antonym relatedness on GRE antonym questions, achieving state-of-the-art performance.
- Model item survival by predicting both P-value range and positive discrimination Rb from MCQ text.
- Use 113 engineered linguistic complexity features as a strong survival-prediction baseline.
- Represent MCQ semantics with multiple embedding types to capture expert-knowledge question difficulty.
- Incorporate information-retrieval-inspired features to estimate survival likelihood beyond text complexity.
- Transaction/sequence text representation for request politeness keyword co-occurrence
