# Validation Report

## Golden Set

### Expected HIGH BPI
| Paper | Year | Citations | BPI | Rank | Percentile |
|-------|------|-----------|-----|------|------------|
| GloVe: Global Vectors for Word Representation... | 2014 | 34143 | 3233.8492 | 2453/5198 | 52.8% |
| Linguistic Regularities in Continuous Space Word R... | 2013 | 3887 | 3897.5382 | 1089/5198 | 79.1% |
| Learning Word Vectors for Sentiment Analysis... | 2011 | 5869 | 3197.7692 | 2576/5198 | 50.5% |
| Convolutional Neural Networks for Sentence Classif... | 2014 | 14058 | 3985.6028 | 1010/5198 | 80.6% |
| Learning Phrase Representations using {RNN} Encode... | 2014 | 25776 | 2802.9100 | 4129/5198 | 20.6% |
| Word Representations: A Simple and General Method ... | 2010 | 2340 | 3912.4665 | 1074/5198 | 79.4% |

### Expected MEDIUM BPI
| Paper | Year | Citations | BPI | Rank | Percentile |
|-------|------|-----------|-----|------|------------|
| Contextual String Embeddings for Sequence Labeling... | 2018 | 1362 | 3287.7517 | 2261/5198 | 56.5% |
| Universal Sentence Encoder for {E}nglish... | 2018 | 987 | 4024.5983 | 973/5198 | 81.3% |
| Dependency-Based Word Embeddings... | 2014 | 1180 | 3518.1960 | 1657/5198 | 68.1% |
| Improving Word Representations via Global Context ... | 2012 | 1274 | 4821.5385 | 428/5198 | 91.8% |

### Expected LOW BPI
| Paper | Year | Citations | BPI | Rank | Percentile |
|-------|------|-----------|-----|------|------------|
| Named Entity Recognition with Bidirectional {LSTM}... | 2016 | 1968 | 3079.9204 | 3022/5198 | 41.9% |
| Relation Classification via Convolutional Deep Neu... | 2014 | 1741 | 3838.3437 | 1153/5198 | 77.8% |
| Knowledge Graph Embedding via Dynamic Mapping Matr... | 2015 | 1672 | 3282.4884 | 2275/5198 | 56.3% |

### Not Found
- [medium] Improved Distributional Similarity with Lessons Learned
- [medium] word2vec Explained

### Golden Set Score
- HIGH papers in top 10%: 0%
- HIGH papers in top 25%: 50%
- HIGH mean percentile: 60.5%

## Citation Bucket Analysis

             bucket  count    mean_BPI  median_BPI     std_BPI   mean_DM     mean_R    mean_C  mean_rho    mean_cit
           0 (zero)    717 4160.320965 4314.960408 1056.081707 10.853275 264.956840 33.570273  2.202108    0.000000
        1-4 (noise)   1244 3317.169042 3133.012081  851.718439  9.954812 251.381099 30.284158  2.354804    2.402733
       5-19 (minor)   1728 3261.418190 3108.869976  771.295276  9.939163 249.833561 29.903501  2.340632   10.214120
     20-99 (decent)   1101 3228.271058 3154.939391  656.051956  9.964577 249.042766 29.692352  2.329571   43.547684
   100-499 (strong)    350 3261.092476 3236.742988  585.534960 10.362229 247.639906 29.348620  2.259296  195.185714
500+ (breakthrough)     58 3373.400781 3313.890610  601.476894 10.691935 247.966513 29.424204  2.197198 2733.275862

## Correlations
- Spearman (BPI vs log_cit): r=-0.179, p=7.89e-39
- Kendall (BPI vs log_cit): τ=-0.117, p=4.42e-35

### Per-Metric Correlations
| Metric | Spearman r | p-value |
|--------|-----------|----------|
| D_M | -0.081 | 5.52e-09 |
| R | -0.239 | 1.57e-68 |
| C | -0.238 | 3.99e-68 |
| rho | 0.059 | 2.39e-05 |
| BPI | -0.179 | 7.89e-39 |

## Top 30 Papers by BPI

| Rank | BPI | Cit | Year | Title |
|------|-----|-----|------|-------|
| 1 | 7800.5617 | 0 | 2023 | Semantic Specialization for Knowledge-based Word Sense  |
| 2 | 7760.4222 | 40 | 2023 | CoCo: Coherence-Enhanced Machine-Generated Text Detecti |
| 3 | 7511.9967 | 0 | 2023 | RetroMAE-2: Duplex Masked Auto-Encoder For Pre-Training |
| 4 | 7057.3259 | 12 | 2024 | Meta-Task Prompting Elicits Embeddings from Large Langu |
| 5 | 7019.3208 | 131 | 2023 | ConFEDE: Contrastive Feature Decomposition for Multimod |
| 6 | 6976.0548 | 4 | 2023 | Graph-based Relation Mining for Context-free Out-of-voc |
| 7 | 6966.3683 | 6 | 2024 | Revisiting Query Variation Robustness of Transformer Mo |
| 8 | 6866.8523 | 0 | 2024 | Can Your Model Tell a Negation from an Implicature? Unr |
| 9 | 6855.7347 | 13 | 2023 | Prompt-based Zero-shot Text Classification with Concept |
| 10 | 6800.3211 | 3 | 2023 | Exploring Enhanced Code-Switched Noising for Pretrainin |
| 11 | 6779.6279 | 3 | 2023 | Entity Contrastive Learning in a Large-Scale Virtual As |
| 12 | 6767.2624 | 7 | 2023 | Bridging the Gap between Synthetic and Natural Question |
| 13 | 6740.9144 | 4 | 2023 | Answer-state Recurrent Relational Network (AsRRN) for C |
| 14 | 6716.1082 | 0 | 2023 | Semantic Frame Induction with Deep Metric Learning |
| 15 | 6713.1189 | 3 | 2023 | Methodological Insights in Detecting Subtle Semantic Sh |
| 16 | 6694.2056 | 5 | 2023 | PCMID: Multi-Intent Detection through Supervised Protot |
| 17 | 6615.8586 | 0 | 2024 | UNSEE: Unsupervised Non-contrastive Sentence Embeddings |
| 18 | 6525.7467 | 0 | 2023 | Swap and Predict – Predicting the Semantic Changes in W |
| 19 | 6508.1579 | 0 | 2025 | MaGiX: A Multi-Granular Adaptive Graph Intelligence Fra |
| 20 | 6487.7517 | 8 | 2023 | Granularity Matters: Pathological Graph-driven Cross-mo |
| 21 | 6448.2021 | 7 | 2023 | Generative Adversarial Training with Perturbed Token De |
| 22 | 6411.8490 | 0 | 2023 | Distinguishability Calibration to In-Context Learning |
| 23 | 6405.9476 | 28 | 2024 | On Fake News Detection with LLM Enhanced Semantics Mini |
| 24 | 6391.4899 | 23 | 2023 | Self-adaptive Context and Modal-interaction Modeling Fo |
| 25 | 6378.0386 | 0 | 2023 | BERM: Training the Balanced and Extractable Representat |
| 26 | 6324.8874 | 6 | 2023 | OssCSE: Overcoming Surface Structure Bias in Contrastiv |
| 27 | 6304.7922 | 0 | 2025 | Evaluating Design Decisions for Dual Encoder-based Enti |
| 28 | 6290.7902 | 7 | 2023 | Example-based Hypernetworks for Multi-source Adaptation |
| 29 | 6282.6622 | 2 | 2023 | Disentangling Aspect and Stance via a Siamese Autoencod |
| 30 | 6276.7390 | 0 | 2023 | RC3: Regularized Contrastive Cross-lingual Cross-modal  |

## Bottom 30 Papers by BPI

| Rank | BPI | Cit | Year | Title |
|------|-----|-----|------|-------|
| 1 | 1192.6031 | 17 | 2016 | Building compositional semantics and higher-order infer |
| 2 | 1221.2500 | 2 | 2017 | Intension, Attitude, and Tense Annotation in a High-Fid |
| 3 | 1236.3085 | 4 | 2019 | Computational Syntax-Semantics Interface with Type-Theo |
| 4 | 1241.9592 | 3 | 2009 | Semantic Representation of Non-Sentential Utterances in |
| 5 | 1288.5619 | 20 | 2019 | {CCG} Parsing Algorithm with Incremental Tree Rotation |
| 6 | 1317.2105 | 3 | 2008 | A Linguistic and Navigational Knowledge Approach to Tex |
| 7 | 1322.2531 | 1 | 2001 | A Decidable Linear Logic for Speech Translation |
| 8 | 1341.9419 | 0 | 2007 | Deep Lexical Semantics: The Ontological Ascent |
| 9 | 1350.6752 | 0 | 2018 | Model-Theoretic Incremental Interpretation Based on {D} |
| 10 | 1391.1135 | 12 | 2019 | Towards Universal Semantic Representation |
| 11 | 1393.1344 | 25 | 2008 | {B}oeing{'}s {NLP} System and the Challenges of Semanti |
| 12 | 1426.6784 | 32 | 1996 | Direct and Underspecified Interpretations of {LFG} f-st |
| 13 | 1429.0296 | 4 | 2010 | Cosubstitution, Derivational Locality, and Quantifier S |
| 14 | 1430.3832 | 8 | 1996 | On Inference-Based Procedures for Lexical Disambiguatio |
| 15 | 1437.7199 | 29 | 2019 | {V}erb{N}et Representations: Subevent Semantics for Tra |
| 16 | 1453.8993 | 1 | 2005 | Structure des repr{\'e}sentations logiques et interface |
| 17 | 1455.5715 | 39 | 1995 | Quantifier Scope and Constituency |
| 18 | 1467.8318 | 2 | 2018 | A Notion of Semantic Coherence for Underspecified Seman |
| 19 | 1471.1703 | 0 | 2007 | Transition and Parsing State and Incrementality in Dyna |
| 20 | 1477.3466 | 7 | 2014 | Propositions, Questions, and Adjectives: a rich type th |
| 21 | 1478.6730 | 0 | 2009 | Towards Establishing a Hierarchy in the {J}apanese Sent |
| 22 | 1480.7431 | 0 | 2017 | Reflexives and Reciprocals in Synchronous {T}ree {A}djo |
| 23 | 1513.6663 | 0 | 2020 | Repr{\'e}sentation s{\'e}mantique des familles d{\'e}ri |
| 24 | 1519.2996 | 9 | 1994 | Syntactic-Head-Driven Generation |
| 25 | 1527.9547 | 1 | 2020 | 基于抽象语义表示的汉语疑问句的标注与分析({C}hinese Interrogative Sentences  |
| 26 | 1532.0328 | 6 | 2018 | A Parser for {LTAG} and Frame Semantics |
| 27 | 1532.2580 | 1 | 2004 | Fine-Grained Lexical Semantic Representations and Compo |
| 28 | 1541.3706 | 1 | 2019 | Presupposition Projection and Repair Strategies in Triv |
| 29 | 1542.3932 | 0 | 1993 | Book Reviews: Functional Grammar in {P}rolog: An Integr |
| 30 | 1546.2699 | 4 | 2013 | Predicative Adjunction in a Modular Dependency Grammar |
