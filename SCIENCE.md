# The Science of Probing

This document provides general background on probing as a technique in NLP and machine learning. For the specific implementation in this project, see [PROBE.md](PROBE.md).

## What is a Probe?

A **probe** (also called a "diagnostic classifier") is a simple model trained to extract information from another model's internal representations. The core idea:

> If a simple classifier can extract information X from a model's hidden states, then the model has learned to represent X.

Probes are typically linear classifiers or shallow MLPs. The simplicity is intentional - if extraction requires a complex model, the information may not be explicitly encoded.

## History and Key Papers

### Origins: Testing What Models Know

**Alain & Bengio, 2016** - "Understanding intermediate layers using linear classifier probes"
- Attached linear classifiers to each layer of deep networks
- Found that class information becomes more linearly separable in deeper layers
- Established probing as a tool for understanding representations

**Belinkov et al., 2017** - "What do Neural Machine Translation Models Learn about Morphology?"
- Probed NMT encoder representations for part-of-speech and morphological features
- Showed that different layers encode different linguistic properties
- Lower layers: surface features; higher layers: semantic features

**Conneau et al., 2018** - "What you can cram into a single $&!#* vector"
- Systematic probing of sentence embeddings
- 10 probing tasks: length, word content, syntax, semantics
- Benchmark for comparing representation quality

### Structural Probes

**Hewitt & Manning, 2019** - "A Structural Probe for Finding Syntax in Word Representations"
- Probed for entire parse trees, not just labels
- Found that syntax trees are encoded as geometric structure
- Distances between words in hidden space reflect tree distances

### Span Extraction

**Pointer Networks (Vinyals et al., 2015)** - "Pointer Networks"
- Networks that point to positions in the input rather than generating from vocabulary
- Key insight: output vocabulary can be the input positions
- Foundation for extractive QA and span-based NER

**BERT for QA (Devlin et al., 2018)**
- Start/end token prediction for extractive question answering
- Linear heads over BERT hidden states
- Demonstrated that pre-trained models encode answer spans

## Why Linear Probes Work

### The Linear Separability Hypothesis

Large language models organize their latent space such that semantic concepts are **linearly separable**. This means:

1. Similar concepts cluster together (cities near cities, times near times)
2. Different concepts are separated by hyperplanes
3. A linear classifier just needs to find these hyperplanes

### Evidence

- Linear probes achieve near-optimal accuracy on many tasks
- Adding nonlinearity (MLPs) provides marginal improvement
- This suggests the information is explicitly encoded, not requiring computation to extract

### Intuition

Think of the LLM as a person who knows the answer but speaks a foreign language. The probe is a translator learning to read what the LLM already knows. The probe doesn't create new knowledge - it learns to decode existing representations.

## Types of Probes

### Classification Probes

Predict a discrete label from a hidden state.

```
hidden_state [4096] → Linear(4096, num_classes) → class_logits
```

Examples:
- Part-of-speech tagging
- Named entity type classification
- Sentiment classification
- Slot presence detection (this project's NullHead)

### Span Extraction Probes

Predict start and end positions within a sequence.

```
query_hidden [4096] + all_hidden [T, 4096] → start_position, end_position
```

Examples:
- Extractive question answering
- Named entity boundary detection
- Slot value extraction (this project's PointerHead)

### Structural Probes

Recover structured representations (trees, graphs) from hidden states.

```
all_hidden [T, 4096] → distance_matrix [T, T] → parse_tree
```

Examples:
- Dependency parsing
- Constituency parsing
- Semantic role labeling

## Methods

### Training a Probe

1. **Freeze the base model** - Don't update LLM weights
2. **Extract hidden states** - Run inputs through frozen model
3. **Train probe on hidden states** - Standard supervised learning
4. **Evaluate** - If probe succeeds, information was encoded

### The LoRA Variation

Instead of fully freezing the base model, use Low-Rank Adaptation (LoRA):
- Add small trainable matrices to attention layers
- Base weights remain frozen
- Model learns to better surface information for the probe

This project uses LoRA because the base model wasn't trained for our specific task format.

### Bilinear Attention for Span Extraction

For pointing to positions, use bilinear attention:

```
score(query, position) = query @ W @ hidden[position]
```

This learns a compatibility function between the query (what we're looking for) and each position (where it might be).

## Applied Problems

### Slot Filling / Named Entity Recognition

Extract structured values from text:
- "Book a flight to **Boston**" → destination: Boston
- Probes point to entity boundaries in the input

### Extractive Question Answering

Find answer spans in a passage:
- Question: "Where was Einstein born?"
- Passage: "Einstein was born in **Ulm, Germany** in 1879."
- Probe points to start/end of answer span

### Information Extraction

Extract relations and attributes:
- "**Apple** was founded by **Steve Jobs**"
- Probe identifies entity spans and their relationships

### Interpretability Research

Understand what models learn:
- Which layer encodes syntax vs semantics?
- Do models learn linguistic structure or surface patterns?
- How does information flow through layers?

## Advantages of Probing

| Aspect | Generation | Probing |
|--------|------------|---------|
| Speed | Slow (autoregressive) | Fast (single forward pass) |
| Failures | JSON errors, hallucinations | None (direct extraction) |
| Interpretability | Opaque | Explicit (positions/classes) |
| Training | Full model or heads | Small classifiers |

## Limitations

1. **Can only extract what's encoded** - If the model doesn't represent information, probes can't find it

2. **Format sensitivity** - Probes are trained on specific input formats; may not generalize

3. **Linear separability assumption** - If information requires nonlinear extraction, linear probes underestimate model knowledge

4. **Correlation vs causation** - High probe accuracy shows correlation, not that the model "uses" the information

## References

- Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644
- Belinkov, Y., et al. (2017). What do Neural Machine Translation Models Learn about Morphology? ACL 2017
- Conneau, A., et al. (2018). What you can cram into a single $&!#* vector. ACL 2018
- Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. NAACL 2019
- Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS 2015
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. NAACL 2019
