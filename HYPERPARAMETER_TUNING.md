# Hyperparameter Tuning Log

## Overview

This document records all hyperparameter configurations explored during the development of the Transformer-based RAG pipeline. Each configuration was evaluated on validation set performance, with the best values selected for the final model.

---

## Encoder Hyperparameter Tuning

### Experiment 1: Baseline Configuration
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.2
learning_rate: 1e-4
batch_size: 64
max_seq_len: 256
epochs: 10

text
**Result:** ❌ Training failed - NaN loss after first epoch  
**Cause:** Poor weight initialization causing exploding gradients  
**Action:** Added Xavier uniform initialization, gradient clipping

---

### Experiment 2: Stabilized Model
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 256
dropout: 0.1
learning_rate: 1e-4
batch_size: 64
max_seq_len: 256
epochs: 10

text
**Result:** ✅ Training successful  
**Validation Sentiment Accuracy:** 83.13%  
**Validation Helpfulness Accuracy:** 71.11%  
**Notes:** Stable training, good convergence. Selected as best encoder configuration.

---

### Experiment 3: Larger Model (Encoder)
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.1
learning_rate: 5e-5
batch_size: 32
max_seq_len: 256
epochs: 10

text
**Result:** ⚠️ Training successful but overfitting  
**Validation Sentiment Accuracy:** 81.25%  
**Validation Helpfulness Accuracy:** 69.45%  
**Notes:** More parameters but worse generalization. Training time 3x longer.

---

### Experiment 4: Lower Learning Rate
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 256
dropout: 0.1
learning_rate: 5e-5
batch_size: 64
max_seq_len: 256
epochs: 10

text
**Result:** ⚠️ Slow convergence  
**Validation Sentiment Accuracy:** 80.12%  
**Notes:** More stable but requires more epochs to reach same accuracy.

---

### Experiment 5: Higher Dropout
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 256
dropout: 0.3
learning_rate: 1e-4
batch_size: 64
max_seq_len: 256
epochs: 10

text
**Result:** ⚠️ Underfitting  
**Validation Sentiment Accuracy:** 78.90%  
**Notes:** Too much regularization, model struggles to fit training data.

---

### Experiment 6: Shorter Sequences
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 256
dropout: 0.1
learning_rate: 1e-4
batch_size: 64
max_seq_len: 128
epochs: 10

text
**Result:** ✅ Fast training, slightly lower accuracy  
**Validation Sentiment Accuracy:** 81.50%  
**Notes:** Good for rapid prototyping. Loss of long-range context.

---

## Decoder Hyperparameter Tuning

### Experiment 7: Initial Decoder (Small)
d_model: 128
n_heads: 4
n_layers: 2
d_ff: 256
dropout: 0.1
learning_rate: 5e-5
batch_size: 16
max_len: 384
epochs: 5
training_samples: 5,000

text
**Result:** ❌ Very high perplexity  
**Test Perplexity:** 433.94  
**Generation Quality:** Model copies input text instead of generating summaries  
**Notes:** Too small for generation task, insufficient training data.

---

### Experiment 8: Larger Decoder (More Data)
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.1
learning_rate: 3e-4
batch_size: 32
max_len: 384
epochs: 10
training_samples: 25,200 (full)

text
**Result:** ✅ Significant improvement  
**Test Perplexity:** 40.34  
**Generation Quality:** Produces novel text, more diverse vocabulary  
**Notes:** Selected as best decoder configuration.

---

### Experiment 9: Decoder with Lower LR
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.1
learning_rate: 1e-4
batch_size: 32
max_len: 384
epochs: 10
training_samples: 25,200

text
**Result:** ⚠️ Slower convergence  
**Test Perplexity:** 52.18  
**Notes:** Requires more epochs to reach lower perplexity.

---

### Experiment 10: Decoder with Higher LR
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.1
learning_rate: 5e-4
batch_size: 32
max_len: 384
epochs: 10
training_samples: 25,200

text
**Result:** ❌ Unstable training  
**Test Perplexity:** 89.45  
**Notes:** Loss oscillates, validation PPL shows high variance.

---

### Experiment 11: Decoder with LR Scheduler
d_model: 256
n_heads: 8
n_layers: 4
d_ff: 512
dropout: 0.1
learning_rate: 3e-4 (with ReduceLROnPlateau)
batch_size: 32
max_len: 384
epochs: 10
training_samples: 25,200

text
**Result:** ✅ Best configuration  
**Test Perplexity:** 40.34  
**Best Validation PPL:** 37.81 (Epoch 4)  
**Notes:** Scheduler automatically reduces LR when validation PPL plateaus.

---

### Experiment 12: 6-Layer Decoder
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 512
dropout: 0.1
learning_rate: 3e-4
batch_size: 16
max_len: 384
epochs: 10
training_samples: 25,200

text
**Result:** ⚠️ Memory intensive, marginal improvement  
**Test Perplexity:** 42.15  
**Notes:** 50% more parameters, longer training, no significant improvement.

---

## Retrieval Hyperparameter Tuning

### Retrieval Depth (k) Experiments

| k | Retrieval Time | Context Diversity | Generation Quality Impact |
|---|---------------|-------------------|--------------------------|
| 1 | Fast | Low - single reference | Limited context for generation |
| 3 | Moderate | Moderate - good balance | **Best overall - selected** |
| 5 | Slower | High - diverse contexts | Longer prompts, occasional topic drift |
| 10 | Slow | Very high | Decoder input too long, diluted relevance |

**Selected k=3:** Provides sufficient context diversity while fitting within sequence length constraints.

---

## Summary Table: Best Configurations

| Parameter | Encoder Value | Decoder Value | Rationale |
|-----------|--------------|---------------|-----------|
| **d_model** | 128 | 256 | Encoder: classification needs less capacity; Decoder: generation needs more |
| **n_heads** | 4 | 8 | Standard d_model/n_heads ratio of 32 |
| **n_layers** | 2 | 4 | Encoder: 2 layers sufficient; Decoder: 4 layers for language modeling |
| **d_ff** | 256 | 512 | Standard 4× d_model expansion |
| **dropout** | 0.1 | 0.1 | Optimal regularization without underfitting |
| **learning_rate** | 1e-4 | 3e-4 | Decoder benefits from faster initial learning |
| **batch_size** | 64 | 32 | Decoder needs smaller batches for stability |
| **max_seq_len** | 256 | 384 | Decoder needs longer sequences for prompt + generation |
| **epochs** | 10 | 10 | Validation metrics stabilize by epoch 4-5 |
| **vocab_size** | 10,000 | 10,000 | Shared vocabulary |

---
