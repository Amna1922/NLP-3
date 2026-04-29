# NLP Assignment 3: Transformer-Based Review Understanding with RAG-Enhanced Explanation Generation



---

## Project Overview

This project implements a complete three-stage NLP pipeline for Amazon review understanding:

1. **Part A - Encoder Model**: Multi-task Transformer encoder that jointly performs sentiment classification (3-class) and helpfulness prediction (binary)
2. **Part B - Retrieval Module**: Dense retrieval system using cosine similarity over encoded review embeddings
3. **Part C - Decoder Model**: Autoregressive Transformer decoder that generates explanations for sentiment predictions, enhanced with retrieved context (RAG)

All Transformer components (attention, encoder, decoder) are implemented **entirely from scratch** without using any pre-built Transformer libraries.

---

## Dataset

- **Source**: Amazon Reviews Dataset
- **Categories Used**: Cell Phones & Accessories, Beauty, Sports & Outdoors
- **Dataset Size**: 36,000 reviews (12,000 per category)
- **Split**: 70% Train (25,200), 15% Validation (5,400), 15% Test (5,400)
- **Features**: Review text, star rating (1-5), helpfulness votes

### Data Files
The following `.json.gz` files are required in the `data/` directory:
- `cellphones.json.gz` (Cell Phones & Accessories reviews)
- `beauty.json.gz` (Beauty reviews)
- `sports.json.gz` (Sports & Outdoors reviews)

---

## Results Summary

### Part A: Encoder Performance
| Metric | Test Accuracy |
|--------|--------------|
| Sentiment Classification (3-class) | **83.70%** |
| Helpfulness Prediction (Binary) | **71.31%** |

### Part B: Retrieval Performance
- **Method**: Cosine similarity over mean-pooled encoder embeddings
- **Index Size**: 25,200 training embeddings
- **Retrieval Depth (k)**: 3
- **Similarity Metric**: Cosine Similarity

### Part C: Decoder Performance
| Metric | Value |
|--------|-------|
| Test Perplexity | **40.34** |
| Architecture | 4 layers, 8 heads, 256 dimensions |
| Generation Method | Top-p (nucleus) sampling |

### RAG Ablation Study
The model was evaluated with and without retrieved context. Results show that RAG-enhanced generation produces more contextually relevant explanations compared to the baseline without retrieval.

---

## Implementation Details

### Preprocessing
- **Text Cleaning**: Lowercasing, punctuation removal
- **Tokenization**: Custom word-level tokenizer (from scratch)
- **Vocabulary**: 10,000 tokens, built from training data only
- **Sequence Length**: Maximum 256 tokens (padded/truncated)
- **Special Tokens**: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<REVIEW>`, `<SENT>`, `<HELP>`, `<RETR>`, `<SEP>`, `<EXP>`

### Part A: Encoder Architecture
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Encoder Layers**: 2
- **Feed-Forward Dimension**: 256
- **Dropout**: 0.1
- **Activation**: GELU
- **Pooling**: Mean pooling (non-padded tokens)
- **Loss**: CrossEntropy (sentiment) + BCEWithLogits (helpfulness)
- **Optimizer**: Adam (lr=1e-4)
- **Training Epochs**: 10

### Part B: Retrieval Module
- **Query Construction**: Mean-pooled encoder output
- **Similarity Search**: Brute-force cosine similarity
- **Self-Match Prevention**: Filtered identical reviews
- **Retrieval Depth (k)**: Configurable (default=3)

### Part C: Decoder Architecture
- **Embedding Dimension**: 256
- **Attention Heads**: 8
- **Decoder Layers**: 4
- **Feed-Forward Dimension**: 512
- **Dropout**: 0.1
- **Causal Masking**: Future token masking enforced
- **Generation**: Autoregressive with top-p (nucleus) sampling
- **Training Objective**: CrossEntropy loss (only on explanation tokens)
- **Optimizer**: AdamW (lr=3e-4) with ReduceLROnPlateau scheduler
- **Training Epochs**: 10

### Input Template (All 4 Required Elements)
<REVIEW> [review_text] <SENT> [sentiment_label] <HELP> [helpfulness_label] <RETR> [retrieved_context] <SEP> <EXP> [generated_explanation] <EOS>


---

## How to Run

###  Google Colab (Recommended)
1. Upload the `.json.gz` files to Colab
2. Open the notebook in Colab
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells sequentially (Runtime → Run all)




