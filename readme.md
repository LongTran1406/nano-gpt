# NanoGPT: A Lightweight GPT from Scratch

## Overview

**NanoGPT** is a compact implementation of the Transformer-based GPT model built from scratch using PyTorch.  
This project demonstrates how to train a character-level or token-level language model, generate text, and deploy a simple text-generation system.  

The model is capable of **streaming text generation**, printing each token as it is generated — similar to how ChatGPT outputs text in real time.  

---

## Features

- Lightweight GPT architecture implemented from scratch.
- Multi-head self-attention with positional embeddings.
- Supports multiple Transformer blocks.
- Character/token-level text generation.
- Streaming output for real-time text display.
- Trained with PyTorch and fully compatible with GPU acceleration.

---

## Model Architecture

The NanoGPT model is composed of several key components, implemented from scratch in PyTorch:

- **Head (Self-Attention Head):**  
  Computes query, key, and value projections from the input embeddings. Applies scaled dot-product attention with a causal mask to prevent attending to future tokens.

- **MultiHead (Multi-Head Attention):**  
  Concatenates multiple `Head` outputs and projects back to the embedding dimension. Includes dropout for regularization.

- **FeedForward:**  
  A fully connected layer with ReLU activation followed by dropout, applied to each token independently.

- **Block (Transformer Block):**  
  Combines `MultiHead` attention and `FeedForward` layers with residual connections and layer normalization:
  - `x = x + MultiHead(LayerNorm(x))`
  - `x = x + FeedForward(LayerNorm(x))`

- **NanoGPT (Full Model):**  
  - **Embedding Layer:** Maps token indices to dense embeddings.  
  - **Positional Embedding:** Adds positional information for each token in the sequence.  
  - **Transformer Blocks:** Stacks multiple `Block`s sequentially.  
  - **Output Layer:** Linear layer projecting embeddings to vocabulary logits.  

- **Text Generation Methods:**  
  - `generate()`: Generates a fixed number of new tokens, returning the full sequence.  
  - `generate_stream()`: Generates tokens one by one, printing each token immediately for real-time streaming output.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/LongTran1406/nano-gpt.git
cd NanoGPT
```

2. Install dependencies:

```bash
pip install requirements.txt
```


## Usage
1. Load Pretrained Model for inference

```bash
python gpt_inference.py
```

2. Training Model using custom dataset
```bash
python gpt_training.py
```


### Acknowledgements

Attention Mechanism & Transformer: Vaswani et al., “Attention is All You Need” (Paper)

NanoGPT Implementation Inspiration: Andrej Karpathy, NanoGPT GitHub

PyTorch: Deep learning framework used for implementation.

HuggingFace Transformers: Reference for model design and tokenization.
