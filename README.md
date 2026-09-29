# TinyWilliam

TinyWilliam is a small Transformer language-model experiment built from scratch with PyTorch.

The project started as a learning path from individual neurons and manual backpropagation to a working causal Transformer.

## Current model

- Vocabulary: 16 tokens
- Context length: 32
- Model dimension: 24
- Feed-forward dimension: 128
- Transformer blocks: 1
- Attention: single-head causal self-attention
- Positional encoding: learned positional embeddings
- Normalization: manual LayerNorm
- Activation: GELU
- Optimizer: AdamW
- Learning rate: 0.001

The model has roughly 10K trainable parameters.

## Current corpus

The current training corpus contains seven short sentences:

- the cat eats fish
- the dog eats meat
- the cat likes milk
- the dog likes bones
- the bird eats seeds
- the bird likes worms
- the fox likes bones

The sentence **the fox eats meat** is intentionally held out.

## First generalization experiment

After training, TinyWilliam was given:

`the fox eats`

without having seen the complete sentence during training.

The model predicted:

`meat` at approximately 99.98% probability.

This is an interesting compositional signal, but it is not evidence of general language understanding. The dataset is deliberately tiny and controlled. Future experiments will test more held-out combinations.

## Repository structure

- `tokenizer.py` - vocabulary and tokenization
- `dataset.py` - next-token training pairs
- `model.py` - TinyWilliam Transformer
- `train.py` - training and evaluation
- `probe.py` - hidden-representation experiments
- `attention.py`, `train_attention.py` - earlier attention experiments
- `transformer_block.py` - earlier Transformer-block experiment
- `tiny_transformer.py` - earlier full Transformer experiment
- earlier Python files - progressive neural-network and backpropagation experiments

## Philosophy

TinyWilliam is intentionally built without Hugging Face Transformers. The goal is to understand the mechanics by implementing the model directly with PyTorch tensors and operations.

The repository is experimental and educational. Results should be treated as experiments, not as claims about general intelligence.
