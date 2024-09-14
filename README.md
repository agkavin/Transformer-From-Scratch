# Transformer From Scratch

## Overview

This repository contains a raw implementation of the Transformer model based on the groundbreaking paper **"Attention Is All You Need"** by Vaswani et al. (2017). The Transformer architecture has revolutionized natural language processing and sequence modeling by relying solely on attention mechanisms, discarding the need for recurrent layers.

## What is the Transformer?

The Transformer model introduced in the paper **"Attention Is All You Need"** leverages self-attention mechanisms to process sequences of data more efficiently than traditional models like RNNs or LSTMs. It consists of an encoder-decoder structure where both components are built from layers of multi-head self-attention and feed-forward neural networks.

Key components include:
- **Self-Attention**: Mechanism to weigh the importance of different tokens in the sequence.
- **Multi-Head Attention**: Multiple self-attention mechanisms running in parallel.
- **Positional Encoding**: Adds information about the position of tokens in the sequence.
- **Feed-Forward Neural Networks**: Applied independently to each position.

## Model Architecture

Below is a visual representation of the Transformer model architecture:

![Transformer Architecture](transformers.png)



### File Structure

- `model.py`: Contains the implementation of the Transformer model.
- `README.md`: This file.

## Getting Started

To get started with the implementation, clone the repository and install the required dependencies:

```bash
git clone https://github.com/agkavin/Transformer-From-Scratch.git
cd Transformer-From-Scratch
pip install torch
```

You can then run the training script to train the Transformer model:

```bash
python model.py
```

