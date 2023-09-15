# Manifold-based Dimensionality Reduction Methods

This repository contains Python implementations of two manifold-based dimensionality reduction methods based on the following research papers:

1. [Sparsity Preserving Projections with Applications to Face Recognition]([link_to_paper1](https://www.sciencedirect.com/science/article/abs/pii/S0031320309001964))
2. [Discriminant Sparse Neighborhood Preserving Embedding for Face Recognition]([link_to_paper2](https://www.sciencedirect.com/science/article/abs/pii/S0031320312000672?via%3Dihub))

These implementations were reconstructed as part of my thesis work.

## Methods Implemented

1. **Sparsity Preserving Projections (SPP):**
   - Description: SPP is a dimensionality reduction method that preserves the sparsity structure of data, which is particularly useful for face recognition applications.
   - Implementation: [SPP_method.py](SPP_method.py)

2. **Discriminant Sparse Neighborhood Preserving Embedding (DSNPE):**
   - Description: DSNPE is a technique designed for face recognition that combines discriminant analysis with sparsity preserving projections.
   - Implementation: [DSNPE_method.py](DSNPE_method.py)

## Usage

To use these dimensionality reduction methods in your Python projects, you can simply import the respective modules:

```python
from SPP_method import SPP
from DSNPE_method import DSNPE
