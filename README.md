<p align="center">
  <img src="https://github.com/tonypapakon/DRmethods/blob/main/extras/wine.png" align="center" width=50% height=50% />
</p>

# Manifold-based Dimensionality Reduction Methods

This repository contains Python implementations of two manifold-based dimensionality reduction methods based on the following research papers:

1. [Sparsity Preserving Projections with Applications to Face Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320309001964)
2. [Discriminant Sparse Neighborhood Preserving Embedding for Face Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320312000672?via%3Dihub)

These implementations were reconstructed as part of my thesis work.

### Research Paper
- Description: My thesis paper explores the implementation and applications of sparsity-preserving projections and discriminant sparse neighborhood preserving embedding for face recognition. It provides in-depth insights into the methods and their performance.
  - [Link to My Thesis Paper (Greek)](https://github.com/tonypapakon/DRmethods/blob/main/Thesis.pdf)
  - Published in: [IKEE Auth (GRI-2023-41084)](http://ikee.lib.auth.gr/record/351350/?ln=el)

### Presentation
- Overview: This presentation offers a visual walkthrough of the key concepts, methodologies, and findings presented in my thesis. It serves as a supplementary resource to better understand the implemented dimensionality reduction methods.
  - [Link to My Thesis Presentation (Greek)](https://github.com/tonypapakon/DRmethods/blob/main/Presentation.pdf)

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

#SPP use
spp = SPP(n_components=50, epsilon=0.05)
spp.fit(X)
X_spp = dsnpe.transform(X)

#DSNPE use
dsnpe = DSNPE(n_components=50, epsilon=0.05, gamma=1.0)
dsnpe.fit(X, y)
X_dsnpe = dsnpe.transform(X)
