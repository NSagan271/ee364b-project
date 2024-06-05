# Evaluation of Convex  Optimization Algorithms for Quantized Plus Low-Rank Matrix Decompositions

**Abstract**: We evaluate several different convex optimization algorithms for the non-convex task of decomposing a matrix into the sum of a quantized matrix and a low-rank matrix, where the low-rank matrix is the product of quantized low-rank factors. This sort of decomposition is versitile and can achieve a high compression ratio. We formulate this factorization as a non-convex optimization problem in several ways, and find approximate solutions using projected alternating minimization, proximal gradient descent, ADMM, and Douglas-Rachford splitting.

## Setup
Run `pip install --editable .` in the current directory to install the `src` directory as the editable Python package `cvx_proj`.