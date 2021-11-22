# DNMF

Unofficial Python implementation of the Discrete Non-negative Matrix Factorization (DNMF) overlapping community
detection algorithm

------------

## Paper

Ye, Fanghua, Chuan Chen, Zibin Zheng, Rong-Hua Li, and Jeffrey Xu Yu. 2019. “Discrete Overlapping Community Detection
with Pseudo Supervision.” In 2019 IEEE International Conference on Data Mining (ICDM),
708–17. https://doi.org/10.1109/ICDM.2019.00081.

-----------

## Requirements

- `python>=3.7.1`
- `torch>=1.9.1`

-----------

## Quick start

- To install the package run one of the two commands:
  - `python -m pip install dnmf-python` (installation from PyPI)
  - `python setup.py install` (compile from source, if cloned the repository)
  

- To run the algorithm, load the graph adjacency matrix into a `torch.FloatTensor` (for ex. `A`), then call:
    ```
    from dnmf-python import DNMF
    dnmf = DNMF()
    F = dnmf(A)
    ```
- To run a quick test of the algorithm with an example graph, run `python src/test.py`

-----------

## Config

The DNMF module supports the following hyperparameters as arguments:

- `alpha`: tradeoff parameter for the U-subproblem
- `beta`: tradeoff parameter for the F-subproblem
- `gamma`: regularization parameter
- `k`: desired number of overlapping communities
- `num_outer_iter`: number of iterations for the outer loop (SDP iterations)
- `num_inner_iter`: number of iterations for the inner loops (U and F subproblems)

-----------

## Author

Andrej Janchevski

andrej.janchevski@epfl.ch

EPFL STI IEM LIONS
