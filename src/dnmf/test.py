""" Test script for the algorithm, can verify output with official Matlab version """

import pandas as pd
import torch

from DNMF import DNMF

if __name__ == "__main__":
    A = torch.from_numpy(pd.read_csv("example_graph/dolphins-edgesMatrix.csv", header=0, index_col=0).values).float()
    print(A)

    dnmf = DNMF()
    F = dnmf(A, verbose=True)
    print(F)
