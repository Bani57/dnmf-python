""" Script containing the algorithm implementation """
from typing import Union

import torch
from torch import nn


class DNMF(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1,
                 k: int = 5, num_outer_iter: int = 50, num_inner_iter: int = 500,
                 device: Union[str, torch.device] = "cpu"):
        """
        Implements https://doi.org/10.1109/ICDM.2019.00081

        :param alpha: U-subproblem weight, float
        :param beta: F-subproblem weight, float
        :param gamma: regularization parameter, float
        :param k: desired number of communities, int
        :param num_outer_iter: number of outer iterations, int
        :param num_inner_iter: number of inner iterations, int
        :param device: hardware device on which to perform computations, str or torch.device
        """

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.num_outer_iter = num_outer_iter
        self.num_inner_iter = num_inner_iter
        self.device = device

        self.epsilon_conv = 1e-6
        self.epsilon_denom = 1e-15

    def __reset(self, A: torch.Tensor):
        """
        Initialize the algorithm variables

        :param A: adjacency matrix of a graph, torch.Tensor
        """

        self.__N = A.size(0)
        I = torch.eye(self.__N, device=self.device)

        H = I - (1 / self.__N)
        K = torch.exp(-0.5 * torch.sum((A.unsqueeze(2) - A) ** 2, 1))
        K_hat = H.T @ K @ H
        self.__S = H - torch.linalg.inv(K_hat + self.gamma * I) @ K_hat
        self.__S_prime = self.beta * self.__S + self.alpha * I
        self.__row_k_ones = torch.ones(1, self.k, device=self.device)
        self.__exclusion_masks = (1 - I).bool()

        self.__U = torch.rand(self.__N, self.k, device=self.device)
        self.__F = torch.rand(self.__N, self.k, device=self.device)
        self.__Q = self.__ProjTF(torch.rand(self.k, self.k, device=self.device))

    def __ProjTF(self, A: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix onto the set of tight frames

        :param A: matrix, torch.Tensor

        :return: tight frame matrix projection of A, torch.Tensor
        """

        U, S, V = torch.linalg.svd(A)
        return U @ torch.eye(len(S), device=self.device) @ V.T

    def __loss_u(self, A: torch.Tensor) -> float:
        """
        Calculate the loss for the U-subproblem: ||A - UU^T||^2_F + alpha ||U - FQ||^2_F

        :param A: adjacency matrix of a graph, torch.Tensor

        :return: loss value, float
        """

        return torch.linalg.norm(A - self.__U @ self.__U.T) ** 2 \
               + self.alpha * torch.linalg.norm(self.__U - self.__F @ self.__Q) ** 2

    def __update_u(self, A: torch.Tensor):
        """
        Solve the U optimization problem

        :param A: adjacency matrix of a graph, torch.Tensor
        """

        step = 0
        converged = False

        Q_plus = (torch.abs(self.__Q) + self.__Q) / 2
        Q_minus = (torch.abs(self.__Q) - self.__Q) / 2
        prev_loss_u = self.__loss_u(A)

        while step < 10 * self.num_inner_iter and not converged:
            numerator = 2 * A @ self.__U + self.alpha * self.__F @ Q_plus
            denominator = torch.clamp(2 * self.__U @ self.__U.T @ self.__U
                                      + self.alpha * self.__U + self.alpha * self.__F @ Q_minus,
                                      min=self.epsilon_denom, max=None)
            self.__U = self.__U * (numerator / denominator) ** 0.25

            step += 1
            cur_loss_u = self.__loss_u(A)
            if abs(cur_loss_u - prev_loss_u) < self.epsilon_conv:
                converged = True
            prev_loss_u = cur_loss_u

    def __loss_f(self) -> float:
        """
        Calculate the loss for the F-subproblem: alpha ||U - FQ||^2_F + beta Tr(F^TSF)

        :return: loss value, float
        """

        return self.alpha * torch.linalg.norm(self.__U - self.__F @ self.__Q) ** 2 \
               + self.beta * torch.trace(self.__F.T @ self.__S @ self.__F)

    def __update_f(self):
        """
        Solve the F optimization problem
        """

        step = 0
        converged = False

        UQ = self.__U @ self.__Q.T

        prev_loss_f = self.__loss_f()

        while step < self.num_inner_iter and not converged:
            for i in range(self.__N):
                e = self.__S_prime[i, i] * self.__row_k_ones \
                    + 2 * (self.__S_prime[i, self.__exclusion_masks[i]] @ self.__F[self.__exclusion_masks[i]]
                           - self.alpha * UQ[i])
                self.__F[i] = ((e == torch.min(e)) | (e < 0)).long()

            step += 1
            cur_loss_f = self.__loss_f()
            if abs(cur_loss_f - prev_loss_f) < self.epsilon_conv:
                converged = True
            prev_loss_f = cur_loss_f

    def __update_q(self):
        """
        Solve the Q optimization problem: alpha ||U - FQ||^2_F, s.t. QQ^T = I
        """

        left_sv, _, right_sv = torch.linalg.svd(self.__U.T @ self.__F)
        self.__Q = right_sv @ left_sv.T

    def forward(self, A: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Run the DNMF overlapping community detection algorithm on the given graph adjacency matrix

        :param A: adjacency matrix of a graph, torch.Tensor
        :param verbose: whether to print at every iteration when U-subproblem is solved

        :return: F, discrete community membership matrix, torch.Tensor
        """

        A = A.to(self.device)
        self.__reset(A)

        step = 0
        while step <= self.num_outer_iter:
            self.__update_u(A)
            if verbose:
                print(f"finish U in round {step + 1}")
            self.__update_f()
            self.__update_q()
            step += 1
        return self.__F
