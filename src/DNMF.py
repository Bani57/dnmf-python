""" Script containing the algorithm implementation """

import torch
from torch import nn


class DNMF(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1,
                 k: int = 5, num_outer_iter: int = 50, num_inner_iter: int = 500):
        """
        Implements https://doi.org/10.1109/ICDM.2019.00081

        :param alpha: U-subproblem weight, float
        :param beta: F-subproblem weight, float
        :param gamma: regularization parameter, float
        :param k: desired number of communities, int
        :param num_outer_iter: number of outer iterations, int
        :param num_inner_iter: number of inner iterations, int
        """

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.num_outer_iter = num_outer_iter
        self.num_inner_iter = num_inner_iter
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __ProjTF(self, A: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix onto the set of tight frames

        :param A: matrix, torch.Tensor

        :return: tight frame matrix projection of A, torch.Tensor
        """

        U, S, V = torch.linalg.svd(A)
        return U @ torch.eye(len(S), device=self.device) @ V.T

    def __loss_u(self, A: torch.Tensor, F: torch.Tensor, U: torch.Tensor, Q: torch.Tensor) -> float:
        """
        Calculate the loss for the U-subproblem: ||A - UU^T||^2_F + alpha ||U - FQ||^2_F

        :param A: adjacency matrix of a graph, torch.Tensor
        :param F: discrete community membership matrix, torch.Tensor
        :param U: continuous community membership matrix, torch.Tensor
        :param Q: rotation matrix, torch.Tensor

        :return: loss value, float
        """

        return torch.linalg.norm(A - U @ U.T) ** 2 + self.alpha * torch.linalg.norm(U - F @ Q) ** 2

    def __update_u(self, A: torch.Tensor, F: torch.Tensor, U: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Solve the U optimization problem

        :param A: adjacency matrix of a graph, torch.Tensor
        :param F: discrete community membership matrix, torch.Tensor
        :param U: continuous community membership matrix, torch.Tensor
        :param Q: rotation matrix, torch.Tensor

        :return: U, updated continuous community membership matrix, torch.Tensor
        """

        iter = 0
        converged = False
        epsilon_conv = 1e-6
        epsilon_denom = torch.tensor(1e-15, device=self.device)

        Q_plus = (torch.abs(Q) + Q) / 2
        Q_minus = (torch.abs(Q) - Q) / 2
        prev_loss_u = self.__loss_u(A, F, U, Q)

        while iter < 10 * self.num_inner_iter and not converged:
            numerator = 2 * A @ U + self.alpha * F @ Q_plus
            denominator = torch.maximum(2 * U @ U.T @ U + self.alpha * U + self.alpha * F @ Q_minus, epsilon_denom)
            U = U * (numerator / denominator) ** 0.25

            iter += 1
            cur_loss_u = self.__loss_u(A, F, U, Q)
            if abs(cur_loss_u - prev_loss_u) < epsilon_conv:
                converged = True
            prev_loss_u = cur_loss_u

        return U

    def __loss_f(self, F: torch.Tensor, U: torch.Tensor, S: torch.Tensor, Q: torch.Tensor) -> float:
        """
        Calculate the loss for the F-subproblem: alpha ||U - FQ||^2_F + beta Tr(F^TSF)

        :param F: discrete community membership matrix, torch.Tensor
        :param U: continuous community membership matrix, torch.Tensor
        :param S: discrimination matrix, torch.Tensor
        :param Q: rotation matrix, torch.Tensor

        :return: loss value, float
        """

        return self.alpha * torch.linalg.norm(U - F @ Q) ** 2 + self.beta * torch.trace(F.T @ S @ F)

    def __update_f(self, F: torch.Tensor, U: torch.Tensor, S: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Solve the F optimization problem

        :param F: discrete community membership matrix, torch.Tensor
        :param U: continuous community membership matrix, torch.Tensor
        :param S: discrimination matrix, torch.Tensor
        :param Q: rotation matrix, torch.Tensor

        :return: updated discrete community membership matrix, torch.Tensor
        """

        iter = 0
        converged = False
        epsilon = 1e-6

        N, K = F.size()
        I = torch.eye(N, device=self.device)
        row_k_ones = torch.ones(1, K, device=self.device)
        UQ = U @ Q.T
        S_prime = self.beta * S + self.alpha * I
        S_exclusion_masks = (1 - I).bool()
        prev_loss_f = self.__loss_f(F, U, S, Q)

        while iter < self.num_inner_iter and not converged:
            for i in range(N):
                e = S_prime[i, i] * row_k_ones \
                    + 2 * (S_prime[i, S_exclusion_masks[i]] @ F[S_exclusion_masks[i]] - self.alpha * UQ[i])
                F[i] = ((e == torch.min(e)) | (e < 0)).long()

            iter += 1
            cur_loss_f = self.__loss_f(F, U, S, Q)
            if abs(cur_loss_f - prev_loss_f) < epsilon:
                converged = True
            prev_loss_f = cur_loss_f

        return F

    def __update_q(self, U: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Solve the Q optimization problem: alpha ||U - FQ||^2_F, s.t. QQ^T = I

        :param U: continuous community membership matrix, torch.Tensor
        :param F: discrete community membership matrix, torch.Tensor

        :return: updated rotation matrix, torch.Tensor
        """

        left_sv, _, right_sv = torch.linalg.svd(U.T @ F)
        return right_sv @ left_sv.T

    def forward(self, A: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Run the DNMF overlapping community detection algorithm on the given graph adjacency matrix

        :param A: adjacency matrix of a graph, torch.Tensor
        :param verbose: whether to print at every iteration when U-subproblem is solved

        :return: F, discrete community membership matrix, torch.Tensor
        """

        A = A.to(self.device)
        n = A.size(0)
        I = torch.eye(n, device=self.device)

        H = I - (1 / n)
        K = torch.exp(-0.5 * torch.sum((A.unsqueeze(2) - A) ** 2, 1))
        K_hat = H.T @ K @ H
        S = H - torch.linalg.inv(K_hat + self.gamma * I) @ K_hat

        U = torch.rand(n, self.k, device=self.device)
        F = torch.rand(n, self.k, device=self.device)
        Q = self.__ProjTF(torch.rand(self.k, self.k, device=self.device))

        iter = 0
        while iter <= self.num_outer_iter:
            U = self.__update_u(A, F, U, Q)
            if verbose:
                print(f"finish U in round {iter + 1}")
            F = self.__update_f(F, U, S, Q)
            Q = self.__update_q(U, F)
            iter += 1
        return F
