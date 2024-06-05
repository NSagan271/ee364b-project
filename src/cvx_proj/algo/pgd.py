import torch
from cvx_proj.algo.abstract import AbstractDecomposition
from cvx_proj.algo.init_cond import init_QLR_data_agnostic, init_QLR_data_aware
from cvx_proj.algo.lplr import approx_lplr_proj
from tqdm import tqdm
from copy import deepcopy


class PGD(AbstractDecomposition):
     
     def ls_cond(self, f_diff, grad, var_diff, eta):
         return f_diff <= sum(torch.trace(grad[i].T @ var_diff[i]) for i in range(2)) + 1/(2*eta) * \
                sum(torch.linalg.norm(var_diff[i])**2 for i in range(2))

     def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int,
                  eta: float = 1, gamma=0.8, n_iter_inner=3, **kwargs):
        self.errors = []
        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            eigH = None
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))

        for t in tqdm(range(n_iter)):
            if self.data_aware:
                grad_Q = -(W - (Q + L @ R)) @ H
                grad_X = -(W - (Q + L @ R)) @ H
            else:
                grad_Q = -(W - (Q + L @ R))
                grad_X = -(W - (Q + L @ R))

            f_old = self.get_error(W, Q + L @ R, H)
            old_Q, old_L, old_R = deepcopy(Q), deepcopy(L), deepcopy(R)

            Q = self.quant_Q.simulated_quant(Q - eta * grad_Q)
            L, R =  approx_lplr_proj(L @ R - eta * grad_X, H, eigH, self.rank, self.quant_LR, 
                                    data_aware=self.data_aware, n_iter=n_iter_inner)
            f_new = self.get_error(W, Q + L @ R, H)

            while not self.ls_cond(f_new - f_old, (grad_Q, grad_X), 
                                   (Q - old_Q, L@R - old_L@old_R), eta):
                eta *= gamma

            self.errors.append(self.get_error(W, Q + L @ R, H))

class PGDRelaxed(AbstractDecomposition):
     
     def ls_cond(self, f_diff, grad, var_diff, eta):
        return f_diff <= sum(torch.trace(grad[i].T @ var_diff[i]) for i in range(3)) + 1/(2*eta) * \
                sum(torch.linalg.norm(var_diff[i])**2 for i in range(3))
     
     def soft_thresh(self, x, alpha):
        return torch.maximum(x - alpha, torch.zeros_like(x)) - \
            torch.maximum(-x - alpha, torch.zeros_like(x))

     def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int,
                  eta: float = 1, gamma=0.8, lmbda=1, **kwargs):
        self.errors = []
        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))


        best_error = float('inf')
        best_W_hat = None
        for t in tqdm(range(n_iter)):
            if self.data_aware:
                grad_Q = -(W - (Q + L @ R)) @ H
                grad_L = -(W - (Q + L @ R)) @ H @ R.T
                grad_R = -L.T @ (W - (Q + L @ R))  @ H
            else:
                grad_Q = -(W - (Q + L @ R))
                grad_L = -(W - (Q + L @ R)) @ R.T
                grad_R = -L.T @ (W - (Q + L @ R))

            f_old = self.get_error(W, Q + L @ R, H)
            old_Q, old_L, old_R = deepcopy(Q), deepcopy(L), deepcopy(R)

            Q = Q - eta * grad_Q
            L = L - eta * grad_L
            R = R - eta * grad_R
            
            Q_quant = self.quant_Q.simulated_quant(Q)
            L_quant = self.quant_LR.simulated_quant(L)
            R_quant = self.quant_LR.simulated_quant(R)

            Q = self.soft_thresh(Q - Q_quant, lmbda*eta) + Q_quant
            L = self.soft_thresh(L - L_quant, lmbda*eta) + L_quant
            R = self.soft_thresh(R - R_quant, lmbda*eta) + R_quant
            f_new = self.get_error(W, Q + L @ R, H)

            # while not self.ls_cond(f_new - f_old, (grad_Q, grad_L, grad_R), 
            #                        (Q - old_Q, L - old_L, R - old_R), eta):
            #     eta *= gamma

            Q_quant = self.quant_Q.simulated_quant(Q)
            L_quant = self.quant_LR.simulated_quant(L)
            R_quant = self.quant_LR.simulated_quant(R)
            self.errors.append(self.get_error(W, Q_quant + L_quant @ R_quant, H))

            if self.errors[-1] < best_error:
                best_error = self.errors[-1]
                best_W_hat = Q + L @ R

        return best_W_hat