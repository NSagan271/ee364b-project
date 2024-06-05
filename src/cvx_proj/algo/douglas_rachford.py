import torch
from cvx_proj.algo.abstract import AbstractDecomposition
from cvx_proj.algo.init_cond import init_QLR_data_agnostic, init_QLR_data_aware
from cvx_proj.algo.lplr import approx_lplr_proj
from tqdm import tqdm
from copy import deepcopy


class DouglasRachford(AbstractDecomposition):
    
    def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int, 
                  n_iter_inner=1, **kwargs):
        self.errors = []
        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            eigH = None
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))

        ZQ = deepcopy(Q)
        ZX = deepcopy(L @ R)

        best_W_hat = None
        best_error = float('inf')

        for t in tqdm(range(n_iter)):
            if self.data_aware:
                I = torch.eye(H.shape[0], device=H.device)
                X_half = (W @ H - ZQ @ H + ZX @ (H + I)) @ torch.linalg.inv(2*H + I)
                Q_half = X_half + ZQ - ZX
            else:
                Y1 = W + ZQ
                Y2 = W + ZX
                Q_half = (2*Y1 - Y2) / 3
                X_half = (2*Y2 - Y1) / 3
            ZQ_half = 2*Q_half - ZQ
            ZX_half = 2*X_half - ZX
            Q = self.quant_Q.simulated_quant(ZQ_half)

            L, R = approx_lplr_proj(ZX_half, H, eigH, self.rank, self.quant_LR, 
                                    data_aware=self.data_aware, n_iter=n_iter_inner)
            X = L @ R
            
            ZQ = ZQ + Q - Q_half
            ZX = ZX + X - X_half

            self.errors.append(self.get_error(W, Q + L @ R, H))

            if self.errors[-1] < best_error:
                best_error = self.errors[-1]
                best_W_hat = Q + L @ R
        return best_W_hat