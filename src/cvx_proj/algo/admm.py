import torch
from cvx_proj.algo.abstract import AbstractDecomposition
from cvx_proj.algo.init_cond import init_QLR_data_agnostic, init_QLR_data_aware
from cvx_proj.algo.lplr import approx_lplr_proj
from tqdm import tqdm
from copy import deepcopy


class ADMMQLR(AbstractDecomposition):
    def update_Q_data_agnostic(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        Qq: torch.Tensor,
        dual_Q: torch.Tensor,
        rho: float
    ):
        return 1/(1 + rho) * (W - L @ R + rho*(Qq - dual_Q))
    
    def update_Q_data_aware(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        Qq: torch.Tensor,
        dual_Q: torch.Tensor,
        rho: float,
        H: torch.Tensor
    ):
        I = torch.eye(H.shape[0], device=H.device)
        return ((W - L @ R) @ H + rho * (Qq - dual_Q)) @ \
                torch.linalg.inv(H + rho * I)
    
    def update_L_data_agnostic(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        Lq: torch.Tensor,
        dual_L: torch.Tensor,
        rho: float
    ):
        I = torch.eye(R.shape[0], device=R.device)
        return ((W - Q) @ R.T + rho * (Lq - dual_L)) @ \
                torch.linalg.inv(R @ R.T + rho * I)
    
    def update_L_data_aware(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        Lq: torch.Tensor,
        dual_L: torch.Tensor,
        rho: float,
        H: torch.Tensor
    ):
        I = torch.eye(R.shape[0], device=R.device)
        return ((W - Q) @ H @ R.T + rho * (Lq - dual_L)) @ \
                torch.linalg.inv(R @ H @ R.T + rho * I)
    
    def update_R_data_agnostic(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        L: torch.Tensor,
        Rq: torch.Tensor,
        dual_R: torch.Tensor,
        rho: float
    ):
        I = torch.eye(L.shape[1], device=L.device)
        return torch.linalg.inv(L.T @ L + rho * I) @ \
                (L.T @ (W - Q) + rho * (Rq - dual_R))
    
    def update_R_data_aware(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        L: torch.Tensor,
        Rq: torch.Tensor,
        dual_R: torch.Tensor,
        rho: float,
        H: torch.Tensor,
        eigH
    ):
        
        L1, U = torch.linalg.eigh(L.T @ L)
        L2, V = eigH
        C = U.T @ (L.T @ (W - Q) @ H + rho * (Rq - dual_R)) @ V / rho

        Y = C / (L1[:, None] * L2[None, :] / rho + 1)
        return U @ Y @ V.T
    
    def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int,
                  rho: float = 1, **kwargs):
        

        self.errors = []
        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            eigH = None
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))

        Qq = deepcopy(Q)
        Lq = deepcopy(L)
        Rq = deepcopy(R)

        dual_Q = torch.zeros_like(Q)
        dual_L = torch.zeros_like(L)
        dual_R = torch.zeros_like(R)

        best_error = float('inf')
        best_W_hat = None

        for t in tqdm(range(n_iter)):
        
            Qq = self.quant_Q.simulated_quant(Q + dual_Q)
            Lq = self.quant_LR.simulated_quant(L + dual_L)
            Rq = self.quant_LR.simulated_quant(R + dual_R)

            if self.data_aware:
                L = self.update_L_data_aware(W, Q, R, Lq, dual_L, rho, H)
                R = self.update_R_data_aware(W, Q, L, Rq, dual_R, rho, H, eigH)
                Q = self.update_Q_data_aware(W, L, R, Qq, dual_Q, rho, H)
            else:
                L = self.update_L_data_agnostic(W, Q, R, Lq, dual_L, rho)
                R = self.update_R_data_agnostic(W, Q, L, Rq, dual_R, rho)
                Q = self.update_Q_data_agnostic(W, L, R, Qq, dual_Q, rho)

            dual_Q += Q - Qq
            dual_L += L - Lq
            dual_R += R - Rq

            self.errors.append(self.get_error(W, Qq + Lq @ Rq, H))

            if self.errors[-1] < best_error:
                best_error = self.errors[-1]
                best_W_hat = Q + L @ R

        return best_W_hat

    
class ADMM(AbstractDecomposition):
    def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int,
                  rho: float = 1, n_iter_inner=3, **kwargs):
        self.errors = []
        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            eigH = None
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))

        X = L @ R
        Qq = deepcopy(Q)
        Xq = deepcopy(X)

        dual_Q = torch.zeros_like(Q)
        dual_X = torch.zeros_like(Q)
    
        best_error = float('inf')
        best_W_hat = None

        for t in tqdm(range(n_iter)):
            Qq = self.quant_Q.simulated_quant(Q + dual_Q)
            Xq = X + dual_X

            L, R = approx_lplr_proj(Xq, H, eigH, self.rank, self.quant_LR, 
                                    data_aware=self.data_aware, n_iter=n_iter_inner)
            Xq = L @ R

            if self.data_aware:
                I = torch.eye(H.shape[0], device=H.device)
                X = (W @ H - (Qq - dual_Q) @ H + (Xq - dual_X) @ (H + rho*I)) @ torch.linalg.inv(2*H + rho*I)
                Q = X + (Qq - dual_Q) - (Xq - dual_X)
            else:
                X = (W  + (Xq - dual_X) * (1 + rho) - (Qq - dual_Q)) / (2 + rho)
                Q = X + (Qq - dual_Q) - (Xq - dual_X)

            dual_Q += Q - Qq
            dual_X += X - Xq

            self.errors.append(self.get_error(W, Qq + Xq, H))

            if self.errors[-1] < best_error:
                best_error = self.errors[-1]
                best_W_hat = Q + L @ R

        return best_W_hat
