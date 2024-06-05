import torch
from cvx_proj.algo.abstract import AbstractDecomposition
from cvx_proj.algo.init_cond import init_QLR_data_agnostic, init_QLR_data_aware
from cvx_proj.algo.lplr import approx_lplr_proj
from tqdm import tqdm


class AlternatingMinimization(AbstractDecomposition):
    def update_Q(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor
    ):
        return self.quant_Q.simulated_quant(W - L @ R)

    def update_L_data_agnostic(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor
    ):
        resid = W - Q

        L = torch.linalg.lstsq(R.T, resid.T)[0].T
        if torch.isnan(L).any():
            L = resid @ torch.linalg.pinv(R)
        return self.quant_LR.simulated_quant(L.T).T

    def update_L_data_aware(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        H_sqrt: torch.Tensor
    ):
        resid = W - Q
        L = torch.linalg.lstsq((R @ H_sqrt).T, (resid @ H_sqrt).T)[0].T
        if torch.isnan(L).any():
            L = (resid @ H_sqrt) @ torch.linalg.pinv(R @ H_sqrt)
        return self.quant_LR.simulated_quant(L.T).T

    def update_R(
        self,
        W: torch.Tensor,
        Q: torch.Tensor,
        L: torch.Tensor
    ):
        resid = W - Q
        R = torch.linalg.lstsq(L, resid)[0]
        if torch.isnan(R).any():
            R = torch.linalg.pinv(L) @ resid
        return self.quant_LR.simulated_quant(R)

    def decompose(self, W: torch.Tensor, H: torch.Tensor, n_iter: int,
                  n_iter_inner:int = 1, re_init_lplr=False, **kwargs):
        self.errors = []

        if self.data_aware:
            eigH = torch.linalg.eigh(H)
            H_sqrt = (eigH.eigenvectors @
                    torch.diag(torch.sqrt(eigH.eigenvalues)) @
                    eigH.eigenvectors.T)
            Q, L, R = init_QLR_data_aware(W, eigH, self.rank, self.quant_Q, self.quant_LR)
        else:
            eigH = None
            Q, L, R = init_QLR_data_agnostic(W, self.rank, self.quant_Q, self.quant_LR)
        self.errors.append(self.get_error(W, Q + L @ R, H))

        best_error = self.errors[-1]
        best_W_hat = Q + L @ R

        for _ in tqdm(range(n_iter)):
            if re_init_lplr:
                L, R = approx_lplr_proj(W - Q, H, eigH, self.rank, self.quant_LR,
                                        self.data_aware, n_iter_inner, L, R)
            else:
                L, R = approx_lplr_proj(W - Q, H, eigH, self.rank, self.quant_LR,
                                        self.data_aware, n_iter_inner)

            Q = self.update_Q(W, L, R)
            self.errors.append(self.get_error(W, Q + L @ R, H))

            if self.errors[-1] < best_error:
                best_error = self.errors[-1]
                best_W_hat = Q + L @ R

        return best_W_hat

