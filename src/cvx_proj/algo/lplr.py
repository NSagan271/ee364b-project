import torch
from cvx_proj.quantization.abstract import AbstractQuantizer


def approx_lplr_proj(X: torch.Tensor, H: torch.Tensor,
                     eigH, rank: int, quantizer: AbstractQuantizer,
                     data_aware: bool=False, n_iter: int=3,
                     init_L = None, init_R = None):
    if data_aware:
        H_sqrt = eigH.eigenvectors @ torch.diag(eigH.eigenvalues.sqrt()) @ eigH.eigenvectors.T
        if (init_L is not None) and (init_R is not None):
            L, R = init_L, init_R
        else:
            Y = X @ H_sqrt @ eigH.eigenvectors
            U, S, _ = torch.svd_lowrank(Y, q=rank*2)
            L = U[:, :rank]
            L = quantizer.simulated_quant(L)
            
        for _ in range(n_iter):
            R = quantizer.simulated_quant(
                torch.linalg.lstsq(L, X)[0]
            )

            L = quantizer.simulated_quant(
                torch.linalg.lstsq((R @ H_sqrt).T, (X @ H_sqrt).T)[0]
            ).T
    else:
        if (init_L is not None) and (init_R is not None):
            L, R = init_L, init_R
        else:
            U, S, _ = torch.svd_lowrank(X, q=rank*2)
            L = U[:, :rank] @ torch.diag(S.sqrt()[:rank])
            L = quantizer.simulated_quant(L)

        for _ in range(n_iter):
            R = quantizer.simulated_quant(
                torch.linalg.lstsq(L, X)[0]
            )

            L = quantizer.simulated_quant(
                torch.linalg.lstsq(R.T, X.T)[0]
            ).T
    return L, R