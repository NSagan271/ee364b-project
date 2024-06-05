import torch
from cvx_proj.quantization.abstract import AbstractQuantizer


def init_QLR_data_agnostic(
    W: torch.Tensor,
    rank: int,
    quant_Q: AbstractQuantizer,
    quant_LR: AbstractQuantizer
):
    U, Sigma, _ = torch.linalg.svd(W, full_matrices=False)
    L = U[:, :rank] @ \
        torch.diag(Sigma[:rank].sqrt())
    L = quant_LR.simulated_quant(L)

    R = torch.linalg.lstsq(L, W)[0]
    if torch.isnan(R).any():
        R = torch.linalg.pinv(L) @ W
    R = quant_LR.simulated_quant(R)

    Q = quant_Q.simulated_quant(W - L @ R)

    return Q, L, R

def init_QLR_data_aware(
    W: torch.Tensor,
    eigH,
    rank: int,
    quant_Q: AbstractQuantizer,
    quant_LR: AbstractQuantizer
):
    H_sqrt = eigH.eigenvectors @ torch.diag(eigH.eigenvalues.sqrt()) @ eigH.eigenvectors.T
    Y = W @ H_sqrt @ eigH.eigenvectors
    U, _, _ = torch.linalg.svd(Y, full_matrices=False)
    L = U[:, :rank]
    L = quant_LR.simulated_quant(L)

    R = torch.linalg.lstsq(L, W)[0]
    if torch.isnan(R).any():
        R = torch.linalg.pinv(L) @ W
    R = quant_LR.simulated_quant(R)

    Q = quant_Q.simulated_quant(W - L @ R)

    return Q, L, R
