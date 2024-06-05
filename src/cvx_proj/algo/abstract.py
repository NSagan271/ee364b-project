from abc import ABC, abstractmethod
from torch import Tensor, trace
from math import sqrt
from cvx_proj.quantization.abstract import AbstractQuantizer

class AbstractDecomposition(ABC):

    def __init__(
        self,
        rank: int,
        data_aware: bool,
        quant_Q: AbstractQuantizer,
        quant_LR: AbstractQuantizer
    ):
        self.rank = rank
        self.errors = []
        self.quant_Q = quant_Q
        self.quant_LR = quant_LR
        self.data_aware = data_aware
        
    @abstractmethod
    def decompose(
        self,
        W: Tensor,
        H: Tensor,
        n_iter: int,
        **kwargs
    ):
        ...

    def get_error(self, W: Tensor, W_hat: Tensor, H: Tensor):
        if self.data_aware:
            error_sq = (trace((W_hat - W) @ H @ (W_hat - W).T) /
                        trace(W @ H @ W.T)).item()
        else:
            error_sq = (trace((W - W_hat) @ (W - W_hat).T) /
                        trace(W @ W.T)).item()
        return sqrt(error_sq)