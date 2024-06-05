from cvx_proj.quantization.abstract import AbstractQuantizer
from cvx_proj.quantization.lattice_codebooks.twobit import E8P12_codebook
from cvx_proj.quantization.lattice_codebooks.fourbit import E8P12RVQ4B_codebook
from copy import deepcopy


class LatticeQuantizer(AbstractQuantizer):
    def __init__(self, num_bits=2, quant_batch_size=32_000):
        self.num_bits = num_bits
        assert num_bits in [2, 4]
        self.quant_batch_size = quant_batch_size

        if self.num_bits == 2:
            self.codebook = E8P12_codebook()
        else:
            self.codebook = E8P12RVQ4B_codebook()
    
    def simulated_quant(self, X, fixed_scale=None):
        self.codebook = self.codebook.to(X.device)
        if fixed_scale is not None:
            scale = fixed_scale
        else:
            scale = X.square().mean().sqrt().item()

        m, n = X.shape
        X = deepcopy(X)
        X = X.reshape(-1, self.codebook.codesz).clone() / scale

        K = self.quant_batch_size
        for i in range(0, X.shape[0], K):
            X[i:i+K] = self.codebook.quantize(X.float()[i:i+K], return_idx=False)
        return X.reshape(m, n) * scale