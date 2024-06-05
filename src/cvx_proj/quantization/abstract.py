
from abc import ABC, abstractmethod

class AbstractQuantizer(ABC):
    @abstractmethod
    def simulated_quant(self, weight):
        ...