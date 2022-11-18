import numpy as np
from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def step(self, x):
        pass
    
class SGD(Optimizer):
    def __init__(self, params : list, lr, momentum = 0):
        self.params = params
        self.momentum = momentum
        self.lr = lr
        self.buf = []
        for param in (self.params):
            self.buf.append((np.zeros(param["val"].shape)))
        
    def step(self):
        for i, param in enumerate(self.params): 
            buffer_momentum = self.buf[i] 
            self.buf[i] = self.momentum * buffer_momentum + param["grad"]
            param["val"] -= (self.lr * self.buf[i])
            