import numpy as np 
import math 

def f(x):
    return x 

class Autocorrelation:
    def __init__(self, data: list | np.array, *, max_lag: int, function : callable = f):
        self.data = data
        self.n = len(data)
        self.max_lag = max_lag
        self.function = function
        