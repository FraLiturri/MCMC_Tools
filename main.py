import os
import numpy as np
from bootstrap import Bootstrap

filepath = "C:/Users/franc/Downloads/Q6.txt"
data = np.loadtxt(filepath)

bootstrap = Bootstrap(data, blocks=100, boot_samples=10**3)
mean_f, stddev = bootstrap()
print(f"\nEstimated Mean: {mean_f} \nEstimated StdDev: {stddev} \n")