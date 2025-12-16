import numpy as np
from src.bootstrap import Bootstrap
from src.jackknife import Jackknife
from src.autocorrelation import Autocorrelation

np.random.seed(42)
data = np.random.normal(loc=2.0, scale=1.0, size=10**6)

bootstrap = Bootstrap(
    data,
    blocks=600,
    boot_samples=10**2,
    function=lambda a: a,
    primary_functions=[lambda x: x**2],
)

jackknife = Jackknife(
    data, blocks=600, function=lambda x: x, primary_functions=[lambda x: x**2]
)

autocorr = Autocorrelation(data, prop=6, therm=100, function=lambda x: x**2)

boot_mean, boot_std = bootstrap()
jack_mean, jack_std = jackknife()
tau_int = autocorr()

print(f"\nBoot: {boot_mean} +/- {boot_std}")
print(f"Jack: {jack_mean} +/- {jack_std}")
print(f"tau_int: {tau_int}\n")
