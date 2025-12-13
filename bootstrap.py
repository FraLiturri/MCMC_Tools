import numpy as np
from typing import Iterable


def f(x):
    return x**2


def mean(data: list | Iterable, function : callable = f) -> float:
    return np.mean(function(data))


def square_mean(data: list | Iterable, function : callable = f) -> float:
    return np.mean(np.square(function(data)))


def checker(
    x: float, y: float
) -> bool:  # returns True if x is significantly greater than y (or viceversa);
    if x >= y:
        return True if x / 100 >= y else False
    else:
        return True if y / 100 >= x else False


class Bootstrap:
    def __init__(self, data: list | Iterable, *, blocks: int, boot_samples: int, function : callable = f):
        self.data = data
        self.k = blocks
        self.r = boot_samples
        self.n = len(data)
        self.function = function

        try:
            assert checker(self.n, self.k) is True
        except Exception:
            print(
                f"\nWarning: Number of blocks k={self.k} is too large for data size n={self.n} (for a good bootstrap k<<n is suggested)."
            )

        self.blocks = np.array_split(data, self.k)

    def sample_blocks(self) -> list:
        sampled_indices = np.random.randint(0, self.k, self.k)
        sampled_blocks = [self.blocks[i] for i in sampled_indices]
        return np.concatenate(sampled_blocks)

    def execute(self) -> tuple[float, float]:
        s = []
        s_square = []
        for i in range(0, self.r):
            sampled_data = self.sample_blocks()
            s.append(mean(sampled_data, function=self.function))
            s_square.append(square_mean(sampled_data, function=self.function))

        estimated_mean = mean(self.data, function=self.function)
        estimated_variance = (
            self.r / (self.r - 1) * (sum(s_square) / self.r - (sum(s) / self.r) ** 2)
        )
        return estimated_mean, estimated_variance**0.5

    def __call__(self):
        return self.execute()
