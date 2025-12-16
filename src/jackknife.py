import numpy as np


def mean(data: np.ndarray, function: callable = lambda x: x**2) -> float:
    return np.mean(function(data))


def checker(x: float, y: float) -> bool:
    return True if max(x, y) / min(x, y) >= 100 else False


class Jackknife:
    def __init__(
        self,
        data: np.ndarray,
        *,
        blocks: int,
        therm: int = 0,
        function: callable = lambda x: x**2,
        primary_functions: list[callable] = []
    ):
        data = data[therm:]  # discard initial therm steps
        self.data = data
        self.k = blocks
        self.n = len(self.data)
        self.function = function
        self.primary_functions = primary_functions

        try:
            assert checker(self.n, self.k) is True
        except AssertionError:
            print(
                f"\nWarning: Number of blocks k={self.k} is too large for data size n={self.n} "
                "(for a good bootstrap k << n is suggested)."
            )

        self.blocks = np.array_split(self.data, self.k)

    def execute(self) -> tuple[float, float]:
        if self.primary_functions != []:
            """Calculating mean:"""
            g_alpha = []
            mean_element = []
            for func in self.primary_functions: 
                g_alpha_list = []
                S = sum(func(self.data))
                for i in range(self.k):
                    g_alpha_i = (S - sum(func(self.blocks[i])))/ (self.n - len(self.blocks[i]))
                    g_alpha_list.append(g_alpha_i)
                _ = np.mean(g_alpha_i)
                mean_element.append(_)
            g_alpha.append(g_alpha_list)
            expected_mean = self.function(*mean_element)

            """Calculating std:"""

            expected_std = 0

            return expected_mean, expected_std

        F_val = []
        S = sum(self.function(self.data))
        aux = 0

        for i in range(self.k):
            F_i = S - sum(self.function(self.blocks[i]))
            F_i = F_i / (self.n - len(self.blocks[i]))
            F_val.append(F_i)

        for i in range(self.k):
            partial_sum = (F_val[i] - np.mean(F_val)) ** 2
            partial_sum = partial_sum * (self.n - self.n // self.k) / self.n
            aux += partial_sum

        expected_std = aux**0.5
        expected_mean = mean(self.data, function=self.function)
        return expected_mean, expected_std

    def __call__(self):
        return self.execute()
