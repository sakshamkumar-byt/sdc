import pandas as pd
from numba import njit


@njit
def series_apply():
    s = pd.Series([20.12, 21.2, 12.3],
                  index=['London', 'New York', 'Helsinki'])

    def square(x, *args):
        return x ** 2

    return s.transform(square)


@njit
def series_apply_arguments():
    s = pd.Series([20, 21, 12],
                  index=['London', 'New York', 'Helsinki'])

    def sum(x, *args):
        for i in args:
            x += i
        return x

    return s.transform(sum, 1, 2, 3, 4)


print(series_apply())
print(series_apply_arguments())
