import numpy as np


def generate_data():
    true_b = 1
    true_w = 2
    N = 100

    np.random.seed(42)
    x = np.random.rand(N, 1)
    epsilon = (.1 * np.random.randn(N, 1))
    y = true_b + true_w * x + epsilon

    return x, y