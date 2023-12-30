import numpy as np


def gen_2D_dataset(num=100):
    
    X = np.random.rand(num)
    y = 3 * X + 2

    y += np.random.randn(num) * 0.2
    return X, y
