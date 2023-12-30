import numpy as np


def gen_2D_dataset(n_pos, n_neg, noise):
    X1_class0 = np.random.normal(1, 0.3, n_pos) + noise * np.random.normal(0.0, 1.0, n_pos)
    X2_class0 = np.random.normal(1, 0.3, n_pos) + noise * np.random.normal(0.0, 1.0, n_pos)
    

    X1_class1 = np.random.normal(3, 0.3, n_neg) + noise * np.random.normal(0.0, 1.0, n_neg)
    X2_class1 = np.random.normal(3, 0.3, n_neg) + noise * np.random.normal(0.0, 1.0, n_neg)

    X_class0 = np.stack((X1_class0, X2_class0), axis=1)
    X_class1 = np.stack((X1_class1, X2_class1), axis=1)

    X = np.concatenate((X_class0, X_class1), axis=0)
    Y = np.array([1]*n_pos + [-1]*n_neg)
    
    return X, Y
