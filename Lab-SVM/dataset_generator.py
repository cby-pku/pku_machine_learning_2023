import numpy as np


def zipper_2D_dataset(n_pos, n_neg, scope=4.0):
    
    halfs = scope / 2
    
    x1_pos= np.random.uniform(low=-halfs * np.pi, high= halfs * np.pi, size=n_pos)
    x2_pos = np.sin(x1_pos) + 0.5
    
    x1_neg = np.random.uniform(low=-halfs * np.pi, high= halfs * np.pi, size=n_neg)
    x2_neg = np.sin(x1_neg) - 0.5
    
    
    noise_strength = 0.1
    x2_pos += np.random.randn(n_pos) * noise_strength
    x2_neg += np.random.randn(n_neg) * noise_strength
    
    
    x_pos = np.stack([x1_pos, x2_pos], axis=1)  # shape (n_pos, 2)
    x_neg = np.stack([x1_neg, x2_neg], axis=1)  # shape (n_neg, 2)
    
    x = np.concatenate([x_pos, x_neg], axis=0)  # shape (n_pos + n_neg, 2)
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])  # shape (n_pos + n_neg,)
    
    
    return x, y



    
    