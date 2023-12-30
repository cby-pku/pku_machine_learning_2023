import numpy as np
import matplotlib.pyplot as plt


def visualize_2D_dataset(x, y):
    
    assert x.shape[1] == 2 # x must be 2D for visualization, otherwise one can consider first using PCA to reduce the dimensionality of x
    
    
    x_pos = x[y == 1]
    x_neg = x[y == -1]
    
    plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', label='Positive')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], c='blue', label='Negative')
    plt.title('Zipper-like Dataset')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def visualize_2D_border(svm, x1_interval, x2_interval, x_train=None, y_train=None):
    '''
        x1_interval: tuple, (x1_min, x1_max)
        x2_interval: tuple, (x2_min, x2_max)
    '''
    
    
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_interval[0], x1_interval[1], 100), np.linspace(x2_interval[0], x2_interval[1], 100))
    x_grid = np.concatenate([x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)], axis=1)
    
    pred_y = svm.predict(x_grid)[1]
    
    plt.contourf(x1_grid, x2_grid, pred_y.reshape(x1_grid.shape), cmap=plt.cm.coolwarm, alpha=0.8)
    plt.contour(x1_grid, x2_grid, pred_y.reshape(x1_grid.shape), colors='k', linewidths=0.5)
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    
    if x_train is not None and y_train is not None:
        x_pos = x_train[y_train == 1]
        x_neg = x_train[y_train == -1]
        
        plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', label='Positive')
        plt.scatter(x_neg[:, 0], x_neg[:, 1], c='blue', label='Negative')
        plt.legend()
    
    plt.show()