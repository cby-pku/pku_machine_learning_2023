import numpy as np
import matplotlib.pyplot as plt


def visualize_2D_dataset(x, y):
    
    assert x.shape[1] == 2 # x must be 2D for visualization, otherwise one can consider first using PCA to reduce the dimensionality of x
    
    
    x_pos = x[y == 1]
    x_neg = x[y == -1]
    
    plt.scatter(x_pos[:, 0], x_pos[:, 1], c='red', label='Positive')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], c='blue', label='Negative')
    plt.title('Dataset')
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
    
def plot_training_progress(i, x, y, w, w_module_history, loss_history):
    plt.figure(figsize=(12, 5))

    # Plot loss history
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Loss')

    # 如果数据只有两个特征，绘制数据和决策边界
    if x.shape[1] == 3:  # 2 features + 1 bias
        plt.subplot(1, 2, 2)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1])
        yy = np.linspace(ylim[0], ylim[1])
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = np.dot(np.c_[xy, np.ones(xy.shape[0])], w)
        Z = Z.reshape(XX.shape)

        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
        plt.title('Decision Boundary')

    plt.tight_layout()
    plt.show()

