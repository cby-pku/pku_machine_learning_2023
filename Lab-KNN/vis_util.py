import matplotlib.pyplot as plt

def visualize_2D_dataset(x, y):
    
    assert x.shape[1] == 2 # x must be 2D for visualization, otherwise one can consider first using PCA to reduce the dimensionality of x
    
    # 绘制数据
    plt.figure(figsize=(6, 4))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title("Datasets")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()


