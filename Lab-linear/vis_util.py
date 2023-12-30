import matplotlib.pyplot as plt


def visualize_2D_dataset(x, y):
    plt.scatter(x, y, color='red')
    
    plt.title('Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def visualize_2D_border(x, y, y_pred):

    plt.scatter(x, y, color='red')
    plt.plot(x, y_pred, color='blue')
    
    plt.title('Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    