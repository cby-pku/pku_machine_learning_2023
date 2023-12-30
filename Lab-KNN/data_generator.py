from sklearn.datasets import make_blobs

def gen_2D_dataset(centers, n_samples, cluster_std):
    # define the centers of different classes
    centers = centers
    # define the number of samples
    n_samples = n_samples
    # generate the dataset
    X, y = make_blobs(n_samples=n_samples, centers=centers,cluster_std=cluster_std, random_state=0)
    
    return X, y

