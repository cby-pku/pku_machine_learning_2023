import numpy as np


class Base_kernel():
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Linear kernel function.
        
        Arguments:
            x1: shape (n1, d)
            x2: shape (n2, d)
            
        Returns:
            y : shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j])
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        
        return np.dot(x1, x2.T)
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function
        
        return (np.dot(x1, x2.T) + self.c) ** self.degree

class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
 
        
        
    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        # FIXME 5 √ same result
        #norm_sq = np.linalg.norm(x1[:,np.newaxis] - x2.T ,axis= -1) ** 2
        #norm_sq = np.linalg.norm(x1 - x2)**2
        # FIXME 15
        norm_sq = np.linalg.norm(x1 - x2, axis=-1)**2
        return np.exp(-norm_sq / (2* self.sigma **2))