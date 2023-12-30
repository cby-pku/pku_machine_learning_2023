
import numpy as np
from typing import Tuple

class SVM():
    
    """
    Support Vector Machine model.
    """

    def __init__(
        self,
        kernel_fn,
    ) -> None:

        """
        Arguments:

        """

        self.kernel_fn = kernel_fn # Kernel function as one object in **kernels.py**
        self.b = None  # SVM's threshold, shape (1,)
        self.alpha = None  # SVM's dual variables, shape (n_support,)
        self.support_labels = None  # SVM's dual variables, shape (n_support,), in {-1, 1}
        self.support_vectors = None  # SVM's support vectors, shape (n_support, d)

    def predict(
        self,
        x: np.ndarray 
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        SVM predict method (dual form with some kernel).

        Arguments:
            x : (n, d), where n represents the number of samples, d the number of features

        Return:
            scores : (n,), SVM scores, where scores[i] is the score for x[i]
            pred : (n,), SVM predictions, where pred[i] is the prediction for x[i], in {-1, 1}
        """
        # FIXME 2023-12-28 002 during training used by compute_E，where xi is single sample whose n is 1
        if len(x.shape) == 1:
            # For train: Single sample case , reshape to (1,D)
            # For test:(N,D) Data
            x = x.reshape(1,-1)
        n_samples , _ =x.shape
        scores = np.zeros(n_samples)
        pred = np.zeros(n_samples)
        for i in range(n_samples):
            scores[i],pred[i] = self.predict_single(x[i])
        return scores,pred

    def predict_single(self,x_i:np.ndarray) -> Tuple[float,int]:
        """
        Predict the score for a single sample.

        Arguments:
            x_i : (D,), a single sample

        Return:
            prediction : float, the score for x_i
            label : int, the predicted label for x_i, in {-1, 1}
        """
        prediction = 0.0
            # caculate the predicion using the support vectors and corresponding dual varialbles
        for alpha,label,support_vector in zip(self.alpha,self.support_labels,self.support_vectors):
            prediction +=  alpha*label*self.kernel_fn(x_i,support_vector)
        prediction += self.b
        label = np.sign(prediction)
        # NOTE Attention! np.sign return 0 when input is 0
        if label == 0:
            label = 1
        return prediction,label