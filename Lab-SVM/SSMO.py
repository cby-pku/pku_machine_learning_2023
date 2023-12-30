import numpy as np
from tqdm import tqdm

class SSMO_optimizer():
    
    """
    Simplified Sequential Minimal Optimization (SSMO) algorithm for training SVM.
    """
    
    def __init__(
        self,
        C: float = 1.,
        kkt_thr: float = 1e-3,
    ) -> None:

        """
        Arguments:
            C : SVM's penalty parameter C.
            kkt_thr : Numerical threshold for KKT conditions, usually set to 1e-3.
        """
        # Initialize
        self.C = float(C)
        self.kkt_thr = kkt_thr
        
        self.b = None
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None


    def fit(
        self,
        SVM,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_passes: int = 1000,
        verbose: bool = False,
    ) -> None:

        """
        Train a SVM classifier.

        Arguments:
            SVM : The to-be-trained SVM classifier. The training process will overwrite the SVM's four attributes: 
                  self.b, self.alpha, self.support_vectors and self.support_labels.
            x_train : (N,D) data matrix, each row is a sample.
            y_train : Labels vector, y must be {-1,1}
            max_passes : Maximum number of passes through the training data for the algorithm.
            verbose : Whether to print the loss during training process.
            
        Return:
            None
            Note that the training process will overwrite the SVM's attributes, so one can first access the trained SVM's by SSMO_optimizer.SVM directly after training.
        """

        # Initialize
        N, _ = x_train.shape
        self.SVM = SVM
        
        # Initialize the four attributes of SVM to ensure that they initially satisfy the constraints of dual form of SVM.
        self.b = 0
        self.alpha = np.zeros(N)
        self.support_vectors = x_train
        self.support_labels = y_train
        self.update_SVM()

        passes = 0

        print("Start training using SMO")

        bar = tqdm()
       
        while passes < max_passes:
            num_changed_alphas = 0 
            for i in range(N):
                # TODO: implement SSMO algorithm for each pass (One pass means one iteration over the training data)
                # Hint: 1. First choose the two to-be-updated alpha variables, i.e., alpha[i] and alpha[j], based on the heuristic method mentioned in SVM_lab.ipynb.
                #       2. Then, use dual form of SVM's constraint to substitute alpha[i] by alpha[j] and optimize the polynomial function of alpha[j].
                #       3. Finally, update alpha[i] and alpha[j] based on the optimal solution of alpha[j].
                # Note: 1. Remember to call self.update_SVM() after updating self.b, self.alpha (self.support_vectors and self.support_labels).
                #       2. If eta is 0., we just skip this pair of alpha variables. 
                # FIXME 1
                if not self.judge_violoate_KKT(self.support_vectors[i], self.support_labels[i], self.alpha[i]):
                    continue
                #j = np.random.choice(np.concatenate((np.arange(i), np.arange(i+1, N))))
                j = np.random.choice(N)
                while j == i:
                    j = np.random.choice(N)
                xi, xj =self.support_vectors[i], self.support_vectors[j]
                yi, yj = self.support_labels[i], self.support_labels[j]
                
                L,H = self.compute_L_H(yi==yj,i,j)
                eta = self.compute_eta(xi, xj)
                
                if 0 < eta:
                    aj_new = self.compute_new_aj(self.support_vectors, self.support_labels, i, j, eta, L, H)
                    ai_new = self.compute_new_ai(self.support_labels, i, j, aj_new)
                    #print(f"debug for ai_new: {ai_new}")
                    #print(f"debug for aj_new: {aj_new}")
                    #print(f"debug for ai_new shape:{ai_new.shape}")
                    #print(f"debug for aj_new shape:{aj_new.shape}")

                # Update alpha[i] and alpha[j]
                    self.alpha[i] = ai_new
                    self.alpha[j] = aj_new

                # Update the bias term
                    self.b = self.compute_new_b(self.support_vectors, self.support_labels, i, j, ai_new, aj_new)
                    #print(f"self.b scop in ssmo.py: {self.b.shape}")
                # Update the SVM attributes
                    self.update_SVM()

                    num_changed_alphas += 1

            if verbose: 
                loss = self.eval_loss(self.support_vectors, self.support_labels) # Report the loss after each pass
                bar.set_description(f"loss: {loss:.4f}")
            
            if num_changed_alphas == 0:
                break
            else:
                passes += 1
                bar.update(1)


        
        # Only keep the support vectors
        self.support_labels = self.support_labels[self.alpha > 0]
        self.support_vectors = self.support_vectors[self.alpha > 0]
        self.alpha = self.alpha[self.alpha > 0]
        self.update_SVM()
        print("Training finished")
        return 

    def judge_violoate_KKT(
        self,
        xi,
        yi,
        alpha_i
    ) -> bool:
        """
        Judge whether the i-th sample violates KKT conditions.
        
        Arguments:
            xi : (D,), the i-th sample.
            yi : (1,), the label of i-th sample, must be {-1, 1}
            alpha_i : The dual variable of i-th sample.
            
        Return:
            True if the i-th sample violates KKT conditions, otherwise False.
            
        Note:
            The following conditions is directly derived from the KKT conditions of dual form of SVM. This is just one heuristic way to check whether the i-th sample is 
            suitable for optimization. kkt_thr is a numerical threshold, usually set to 1e-3.
        """
        
        if alpha_i == 0:
            return yi * self.predict_score(xi) < 1 - self.kkt_thr
        elif alpha_i == self.C:
            return yi * self.predict_score(xi) > 1 + self.kkt_thr
        else:
            return abs(yi * self.predict_score(xi) - 1) > self.kkt_thr
    
    
    def compute_E(
        self,
        xi,
        yi
    ):
        """
        Compute prediction error of the i-th sample.
        
        Arguments:
            xi : (D,), the i-th sample.
            yi : (1,), the label of i-th sample, must be {-1, 1}
            
        Return:
            The error of the prediction of the i-th sample.
        """
        #好叭这里没有问题！是我理解错了！！www
        return self.predict_score(xi) - yi
    
    
    def compute_L_H(
        self,
        is_yi_equals_yj,
        i,
        j,
    ):
        """
        Compute the left and right boundary of alpha[j], given the constraint of dual form of SVM, i.e., 0 <= alpha[i] <= C and \sum_k alpha[k] * y[k] = 0.
        
        Arguments:
            is_yi_equals_yj : True if y[i] == y[j], otherwise False.
            i : The index of the i-th sample.
            j : The index of the j-th sample.
            
        Return:
            L : The left boundary of **alpha[j]**.
            H : The right boundary of **alpha[j]**.
        """
        # TODO: implement compute_L_H. You can only use self.C and self.alpha additionally in this function.
        # FIXME 112 to 0.0
        if is_yi_equals_yj:
            L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0.0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        return L, H
        
    def compute_eta(self, xi, xj):
        '''
        Compute eta, which is the second derivative of the objective function of dual form of SVM (The equivlent minimization problem).
        
        Arguments:
            xi : (D,), the i-th sample.
            xj : (D,), the j-th sample.
        
        Return:
            eta : The second derivative of the objective function of dual form of SVM.
        '''
        
        eta = (
            - 2 * self.kernel_fn(xi, xj) 
            + self.kernel_fn(xi, xi)
            + self.kernel_fn(xj, xj)
        )
        eta = eta.squeeze()
        # FIXME 11
        return eta
    
    
    def compute_new_aj(self, x_train, y_train, i, j, eta, L, H):
        '''
        Compute the new value of alpha[j].

        Arguments:
            x_train : (N, D), the training data.
            y_train : (N, ), the labels of training data.
            i : The index of the i-th sample.
            j : The index of the j-th sample.
            eta : The second derivative of the objective function of dual form of SVM.
            L : The left boundary of alpha[j].
            H : The right boundary of alpha[j].
            
        Return:
            aj_new : The new value of alpha[j].
            
        Note:
            This is equivalent to solving the quadratic optimization problem for alpha[j] (alpha[i] is substituted by alpha[j] based on the constraint of dual form of SVM).
        '''
        
        if eta > 0:
            # TODO: implement compute_new_aj when eta > 0. 
            # Hint: By letting the first derivative of the objective function of dual form of SVM equal to 0, 
            #       one can finally derive that the non-bounded optimal solution of alpha_j_opt is alpha_j_opt = alpha_j_old + y_j * (E_i - E_j) / eta.
            # Compute the optimal solution of alpha[j] using the formula derived from the dual form of SVM optimization problem
            aj_opt = self.alpha[j] + y_train[j] * (self.compute_E(x_train[i], y_train[i]) - self.compute_E(x_train[j], y_train[j])) / eta
            
            # Clip aj_opt to ensure it lies within the bounds [L, H]
            # FIXME 2 np.clip()[0]
            # FIXME 12
            aj_new = np.clip(aj_opt, L, H)


            return aj_new
        else:
            # evaluate objective function at a1 = L and a1 = H
            
            Ei = self.compute_E(x_train[i], y_train[i])
            Ej = self.compute_E(x_train[j], y_train[j])
            kii = self.kernel_fn(x_train[i], x_train[i]).squeeze()
            kij = self.kernel_fn(x_train[i], x_train[j]).squeeze()
            kjj = self.kernel_fn(x_train[j], x_train[j]).squeeze()
            s = y_train[i] * y_train[j]
            
            f1 = y_train[i] * (Ei + self.b) - self.alpha[i] * kii - s * self.alpha[j] * kij
            f2 = y_train[j] * (Ej + self.b) - s * self.alpha[i] * kij - self.alpha[j] * kjj
            L1 = self.alpha[i] + s * (self.alpha[j] - L)
            H1 = self.alpha[i] + s * (self.alpha[j] - H)
            
            L_obj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * kii + 0.5 * L ** 2 * kjj + s * L * L1 * kij
            H_obj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * kii + 0.5 * H ** 2 * kjj + s * H * H1 * kij
            
            if L_obj < H_obj - 1e-3:
                aj_new = L
            elif L_obj > H_obj + 1e-3:
                aj_new = H
            else:
                aj_new = self.alpha[j]
                
            return aj_new
        
    
    
    def compute_new_ai(self, y_train, i, j, aj_new):
        '''
        Compute the new value of alpha[i].

        Arguments:
            y_train : (N, ), the labels of training data.
            i : The index of the i-th sample.
            j : The index of the j-th sample.
            aj_new : The new value of alpha[j].
            
        Return:
            ai_new : The new value of alpha[i].
        '''
        
        
        # TODO: implement compute_new_ai. You may only use self.alpha additionally in this function.
        # Compute the new value of alpha[i] using the relationship between alpha[i] and alpha[j] in the dual form of SVM
        ai_new = self.alpha[i] + y_train[i] * y_train[j] * (self.alpha[j] - aj_new)
        # FIXME 13
        return ai_new
    
    def compute_new_b(self, x_train, y_train, i, j, ai_new, aj_new):
        
        # Compute the new value of b based on the new values of alpha[i] and alpha[j].
        # This is to ensure that the KKT conditions are satisfied under the new values of alpha[i] and alpha[j].
        
        Ei = self.compute_E(x_train[i], y_train[i])
        Ej = self.compute_E(x_train[j], y_train[j])
        kii = self.kernel_fn(x_train[i], x_train[i]).squeeze()
        kjj = self.kernel_fn(x_train[j], x_train[j]).squeeze()
        kij = self.kernel_fn(x_train[i], x_train[j]).squeeze()
        #print(f"debug for Ei shape:{Ei.shape}")
        #print(f"debug for kii shape: {kii.shape}")
        
        bi = -(Ei + y_train[i] * (ai_new - self.alpha[i]) * kii + y_train[j] * (aj_new - self.alpha[j]) * kij) + self.b
        bj = -(Ej + y_train[i] * (ai_new - self.alpha[i]) * kij + y_train[j] * (aj_new - self.alpha[j]) * kjj) + self.b
        #print(f"debug for bi shape:{bi.shape}")
        #print(f"debug for bj shape:{bj.shape}")
        if 0 < ai_new < self.C:
            b_new = bi
        elif 0 < aj_new < self.C:
            b_new = bj
        else:
            
            b_new = (bi + bj) / 2
        # FIXME 14
        return float(b_new)
    
    
    def predict_score(self, x):
        return self.SVM.predict(x)[0]
    
    def predict_pred(self,x):
        return self.SVM.predict(x)[1]
    def kernel_fn(self, x1, x2):
        return self.SVM.kernel_fn(x1, x2)
    
    def update_SVM(self):
        # Update the four attributes of SVM after each pass
        self.SVM.b = self.b
        self.SVM.alpha = self.alpha
        self.SVM.support_vectors = self.support_vectors
        self.SVM.support_labels = self.support_labels
        
    def eval_loss(self, x_train, y_train):
        # Evaluate the loss of the current SVM model w.r.t. the training data
        alpha_M = self.alpha[:, np.newaxis] * self.alpha[np.newaxis, :]
        y_train_M = y_train[:, np.newaxis] * y_train[np.newaxis, :]
        kernel_M = self.kernel_fn(x_train, x_train)
        
        loss = (0.5 * kernel_M * alpha_M * y_train_M).sum() - self.alpha.sum()
            
        return loss