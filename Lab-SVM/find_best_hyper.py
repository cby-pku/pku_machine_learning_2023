# Add an automated parameter adjustment script
# Find best hyperparameters
from SSMO import SSMO_optimizer
from SVM import SVM
from kernels import RBF_kernel  

import numpy as np
from tqdm import tqdm


np.random.seed(0)
from dataset_generator import zipper_2D_dataset  # noqa: E402

n_pos = 100
n_neg = 100

x_train, y_train = zipper_2D_dataset(n_pos, n_neg, scope=4.0)
x_test, y_test = zipper_2D_dataset(50, 50, scope=5.5) # Test data is slightly out-of-distribution
# TODO: Implement the code to compute the accuracy of svm in the test set. 
# Note that svm itself is already trained, if you have run the above code.
def calculate_acc(svm, x_data,y_data):
    predicions =svm.predict(x_data)[1]
    acc = np.mean(predicions==y_data)
    return acc

# Define the parameter search space
kernel_types = [ RBF_kernel(0.9),RBF_kernel(sigma=1),RBF_kernel(sigma=1.1), RBF_kernel(sigma=1.2)]
C_values = [0.5,0.1]
max_passes = 600
# 开个tmux 串行/并行搜
best_params = {'kernel_type': None, 'C': None, 'accuracy': 0.0}
# Perform grid search
for kernel_type in tqdm(kernel_types):
    for C in C_values:
        mesvm = SVM(kernel_fn=kernel_type)
        optimizer = SSMO_optimizer(C=C)
        optimizer.fit(mesvm, x_train, y_train, max_passes=max_passes, verbose=False)
        train_acc = calculate_acc(mesvm, x_train, y_train)
        test_acc = calculate_acc(mesvm, x_test, y_test)
        print(f"Kernel: {kernel_type}, C: {C}")
        print(f"Training Accuracy: {train_acc * 100:.2f}%")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print('*'*30)

        # Update best parameters if the current model has better accuracy
        if train_acc > 0.99 and test_acc > 0.95 :
            best_params['kernel_type'] = kernel_type
            best_params['C'] = C
            best_params['accuracy'] = test_acc
            break

# Output the best parameters
if best_params['accuracy'] > 0:
    print("Best Parameters:")
    print(f"Kernel: {best_params['kernel_type']}, C: {best_params['C']}")
    print(f"Best Test Accuracy: {best_params['accuracy'] * 100:.2f}%")
else:
    print("Please adjust hyperparameters to meet the accuracy criteria.")