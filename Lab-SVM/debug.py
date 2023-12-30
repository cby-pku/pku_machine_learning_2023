import numpy as np


np.random.seed(0)

from dataset_generator import zipper_2D_dataset  # noqa: E402

n_pos = 100
n_neg = 100

x_train, y_train = zipper_2D_dataset(n_pos, n_neg, scope=4.0)
x_test, y_test = zipper_2D_dataset(50, 50, scope=5.5)

from vis_utils import visualize_2D_dataset  # noqa: E402

visualize_2D_dataset(x_train, y_train)

from SVM import SVM  # noqa: E402
from kernels import Linear_kernel  # noqa: E402

kernel_fn = Linear_kernel()
svm = SVM(kernel_fn=kernel_fn)

from SSMO import SSMO_optimizer  # noqa: E402
C = 1.0
max_passes = 1000

optimizer = SSMO_optimizer(C=C)
optimizer.fit(svm, x_train, y_train, max_passes=max_passes, verbose=True)