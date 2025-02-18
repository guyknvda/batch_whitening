'''
The code is base on: https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/matrix_functions.py
'''

import torch
from torch import nn

from timeit import default_timer as timer
import matplotlib.pyplot as plt


device = torch.device('cuda:0')  # or other device/cpu (cuda:0)


#@torch.no_grad()
def MatPower(mat_m, p):
    """Computes mat_m^p, for p a positive integer.

    Args:
        mat_m: a square matrix
        p: a positive integer

    Returns:
        mat_m^p
    """
    if p in [1, 2, 4, 8, 16, 32]:
        p_done = 1
        res = mat_m
        while p_done < p:
            res = torch.matmul(res, res)
            p_done *= 2
        return res

    power = None
    while p > 0:
        if p % 2 == 1:
            power = torch.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = torch.matmul(mat_m, mat_m)
    return power


#@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
    """Power iteration.

    Compute the maximum eigenvalue of mat, for scaling.
    v is a random vector with values in (-1, 1)

    Args:
        mat_g: the symmetric PSD matrix.
        error_tolerance: Iterative exit condition.
        num_iters: Number of iterations.

    Returns:
        eigen vector, eigen value, num_iters
    """
    v = torch.rand(list(mat_g.shape)[0], device=device) * 2 - 1
    error = 1
    iters = 0
    singular_val = 0
    while error > error_tolerance and iters < num_iters:
        v = v / torch.norm(v)
        mat_v = torch.mv(mat_g, v)
        s_v = torch.dot(v, mat_v)
        error = torch.abs(s_v - singular_val)
        v = mat_v
        singular_val = s_v
        iters += 1
    return singular_val, v / torch.norm(v), iters


#@torch.no_grad()
def ComputePower(mat_g, p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
    """A method to compute G^{-1/p} using a coupled Newton iteration.

    See for example equation 3.2 on page 9 of:
    A Schur-Newton Method for the Matrix p-th Root and its Inverse
    by Chun-Hua Guo and Nicholas J. Higham
    SIAM Journal on Matrix Analysis and Applications,
    2006, Vol. 28, No. 3 : pp. 788-804
    https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

    Args:
        mat_g: A square positive semidefinite matrix
        p: a positive integer
        iter_count: Stop iterating after this many rounds.
        error_tolerance: Threshold for stopping iteration
        ridge_epsilon: We add this times I to G, to make it positive definite.
                    For scaling, we multiply it by the largest eigenvalue of G.
    Returns:
        (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
    """
    shape = list(mat_g.shape)
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1/p)
    identity = torch.eye(shape[0], device=device)
    if shape[0] == 1:
        return identity
    alpha = -1.0/p
    max_ev, _, _ = PowerIter(mat_g)
    ridge_epsilon *= max_ev
    mat_g += ridge_epsilon * identity
    z = (1 + p) / (2 * torch.norm(mat_g))
    # The best value for z is
    # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
    #            (c_max^{1+1/p} - c_min^{1+1/p})
    # where c_max and c_min are the largest and smallest singular values of
    # mat_g.
    # The above estimate assumes that c_max > c_min * 2^p
    # Can replace above line by the one below, but it is less accurate,
    # hence needs more iterations to converge.
    # z = (1 + p) / tf.trace(mat_g)
    # If we want the method to always converge, use z = 1 / norm(mat_g)
    # or z = 1 / tf.trace(mat_g), but these can result in many
    # extra iterations.

    mat_root = identity * torch.pow(z, 1.0/p)
    mat_m = mat_g * z
    error = torch.max(torch.abs(mat_m - identity))
    count = 0
    while error > error_tolerance and count < iter_count:
        tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
        new_mat_root = torch.matmul(mat_root, tmp_mat_m)
        mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
        new_error = torch.max(torch.abs(mat_m - identity))
        if new_error > error * 1.2:
            break
        mat_root = new_mat_root
        error = new_error
        count += 1
    return mat_root


def inv_sqrt_through_eigendecomposition(A):
    '''
    Compute the inverse square root of A using eigendecomposition.
    Note: the eigenvectors of A are the columns of the 'eigenvectors' matrix returned by torch.linalg.eigh().
    '''
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    inv_sqrt_eigenvalues_diag = torch.eye(A.shape[0], device=device) * eigenvalues**(-0.5)
    inv_sqrt = eigenvectors @ inv_sqrt_eigenvalues_diag @ eigenvectors.T

    return inv_sqrt


def random_symmetric_psd_matrix(dim):
    B = torch.randn(dim, dim, device=device)
    B = B @ B.T  # B is now symmetric and PSD
    return B


def inv_sqrt_through_cholesky(A):
    '''
    Note that cholesky factorization returns a matrix L such that A = L @ L.T
    However, A != L @ L. Hence L is not the sqrt of A and therefore inv(L) does not equal to the inverse sqrt of A.
    As a result, cholesky cannot be used for this task.
    
    Also, the matrix A must be positive definite for cholesky. We can convert A to be PD by adding a small epsilon to the eigenvalues:
    >> eigenvalues, eigenvectors = torch.linalg.eigh(A)
    >> eigenvalues = eigenvalues + 1e-8
    >> A = eigenvectors @ (torch.eye(dim) * eigenvalues) @ eigenvectors.T
    '''
    inv_sqrt_cholesky = torch.inverse(torch.linalg.cholesky(A))
    return inv_sqrt_cholesky


# https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314/8
# https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


if __name__ == "__main__":
    # For this example the result was validated with wolframalpha: https://www.wolframalpha.com/input?i=%7B%7B3%2C+1%7D%2C+%7B1%2C+3%7D%7D%5E%28-0.5%29
    A = torch.tensor(((3., 1.), (1., 3.)), device=device)
    inv_sqrt = inv_sqrt_through_eigendecomposition(A)
    inv_sqrt_newton = ComputePower(A, 2)
    assert torch.allclose(inv_sqrt, inv_sqrt_newton)

    B = random_symmetric_psd_matrix(1000)
    inv_sqrt = inv_sqrt_through_eigendecomposition(B)
    inv_sqrt_newton = ComputePower(B.clone(), 2, ridge_epsilon=1e-10)
    #assert torch.allclose(inv_sqrt, inv_sqrt_newton, atol=1e-1)



    # Get accuracy as a function of ridge_epsilon, iter_count, and error_tolerance
    mse_fn = nn.MSELoss()
    
    ridge_epsilons = [10**(-x) for x in range(1, 15)]
    mse_arr = []
    for ridge_epsilon in ridge_epsilons:
        inv_sqrt_newton = ComputePower(B.clone(), 2, ridge_epsilon=ridge_epsilon)
        mse = mse_fn(inv_sqrt, inv_sqrt_newton)
        mse_arr.append(mse.cpu())
    
    fig, ax = plt.subplots()
    ax.plot(ridge_epsilons, mse_arr)
    ax.set_xlabel('ridge_epsilon')
    ax.set_ylabel('mse')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Newton accuracy as a function of ridge_epsilon')
    plt.show(block=False)

    iter_counts = range(1, 101)
    mse_arr = []
    for iter_count in iter_counts:
        inv_sqrt_newton = ComputePower(B.clone(), 2, iter_count=iter_count, ridge_epsilon=1e-10)
        mse = mse_fn(inv_sqrt, inv_sqrt_newton)
        mse_arr.append(mse.cpu())
    
    fig, ax = plt.subplots()
    ax.plot(iter_counts, mse_arr)
    ax.set_xlabel('iter_count')
    ax.set_ylabel('mse')
    ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Newton accuracy as a function of iter_count')
    plt.show(block=False)

    error_tolerances = [10**(-x) for x in range(1, 15)]
    mse_arr = []
    for error_tolerance in error_tolerances:
        inv_sqrt_newton = ComputePower(B.clone(), 2, error_tolerance=error_tolerance, ridge_epsilon=1e-10)
        mse = mse_fn(inv_sqrt, inv_sqrt_newton)
        mse_arr.append(mse.cpu())
    
    fig, ax = plt.subplots()
    ax.plot(error_tolerances, mse_arr)
    ax.set_xlabel('error_tolerance')
    ax.set_ylabel('mse')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Newton accuracy as a function of error_tolerance')
    plt.show(block=False)

    # Get ComputePower runtime as a function of iter_count (for small error_tolerance, such that iter_count is the effective stopping condition)
    num_iters = 20
    iter_counts = range(0, 200, 5)
    runtime_arr = []
    for iter_count in iter_counts:
        start = timer()
        for i in range(num_iters):
            inv_sqrt_newton = ComputePower(B.clone(), 2, iter_count=iter_count+1, ridge_epsilon=1e-10, error_tolerance=1e-14)
        end = timer()
        runtime_arr.append((end - start) / num_iters)
        if iter_count % 20 == 0:
            print(f'Completed {iter_count}')
    
    fig, ax = plt.subplots()
    ax.plot(iter_counts, runtime_arr)
    ax.set_xlabel('iter_count')
    ax.set_ylabel('runtime (sec)')
    ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Newton runtime as a function of iter_count')
    plt.show(block=False)
    



    print('Done')
