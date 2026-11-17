"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import contextlib
import math
import inspect
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import sys

sys.path.append('../..')


def get_batch_whitening_config(B, T, C, momentum=0.99, threshold=0.01):
    """Set Batch Whitening configuration based on number of samples

    Args:
        B (int): Number of sequences in the batch
        T (int): Number of tokens in a sequence
        C (int): Number of channels / features per token
        threshold (float): Threshold for using BatchWhitening.
            This is the allowed max estimation error.

    The mechanism:
    the effective number of samples is N*H*W/(1-momentum)
    this number should be >= blk_size*blk_size/2*threshold^2

    we can control blk_size, so we want to find blk_size that satisfies
    the condition above.
    so we solve for blk_size in the equation:
    N*H*W/(1-momentum) >= blk_size*blk_size/2*threshold^2
    blk_size >= sqrt(2*(B - 1)*T/(1-momentum)*threshold^2)
    now, if blk_size < 2 , we set blk_size to 1, which means using batchnorm.
    if blk_size > C we clip it to C which means using the whole channel
    as a group.
    in between, we use blk_size.
    """
    print('-----------BW triage------------ \n')
    n_samples = (B - 1) * T / (1-momentum)

    blk_size = int(np.sqrt(2*n_samples*threshold*threshold))
    print(f'raw block size: {blk_size}')
    new_mom = momentum
    if blk_size < 2:
        blk_size = 1
    elif blk_size > C:
        blk_size = C
        # new_mom=max(0,1-(2*threshold*threshold*N*H*W)/(C*C))
    else:
        # Need the block size to be the closest divisor of C
        target = blk_size
        divisors = [d for d in range(1, C + 1) if C % d == 0]
        blk_size = min(divisors, key=lambda d: (abs(d - target), -d))
    print(
        f'B, T, C, mu={B,T,C,momentum} '
        f'--> blk_size = {blk_size}, momentum={new_mom}'
    )
    return blk_size, new_mom


def fix_cov(covmat, fix_factor=0.9):
    """
    Stabilize covariance matrix:
    - Diagonal entries set to 1.0
    - Off-diagonal entries scaled by fix_factor
    Supports 2D [D,D] or 3D [B,D,D] or 4D [B,n_groups,D,D]
    """
    if covmat.dim() not in (2, 3, 4):
        raise ValueError(f"Unsupported covmat.dim()={covmat.dim()}")
    D = covmat.size(-1)
    key = (D, float(fix_factor), covmat.device, covmat.dtype)
    a = _FIX_MASKS.get(key)
    if a is None:
        a = torch.full((D, D), float(fix_factor), device=covmat.device, dtype=covmat.dtype)
        a.fill_diagonal_(1.0)
        _FIX_MASKS[key] = a
    return a * covmat


_FIX_MASKS = {}
_EYES = {}


def _eye(D, device, dtype):
    """Cached identity (read-only)."""
    key = (D, device, dtype)
    if key not in _EYES:
        _EYES[key] = torch.eye(D, device=device, dtype=dtype)
    return _EYES[key]


def shrink_cov(covmat, fix_factor, target="diag"):
    """fix_cov for target="diag"; "identity" shrinks toward (mean variance) * I
    instead (Ledoit-Wolf), which also pulls the channel variances together."""
    if target == "diag":
        return fix_cov(covmat, fix_factor)
    G = covmat.size(-1)
    eye = torch.eye(G, device=covmat.device, dtype=covmat.dtype)
    mean_var = covmat.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    return fix_factor * covmat + (1 - fix_factor) * mean_var * eye


def batch_center_only(
    X,
    gamma=None,
    beta=None,
    running_mean=None,
    eps=1e-8,
    momentum=0.1,
    n_groups=32,
    learn_affine=True,
    bias=True,
    training_mode=None,
    update_running_stats=True,
    center_mode="leave_one_out",
):
    """
    Mean centering only (no covariance whitening) for debugging.
    
    center_mode options:
    - "leave_one_out": Simple leave-one-out mean centering. No running mean during training.
                       At inference, uses the last computed mean (stored in running_mean).
    - "running_mean": Mix running_mean with leave-one-out mean (like current BW).
                      Updates running_mean during training.
    - "calibrated": Same as running_mean but assumes running_mean was set by calibration.
                    Does NOT update running_mean during training.
    """
    B, T, C = X.shape
    device = X.device
    dtype = X.dtype

    assert C % n_groups == 0, "C must be divisible by n_groups"
    group_size = C // n_groups

    if training_mode is None:
        training = torch.is_grad_enabled()
    else:
        training = bool(training_mode)

    # Initialize running_mean
    mean_shape = (1, n_groups, group_size)
    if running_mean is None:
        running_mean = torch.zeros(mean_shape, device=device, dtype=dtype)

    # Initialize affine parameters
    if learn_affine:
        if gamma is None:
            gamma = nn.Parameter(torch.ones(C, device=device, dtype=dtype))
        if beta is None and bias:
            beta = nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
    else:
        if gamma is None:
            gamma = torch.ones(C, device=device, dtype=dtype)
        if beta is None and bias:
            beta = torch.zeros(C, device=device, dtype=dtype)

    # Reshape into groups
    Xg = X.view(B, T, n_groups, group_size)

    if training and B > 1:
        # Calculate leave-one-out mean (averaged over T)
        sum_all_flat = Xg.sum(dim=(0, 1), keepdim=True)  # (1, 1, ng, gs)
        sum_per_seq = Xg.sum(dim=1, keepdim=True)  # (B, 1, ng, gs)
        count_other = (B - 1) * T
        mean_other = (sum_all_flat - sum_per_seq) / count_other  # (B, 1, ng, gs)

        if center_mode == "leave_one_out":
            # Option 1: Pure leave-one-out, no running mean mixing
            current_mean = mean_other
            # Update running_mean to store for inference
            with torch.no_grad():
                mean_all = Xg.mean(dim=(0, 1))  # (n_groups, gs)
                running_mean.copy_(mean_all.view(1, n_groups, group_size))
        
        elif center_mode == "running_mean":
            # Option 2: Mix running_mean with leave-one-out
            current_mean = (
                (1 - momentum) * running_mean.view(1, 1, n_groups, group_size)
                + momentum * mean_other
            )
            # Update running_mean
            if update_running_stats:
                with torch.no_grad():
                    mean_all = Xg.mean(dim=(0, 1))
                    running_mean.mul_(1 - momentum).add_(
                        momentum * mean_all.view(1, n_groups, group_size)
                    )
        
        elif center_mode == "calibrated":
            # Option 3: Mix calibrated running_mean with leave-one-out
            # running_mean was set by calibration, don't update it
            current_mean = (
                (1 - momentum) * running_mean.view(1, 1, n_groups, group_size)
                + momentum * mean_other
            )
            # Do NOT update running_mean (it's calibrated)
        
        else:
            raise ValueError(f"Unknown center_mode: {center_mode}")
    
    else:
        # Inference: use running_mean directly
        current_mean = running_mean.view(1, 1, n_groups, group_size)

    # Center the data
    Xc = Xg - current_mean

    # Restore shape (B, T, C)
    Y = Xc.reshape(B, T, C)

    # Optional affine transform
    if gamma is not None:
        Y = Y * gamma.view(1, 1, -1)
    if beta is not None and bias:
        Y = Y + beta.view(1, 1, -1)

    return Y, running_mean.detach(), gamma, beta


# The calibrated mode weights the current batch by `momentum`, which attenuates
# its gradient by the same factor, so 1/momentum exactly restores leave-one-out
# gradient strength (not applied to running_mean). The alignment is sharp: 2x that
# value already makes the gradient orthogonal to the leave-one-out direction,
# so keep the scale tied to momentum unless deliberately sweeping it.
STAT_GRAD_SCALE = None  # override; None -> 1/momentum


def boost_grad(t, scale):
    """Return `t` unchanged in the forward pass, with its gradient scaled."""
    return scale * t - (scale - 1.0) * t.detach()


class _RenormWhitenFn(torch.autograd.Function):
    """Forward: Y_R = W_R (x - mu_R). Backward: the gradient of
    Z = R L_B^-1 (x - mu_B) + d (R, d constant), written out by hand.

    Same value and gradient as Y_R + (Z - Z.detach()) in _renorm_batch_whiten,
    without its activation-sized forward product and elementwise passes; the
    Cholesky and inverse backward run on the small (G, G) matrices.
    """

    @staticmethod
    def forward(ctx, Xg, Xc, mu_R, W_R, L_B, W_B, cov_scale, fix_factor, shrink_target):
        R = W_R @ L_B
        ctx.save_for_backward(Xc, R, L_B, W_B)
        ctx.cov_scale, ctx.fix_factor, ctx.shrink_target = cov_scale, fix_factor, shrink_target
        return torch.einsum("gij,btgj->btgi", W_R[0], Xg - mu_R.unsqueeze(1))

    @staticmethod
    def backward(ctx, g):
        Xc, R, L, W = ctx.saved_tensors
        B, T, _, G = Xc.shape
        M = R @ W
        dXc = torch.einsum("btgi,gij->btgj", g, M[0])
        dM = torch.einsum("btgi,btgj->gij", g, Xc).unsqueeze(0)
        # M = R W, W = L^-1, L = chol(A): back to A (Murray 2016), then to cov.
        dL = -W.transpose(-1, -2) @ (R.transpose(-1, -2) @ dM) @ W.transpose(-1, -2)
        P = torch.tril(L.transpose(-1, -2) @ dL)
        P = P - 0.5 * torch.diag_embed(P.diagonal(dim1=-2, dim2=-1))
        dA = W.transpose(-1, -2) @ P @ W
        dA = 0.5 * (dA + dA.transpose(-1, -2))
        ff = ctx.fix_factor
        diag = torch.diag_embed(dA.diagonal(dim1=-2, dim2=-1))
        if ctx.shrink_target == "diag":
            dA = ff * dA + (1 - ff) * diag
        else:
            eye = torch.eye(G, device=dA.device, dtype=dA.dtype)
            dA = ff * dA + (1 - ff) * diag.sum((-2, -1), keepdim=True) / G * eye
        dcov = ctx.cov_scale * dA
        dXc = dXc + torch.einsum("btgc,gcd->btgd", Xc, (2.0 / (B * T)) * dcov[0])
        return dXc - dXc.mean(dim=(0, 1), keepdim=True), None, None, None, None, None, None, None, None


NS_COLD_ITERS = 15


def ns_whiten(A, iters, scale=True):
    """Coupled Newton-Schulz: (W, S) ~ (A^-1/2, A^1/2), symmetric (ZCA) whitening.

    A is scaled by its Frobenius norm so every eigenvalue lies in (0, 1]. An
    eigenvalue x of the scaled matrix gains about 1.5x per iteration on the way
    to x^-1/2, so a few iterations whiten the large directions and leave the
    small ones under-amplified. S = A W exactly at any iteration count.
    scale=False for A near I (eigenvalues must stay below 3).
    """
    G = A.size(-1)
    eye = torch.eye(G, device=A.device, dtype=A.dtype)
    nrm = torch.linalg.matrix_norm(A, keepdim=True) if scale else torch.ones_like(A[..., :1, :1])
    Y = A / nrm
    Z = 1.5 * eye - 0.5 * Y
    Y = Y @ Z
    for _ in range(iters - 1):
        Tm = 1.5 * eye - 0.5 * (Z @ Y)
        Y = Y @ Tm
        Z = Tm @ Z
    return Z / nrm.sqrt(), Y * nrm.sqrt()


def ns_inv_sqrt(A, iters):
    """ns_whiten(A, iters)[0] for one (G, G) matrix in fewer launches: T = 1.5I - 0.5 ZY
    in one addmm, and no Y update in the last iteration (only Z is returned)."""
    eye = _eye(A.size(-1), A.device, A.dtype)
    nrm = torch.linalg.matrix_norm(A)
    Y = A / nrm
    Z = torch.add(1.5 * eye, Y, alpha=-0.5)
    Y = Y @ Z
    eye15 = 1.5 * eye
    for k in range(iters - 1):
        Tm = torch.addmm(eye15, Z, Y, alpha=-0.5)
        if k < iters - 2:
            Y = Y @ Tm
        Z = Tm @ Z
    return Z * nrm.rsqrt()


def whiten_matrix(A, whiten="cholesky", ns_iters=5, init=None):
    """(W, W^-1) with W A W^T ~ I: (L^-1, L) for cholesky, (A^-1/2, A^1/2) for ns.

    ns_warm starts from a whitening matrix P of a nearby covariance, init = (P, c):
    W = (P A P^T)^-1/2 P, with P A P^T ~ I so few iterations converge. c, about
    the largest eigenvalue of P A P^T, keeps the iteration inside its
    convergence region (eigenvalues below 3), which early training, with the
    covariance changing fast, leaves otherwise. W whitens A but is symmetric
    (ZCA) only if P commutes with A; chained over steps it drifts by a rotation.
    """
    if whiten == "cholesky":
        L = torch.linalg.cholesky_ex(A)[0]  # no error check: no device synchronization
        eye = _eye(A.size(-1), A.device, A.dtype)
        return torch.linalg.solve_triangular(L, eye if L.dim() == 2 else eye.expand_as(L), upper=False), L
    if whiten in ("ns", "ns_chol"):
        return ns_whiten(A, ns_iters)
    if whiten == "ns_warm":
        P, c = init
        Z, _ = ns_whiten(P @ A @ P.transpose(-1, -2) / c, ns_iters, scale=False)
        W = Z @ P / c.sqrt()
        return W, A @ W.transpose(-1, -2)
    raise ValueError(f"Unknown whiten: {whiten}")


class _RenormWhitenAutoFn(torch.autograd.Function):
    """Same as _RenormWhitenFn for any whitening map W = f(shrink(cov_w)): the
    activation-sized part of the backward is written out, the (G, G) part
    through W, the shrinkage and f runs through autograd on the small matrices.
    """

    @staticmethod
    def forward(ctx, Xg, Xc, mu_R, W_R, R, M, cov_w, cov_scale, fix_factor, shrink_target, whiten, ns_iters,
                init):
        ctx.save_for_backward(Xc, R, M, cov_w)
        ctx.args = cov_scale, fix_factor, shrink_target, whiten, ns_iters
        ctx.init = init
        return torch.einsum("gij,btgj->btgi", W_R[0], Xg - mu_R.unsqueeze(1))

    @staticmethod
    def backward(ctx, g):
        Xc, R, M, cov_w = ctx.saved_tensors
        cov_scale, ff, target, whiten, ns_iters = ctx.args
        init = ctx.init
        B, T, _, G = Xc.shape
        dXc = torch.einsum("btgi,gij->btgj", g, M[0])
        dM = torch.einsum("btgi,btgj->gij", g, Xc).unsqueeze(0)
        with torch.enable_grad():
            A = cov_w.detach().requires_grad_()
            W = whiten_matrix(shrink_cov(A, ff, target), whiten, ns_iters, init)[0]
            dA, = torch.autograd.grad(W, A, R.transpose(-1, -2) @ dM)
        dcov = cov_scale * 0.5 * (dA + dA.transpose(-1, -2))
        dXc = dXc + torch.einsum("btgc,gcd->btgd", Xc, (2.0 / (B * T)) * dcov[0])
        return (dXc - dXc.mean(dim=(0, 1), keepdim=True),) + (None,) * 12


def _whiten_cov_grad(dM, R, L, W, cov_w, ff, target, whiten, ns_iters, init):
    """Gradient w.r.t. cov_w (symmetric) of <dM, R W(shrink(cov_w))> for one
    group of (G, G) matrices; L = W^-1 (the Cholesky factor for cholesky)."""
    dW = R.transpose(-1, -2) @ dM
    if whiten != "cholesky":
        with torch.enable_grad():
            A = cov_w.detach().requires_grad_()
            dA, = torch.autograd.grad(whiten_matrix(shrink_cov(A, ff, target), whiten, ns_iters, init)[0], A, dW)
        return 0.5 * (dA + dA.transpose(-1, -2))
    # W = L^-1, L = chol(A) (Murray 2016), then through the shrinkage.
    G = W.size(-1)
    Wt = W.transpose(-1, -2)
    P = torch.tril(L.transpose(-1, -2) @ (-Wt @ dW @ Wt))
    P = P - 0.5 * torch.diag_embed(P.diagonal(dim1=-2, dim2=-1))
    dA = Wt @ P @ W
    dA = 0.5 * (dA + dA.transpose(-1, -2))
    diag = torch.diag_embed(dA.diagonal(dim1=-2, dim2=-1))
    if target == "diag":
        return ff * dA + (1 - ff) * diag
    eye = torch.eye(G, device=dA.device, dtype=dA.dtype)
    return ff * dA + (1 - ff) * diag.sum((-2, -1), keepdim=True) / G * eye


class _RenormWhitenFast(torch.autograd.Function):
    """One group (G = C), tokens flattened to X (N, G); BW's gain and bias folded in.

    Forward: Y = gamma * W_R (x - mu_R) + beta = X Wg + bias, one matrix product.
    Backward: the gradient of gamma * (R W_B (x - mu_B)) + beta (R constant), as
    in _RenormWhitenFn, but centering, the token-mean subtraction and the
    gain/bias gradients are folded into the small matrices, so the activations
    see three matrix products (g^T X, g Mg, X K) and one reduction. R W_B = W_R,
    so the data term is g (gamma W_R) = g Mg.
    post_eps not None: the output is followed by a LayerNorm without gain or
    bias (eps post_eps), applied here with the native kernels.
    """

    @staticmethod
    def forward(ctx, X, gamma, beta, Wg, bias, Mg, mu_B, mu_R, W_R, R, L_B, W_B, cov_w, cov_scale, ff, target,
                whiten, ns_iters, init, event, post_eps=None, ns_job=None):
        ctx.args = cov_scale, ff, target, whiten, ns_iters, post_eps
        ctx.init, ctx.event, ctx.ns_job = init, event, ns_job
        Y = torch.addmm(bias, X, Wg)
        saved = (X, gamma, Mg, mu_B, mu_R, W_R, R, L_B, W_B, cov_w)
        if post_eps is None:
            ctx.save_for_backward(*saved)
            return Y
        Z, mean, rstd = torch.native_layer_norm(Y, [Y.size(1)], None, None, post_eps)
        ctx.save_for_backward(*saved, Y, mean, rstd)
        return Z

    @staticmethod
    def backward(ctx, g):
        X, gamma, Mg, mu_B, mu_R, W_R, R, L_B, W_B, cov_w = ctx.saved_tensors[:10]
        cov_scale, ff, target, whiten, ns_iters, post_eps = ctx.args
        N = X.size(0)
        job = ctx.ns_job
        if job is not None:  # next step's W_R, on the side stream, overlapping this backward
            main, side = torch.cuda.current_stream(), _side_stream(g.device)
            with torch.cuda.stream(side):
                W = ns_inv_sqrt(job["A"], job["iters"])
                job["event"].record(side)
            W.record_stream(main)
            job["W"] = W
        if post_eps is not None:
            Y, mean, rstd = ctx.saved_tensors[10:]
            g = torch.ops.aten.native_layer_norm_backward(g, Y, [Y.size(1)], mean, rstd, None, None,
                                                          [True, False, False])[0]
        gs = _colsum(g)
        GX = g.transpose(0, 1) @ X  # sum_t g_i x_j
        if ctx.event is not None:  # the side stream's matrices are first needed here
            torch.cuda.current_stream().wait_event(ctx.event)
        if whiten == "cholesky":
            dgamma, K, bias = _glue(_fast_bwd_small, g.is_cuda)(
                GX, gs, gamma, Mg, mu_R, mu_B, W_R, R, W_B, N, float(cov_scale), float(ff), target == "diag")
        else:
            dgamma = (W_R * (GX - gs[:, None] * mu_R[None, :])).sum(1)
            dM = gamma[:, None] * (GX - gs[:, None] * mu_B[None, :])
            K = (2.0 / N) * cov_scale * _whiten_cov_grad(dM, R, L_B, W_B, cov_w, ff, target, whiten, ns_iters,
                                                         ctx.init)
            bias = -(gs / N) @ Mg - mu_B @ K
            K = _colmajor(K)
        dX = torch.addmm(bias, g, Mg)
        dX.addmm_(X, K)
        return (dX, dgamma, gs) + (None,) * 19


_SIDE_STREAMS = {}


def _side_stream(device):
    if device not in _SIDE_STREAMS:
        # High priority: its small, latency-bound kernels gate the backward pass
        # and would otherwise queue behind the large activation kernels.
        _SIDE_STREAMS[device] = torch.cuda.Stream(device, priority=-1)
    return _SIDE_STREAMS[device]


def _fast_stats(S, s, cov_R, mu_R, N: int, eps: float, m: float, loo: bool):
    """Batch mean and covariance from X^T X and the column sums, the whitening
    covariance, and the new buffer values."""
    mu_B = s / N
    eye = torch.eye(S.size(0), device=S.device, dtype=S.dtype)
    cov = S / N - torch.outer(mu_B, mu_B) + eps * eye
    if loo:
        return mu_B, cov, cov, mu_B
    return mu_B, cov, (1 - m) * cov_R + m * cov, (1 - m) * mu_R + m * mu_B


def _colmajor(A):
    """A with column-major strides: as the right operand of an (N, G) x (G, G)
    product, cuBLAS picks a faster kernel for it on the H100 (~30 vs ~43 us)."""
    return A.transpose(0, 1).contiguous().transpose(0, 1)


_ONES = {}


def _colsum(A):
    """A.sum(0) for A (N, G) as a matrix-vector product (no tensor cores, so
    full fp32): about twice as fast as the reduction kernel on the H100."""
    key = (A.size(0), A.device, A.dtype)
    if key not in _ONES:
        _ONES[key] = torch.ones(A.size(0), device=A.device, dtype=A.dtype)
    return torch.mv(A.transpose(0, 1), _ONES[key])


def _fast_bwd_small(GX, gs, gamma, Mg, mu_R, mu_B, W_R, R, W, N: int, cov_scale: float, ff: float,
                    diag_target: bool):
    """Cholesky case of _RenormWhitenFast.backward on the (G, G) matrices.

    With W = L^-1 and R = W_R L: the gradient at L, L^T (-W^T R^T dM W^T), is
    -R^T dM W^T (W^T R^T = W_R^T, L^T W_R^T = R^T), two products instead of four.
    """
    dgamma = (W_R * (GX - gs[:, None] * mu_R[None, :])).sum(1)
    dM = gamma[:, None] * (GX - gs[:, None] * mu_B[None, :])
    G = W.size(-1)
    P = torch.tril(-(R.transpose(0, 1) @ dM) @ W.transpose(0, 1))
    P = P - 0.5 * torch.diag_embed(P.diagonal())
    dA = W.transpose(0, 1) @ P @ W
    dA = 0.5 * (dA + dA.transpose(0, 1))
    diag = torch.diag_embed(dA.diagonal())
    if diag_target:
        dA = ff * dA + (1 - ff) * diag
    else:
        dA = ff * dA + (1 - ff) * diag.sum() / G * torch.eye(G, device=dA.device, dtype=dA.dtype)
    K = (2.0 / N) * cov_scale * dA
    # K is symmetric, so its transpose is the column-major copy
    return dgamma, K.transpose(0, 1), -(gs / N) @ Mg - mu_B @ K


_COMPILED = {}


def _glue(fn, cuda):
    """fn compiled (its elementwise ops fused into a few kernels) on CUDA."""
    if not cuda or not COMPILE_GLUE:
        return fn
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=False)
    return _COMPILED[fn]


COMPILE_GLUE = True


def renorm_fast_whiten(X, gamma, beta, running_mean, running_cov, eps, momentum, fix_factor, center_mode,
                       update_running_stats, update_running_cov_stats, cache, shrink_target,
                       whiten="cholesky", ns_iters=5, ns_refresh=0, side_stream=False, post_eps=None):
    """Training forward of BW with renorm, shared statistics, one group, the fused
    backward and no warmup or boost: same value and gradient as
    batch_orthonorm(...) * gamma + beta, with fewer passes over the activations.
    X: (B, T, C). Returns Y (B, T, C) and updates the buffers and cache in place.
    post_eps not None: Y is followed by F.layer_norm(Y, (C,), eps=post_eps), done here.
    """
    B, T, G = X.shape
    N = B * T
    X2 = X.reshape(N, G)
    cuda = X.is_cuda
    loo = center_mode == "leave_one_out"
    cov_R = running_cov.detach().reshape(G, G)
    main = torch.cuda.current_stream() if cuda else None
    side = _side_stream(X.device) if side_stream and cuda else None
    with torch.no_grad():
        if cache is not None and "event" in cache:  # last step's buffer updates and W_B
            main.wait_event(cache["event"])
        if cache is not None and "ns_job" in cache:  # ns_chol: W_B of last step, from its backward
            job = cache.pop("ns_job")
            if job["W"] is None:  # no backward ran
                job["W"] = ns_inv_sqrt(job["A"], job["iters"])
            else:
                main.wait_event(job["event"])
            cache["W"] = job["W"]
        mu_R = running_mean.detach().reshape(G).clone()  # running_mean is overwritten below
        key = (running_cov.data_ptr(), running_cov._version)
        if cache is not None and cache.get("key") == key:
            W_R = cache["W"].reshape(G, G)
        elif whiten == "ns_warm":
            W_R = ns_whiten(shrink_cov(cov_R, fix_factor, shrink_target), NS_COLD_ITERS)[0]
        else:
            W_R = whiten_matrix(shrink_cov(cov_R, fix_factor, shrink_target), whiten, ns_iters)[0]
        # The forward value needs only W_R: Y = X Wg + bias. Wg (column-major,
        # the faster layout for cuBLAS) and Mg are views of one matrix.
        Mg_rm = gamma.detach()[:, None] * W_R
        Wg, Mg = Mg_rm.transpose(0, 1), _colmajor(Mg_rm)
        bias = torch.addmv(beta.detach(), Mg_rm, mu_R, alpha=-1)
        step = cache.get("step", 0) if cache is not None else 0
        # The batch statistics, the buffer updates and the batch whitening
        # matrix are first needed in the backward pass (and next step), so they
        # run on a side stream while the forward pass continues.
        if side is not None:
            side.wait_stream(main)
            for t in (X2, mu_R, W_R):
                t.record_stream(side)
        with torch.cuda.stream(side) if side is not None else contextlib.nullcontext():
            mu_B, cov, cov_w, mu_new = _glue(_fast_stats, cuda)(
                X2.transpose(0, 1) @ X2, _colsum(X2), cov_R, mu_R, N, float(eps), float(momentum), loo)
            cov_scale = 1.0 if loo else momentum
            A_B = shrink_cov(cov_w, fix_factor, shrink_target)
            spec, v = (whiten, ns_iters, None), None
            if whiten == "ns_warm":
                if ns_refresh and step % ns_refresh == 0:
                    spec = ("ns", NS_COLD_ITERS, None)
                else:
                    Ap = W_R @ A_B @ W_R.transpose(0, 1)
                    v = cache.get("v") if cache is not None else None
                    v = torch.ones_like(Ap[:, :1]) if v is None else v.reshape(G, 1)
                    for _ in range(2):
                        v = Ap @ v
                        v = v / torch.linalg.vector_norm(v)
                    c = 1.25 * (v.transpose(0, 1) @ Ap @ v)
                    spec = (whiten, ns_iters, (W_R, c))
            if whiten == "ns_chol":
                spec = ("cholesky", ns_iters, None)
            W_B, L_B = whiten_matrix(A_B, *spec)
            R = W_R @ L_B
            if loo or update_running_stats:
                running_mean.copy_(mu_new.view_as(running_mean))
            if loo or update_running_cov_stats:
                running_cov.copy_(cov_w.view_as(running_cov))
            # ns_chol: the forward (next step's W_R) whitens symmetrically,
            # W_next = A_B^-1/2 = Q L_B^-1 with Q orthogonal; the backward is the
            # Cholesky one (R = W_R L_B), i.e. the ZCA gradient with Q held constant.
            # With a side stream W_next is computed in the backward (ns_job).
            W_next, ns_job = W_B, None
            if whiten == "ns_chol" and (loo or update_running_cov_stats):
                if side is not None and cache is not None:
                    W_next, ns_job = None, dict(A=A_B, iters=ns_iters, W=None, event=torch.cuda.Event())
                else:
                    W_next = ns_inv_sqrt(A_B, ns_iters)
        event = None
        if side is not None:
            event = torch.cuda.Event()
            event.record(side)
            for t in (mu_B, cov_w, W_B, L_B, R, A_B) + ((v, spec[2][1]) if v is not None else ()):
                t.record_stream(main)
            if W_next is not None:
                W_next.record_stream(main)
            if cache is None:  # nothing would make the next reader of the buffers wait
                main.wait_event(event)
    Y = _RenormWhitenFast.apply(X2, gamma, beta, Wg, bias, Mg, mu_B, mu_R, W_R, R, L_B, W_B, cov_w, cov_scale,
                                fix_factor, shrink_target, *spec, event, post_eps, ns_job)
    if cache is not None:
        new_key = (running_cov.data_ptr(), running_cov._version)
        if loo or update_running_cov_stats:
            if ns_job is not None:
                cache.pop("W", None)
                cache.update(key=new_key, ns_job=ns_job)
            else:
                cache.update(key=new_key, W=W_next.view_as(running_cov))
            if event is not None:
                cache["event"] = event
            else:
                cache.pop("event", None)
        elif new_key == key:
            cache.update(key=key, W=W_R.view_as(running_cov))
        else:
            cache.clear()
        if whiten == "ns_warm":
            cache["step"] = step + 1
            if v is not None:
                cache["v"] = v
    return Y.view(B, T, G)


def _renorm_batch_whiten(Xg, running_mean, running_cov, eps, momentum, fix_factor,
                         cov_warmup, center_mode, stat_grad_scale, warmup_ema_buffer,
                         update_running_stats, update_running_cov_stats,
                         stat_grad="full", cache=None, shrink_target="diag", fused=False,
                         whiten="cholesky", ns_iters=5, ns_refresh=0):
    """renorm with whole-batch gradient statistics (renorm_stats="batch").

    The forward value is the whitening with the stored buffers, so the current
    batch only shapes the gradient; nothing there needs to exclude a sequence's
    own tokens. One mean and covariance per group replace the B leave-one-out
    ones, and the whitening is applied as a matrix product instead of B
    triangular solves. calibrated takes the running_mean branch; its buffers
    hold the calibration statistics and are not updated here (the update flags
    are off outside calibration).

    stat_grad: which current-batch statistics carry gradient. "full" (default)
    the mean and covariance; "mean" the mean only; "none" neither, so the
    layer backpropagates as the fixed linear map W_R (ablations).

    cache: a per-layer dict holding the whitening matrix of the buffers, keyed
    by the buffer's storage and version. When the buffer update below
    reproduces cov_w bit for bit (running_mean and leave_one_out without
    boost), next step's W_R is this step's W_B, so its Cholesky and inverse are
    skipped.

    whiten: "cholesky" (W = L^-1) or "ns" (W ~ A^-1/2 from ns_iters coupled
    Newton-Schulz steps, matrix products only). "ns_warm" starts the iterations
    from W_R; every ns_refresh-th step (0: never) it restarts cold with
    NS_COLD_ITERS iterations, which resets the drift toward ZCA.
    """
    B, T, n_groups, G = Xg.shape
    eye = torch.eye(G, device=Xg.device, dtype=Xg.dtype).view(1, 1, G, G)
    mu_R = running_mean.detach().clone()  # (1, ng, G)
    cov_R = running_cov.detach().clone()  # (1, ng, G, G)

    fused = fused and stat_grad == "full" and not cov_warmup
    with torch.no_grad() if fused else contextlib.nullcontext():
        mu_B = Xg.mean(dim=(0, 1)).unsqueeze(0)  # (1, ng, G)
        Xc = Xg - (mu_B.detach() if stat_grad == "none" else mu_B).unsqueeze(1)
        cov = torch.einsum("btgc,btgd->gcd", Xc, Xc).unsqueeze(0) / (B * T) + eps * eye

    if center_mode == "leave_one_out":
        cov_w = cov if stat_grad_scale == 1.0 else boost_grad(cov, stat_grad_scale)
    elif cov_warmup:
        cov_w = (1 - momentum) * eye * cov + momentum * cov
    else:
        cov_w = boost_grad((1 - momentum) * cov_R + momentum * cov, stat_grad_scale)
    if stat_grad != "full":
        cov_w = cov_w.detach()

    key = (running_cov.data_ptr(), running_cov._version)
    with torch.no_grad():
        if cache is not None and cache.get("key") == key:
            if "event" in cache:
                torch.cuda.current_stream().wait_event(cache["event"])
            W_R = cache["W"]
        elif whiten == "ns_warm":
            W_R = ns_whiten(shrink_cov(cov_R, fix_factor, shrink_target), NS_COLD_ITERS)[0]
        else:
            W_R = whiten_matrix(shrink_cov(cov_R, fix_factor, shrink_target), whiten, ns_iters)[0]
    spec = (whiten, ns_iters, None)
    A_B = shrink_cov(cov_w, fix_factor, shrink_target)
    if whiten == "ns_warm":
        step = cache.get("step", 0) if cache is not None else 0
        if ns_refresh and step % ns_refresh == 0:
            spec = ("ns", NS_COLD_ITERS, None)
        else:
            with torch.no_grad():
                Ap = W_R @ A_B @ W_R.transpose(-1, -2)
                v = cache.get("v") if cache is not None else None
                v = torch.ones_like(Ap[..., :1]) if v is None else v
                for _ in range(2):
                    v = Ap @ v
                    v = v / torch.linalg.vector_norm(v, dim=-2, keepdim=True)
                c = 1.25 * (v.transpose(-1, -2) @ Ap @ v)
            spec = (whiten, ns_iters, (W_R, c))
    if whiten == "ns_chol":  # Cholesky gradient; the next step's W_R is A_B^-1/2
        spec = ("cholesky", ns_iters, None)
    W_B, L_B = whiten_matrix(A_B, *spec)
    if fused and spec[0] != "cholesky":
        cov_scale = stat_grad_scale * (1.0 if center_mode == "leave_one_out" else momentum)
        R = W_R @ L_B
        Y = _RenormWhitenAutoFn.apply(Xg, Xc, mu_R, W_R, R, R @ W_B, cov_w.detach(), cov_scale,
                                      fix_factor, shrink_target, *spec)
    elif fused:
        cov_scale = stat_grad_scale * (1.0 if center_mode == "leave_one_out" else momentum)
        Y = _RenormWhitenFn.apply(Xg, Xc, mu_R, W_R, L_B, W_B, cov_scale, fix_factor, shrink_target)
    else:
        with torch.no_grad():
            R = W_R @ L_B
            d = torch.einsum("xgij,xgj->xgi", W_R, mu_B - mu_R)
            Y_R = torch.einsum("gij,btgj->btgi", W_R[0], Xg - mu_R.unsqueeze(1))
        M = R @ W_B  # value W_R, gradient through the batch statistics
        Z = torch.einsum("gij,btgj->btgi", M[0], Xc) + d.unsqueeze(1)
        Y = Y_R + (Z - Z.detach())

    with torch.no_grad():
        buffer_is_cov_w = stat_grad_scale == 1.0 and not cov_warmup
        if center_mode == "leave_one_out":
            running_mean.copy_(mu_B)
            running_cov.copy_(cov)
        else:
            if update_running_stats:
                running_mean.mul_(1 - momentum).add_(momentum * mu_B)
            if update_running_cov_stats:
                if cov_warmup and not warmup_ema_buffer:
                    running_cov.copy_((1 - momentum) * eye * cov + momentum * cov)
                else:
                    running_cov.mul_(1 - momentum).add_(momentum * cov)
            buffer_is_cov_w = buffer_is_cov_w and update_running_cov_stats
        if cache is not None:
            new_key = (running_cov.data_ptr(), running_cov._version)
            if buffer_is_cov_w:
                cache.update(key=new_key, W=ns_whiten(A_B.detach(), ns_iters)[0] if whiten == "ns_chol" else W_B.detach())
                cache.pop("event", None)
            elif new_key == key:
                cache.update(key=key, W=W_R)
            else:
                cache.clear()
            if whiten == "ns_warm":
                cache["step"] = step + 1
                if spec[0] == "ns_warm":
                    cache["v"] = v
    return Y.reshape(B, T, n_groups * G)


def batch_orthonorm(
    X,
    gamma=None,
    beta=None,
    running_mean=None,
    running_cov=None,
    eps=1e-8,
    momentum=0.1,
    n_groups=32,
    cov_warmup=False,
    fix_factor=0.9,
    learn_affine=True,
    bias=True,
    training_mode=None,
    update_running_stats=True,
    update_running_cov_stats=True,
    center_mode="running_mean",
    warmup_ema_buffer=False,
    inference_ema=False,
    grad_scale=None,
    calib_center="loo",
    renorm=False,
    renorm_stats="loo",
    stat_grad="full",
    renorm_cache=None,
    shrink_target="diag",
    renorm_fused=False,
    whiten="cholesky",
    ns_iters=5,
    ns_refresh=0,
):
    """
    Fully vectorized block whitening for X of shape (B, T, C).

    renorm: causal training forward (Batch Renormalization, Ioffe 2017, applied
        to whitening). Any current-batch statistic in the forward pass breaks
        causality once two BW layers are stacked, even leave-one-out: layer k
        writes sequence b's future tokens into sequence s's activations, and
        layer k+1 reads them back into b's statistics. With renorm the forward
        value is exactly the whitening with the buffers from previous
        iterations (what inference computes), while the gradient still flows
        through the current-batch mean and covariance, as
        R L_B^-1 (x - mu_B) + d with R = L_R^-1 L_B and d = L_R^-1 (mu_B - mu_R)
        held constant.
    renorm_stats (renorm only): "loo" takes the gradient statistics per
        sequence from the other sequences, as without renorm; "batch" uses one
        whole-batch mean and covariance, which renorm keeps causal because they
        never reach the forward value. "batch" is much cheaper.

    calib_center (calibrated only): "loo" (default) centers data and
        covariance with the leave-one-out mean, keeping only the calibrated
        covariance as history. "blend" centers both with
        (1-m)*calibrated + m*leave-one-out mean; the mostly detached centering
        mean breaks scale invariance, as in running_mean before cdcba46, and the
        BW input RMS runs away (150-21,000 by step 5000 at any boost). Kept for
        comparison only. "blend_loo_cov" centers the data with the blended mean
        but the covariance about the leave-one-out mean (the pre-cdcba46
        behaviour); stable, but worse than "loo".

    grad_scale: gradient boost on the current-batch statistics (forward pass
        unchanged). None keeps the per-mode default (1/momentum for calibrated,
        1 otherwise). An explicit value applies to every mode: the mean and
        covariance in calibrated, the covariance in running_mean and
        leave_one_out (their centering mean already carries the full gradient).

    warmup_ema_buffer: during cov_warmup, still EMA-accumulate the full
        covariance into running_cov (only the whitening uses the diagonal
        blend), so the history is complete when warmup ends.
    inference_ema: in leave_one_out mode, keep EMA buffers for inference
        instead of copying the last training batch. Training is unchanged.
    No loops. Running stats updated correctly. Whitening is correct.
    """
    B, T, C = X.shape
    device = X.device
    dtype = X.dtype

    if not getattr(batch_orthonorm, "_printed_group_size", False):
        print(f"n_groups: {n_groups}, group_size: {C // n_groups}")
        batch_orthonorm._printed_group_size = True

    assert C % n_groups == 0, "C must be divisible by n_groups"
    group_size = C // n_groups

    if training_mode is None:
        training = torch.is_grad_enabled()
    else:
        training = bool(training_mode)

    # Only calibrated recomputes its history term at the current weights, so
    # only there does the history really co-vary with the current batch and the
    # boost restore a real derivative. running_mean's history is a stale EMA
    # that does not respond to this step; boosting it overstates the gradient
    # 1/momentum-fold and makes the activations run away.
    if calib_center not in ("blend", "blend_loo_cov", "loo"):
        raise ValueError(f"Unknown calib_center: {calib_center}")
    if grad_scale is not None:
        stat_grad_scale = float(grad_scale)
    elif center_mode == "calibrated":
        stat_grad_scale = (
            STAT_GRAD_SCALE if STAT_GRAD_SCALE is not None else 1.0 / momentum
        )
    else:
        stat_grad_scale = 1.0

    # Verification-only oracle: whiten with the whole-batch mean+covariance
    # computed over ALL B*T tokens (including self). This breaks causality by
    # construction and is NOT meant for training; it exists so that tests can
    # confirm the whitening produces an exactly-identity output covariance
    # (with fix_factor=1.0 and no grouping).
    full_mode = (center_mode == "full")

    # --------------------------------------
    # 1. Initialize running stats correctly
    # --------------------------------------
    mean_shape = (1, n_groups, group_size)
    cov_shape = (B, n_groups, group_size, group_size)

    if running_mean is None:
        running_mean = torch.zeros(mean_shape, device=device, dtype=dtype)

    if running_cov is None:
        running_cov = (
            torch.eye(group_size, device=device, dtype=dtype)
            .view(1, 1, group_size, group_size)
            .repeat(1, n_groups, 1, 1)
        )
    renorm = renorm and training and not full_mode
    if renorm_stats not in ("loo", "batch"):
        raise ValueError(f"Unknown renorm_stats: {renorm_stats}")
    if stat_grad not in ("full", "mean", "none"):
        raise ValueError(f"Unknown stat_grad: {stat_grad}")
    if stat_grad != "full" and training and not (renorm and renorm_stats == "batch"):
        raise ValueError("stat_grad other than 'full' needs renorm=True, renorm_stats='batch'")
    if shrink_target not in ("diag", "identity"):
        raise ValueError(f"Unknown shrink_target: {shrink_target}")
    if whiten not in ("cholesky", "ns", "ns_warm", "ns_chol"):
        raise ValueError(f"Unknown whiten: {whiten}")
    if whiten != "cholesky" and training and not (renorm and renorm_stats == "batch"):
        raise ValueError("whiten other than 'cholesky' needs renorm=True, renorm_stats='batch'")
    if renorm and renorm_stats == "batch":
        if center_mode not in ("leave_one_out", "running_mean", "calibrated"):
            raise ValueError("renorm_stats='batch' supports leave_one_out, running_mean and calibrated")
        Y = _renorm_batch_whiten(
            X.view(B, T, n_groups, group_size), running_mean, running_cov, eps,
            momentum, fix_factor, cov_warmup, center_mode, stat_grad_scale,
            warmup_ema_buffer, update_running_stats, update_running_cov_stats,
            stat_grad=stat_grad, cache=renorm_cache, shrink_target=shrink_target, fused=renorm_fused,
            whiten=whiten, ns_iters=ns_iters, ns_refresh=ns_refresh)
        if gamma is not None:
            Y = Y * gamma.view(1, 1, -1)
        if beta is not None and bias:
            Y = Y + beta.view(1, 1, -1)
        return Y, running_mean.detach(), running_cov.detach(), gamma, beta
    if renorm:
        # Buffers as they were before this batch updates them in place.
        prev_mean = running_mean.detach().clone().view(1, 1, n_groups, group_size)
        prev_cov = running_cov.detach().clone()
    if training and not full_mode:
        running_cov = running_cov.expand(cov_shape).clone()

    # --------------------------------------
    # 1b. Initialize affine parameters if requested
    # --------------------------------------
    if learn_affine:
        if gamma is None:
            gamma = nn.Parameter(torch.ones(C, device=device, dtype=dtype))
        if beta is None and bias:
            beta = nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
    else:
        if gamma is None:
            gamma = torch.ones(C, device=device, dtype=dtype)
        if beta is None and bias:
            beta = torch.zeros(C, device=device, dtype=dtype)

    # --------------------------------------
    # 2. Reshape into groups
    # --------------------------------------
    Xg = X.view(B, T, n_groups, group_size)

    # --------------------------------------
    # Center using running_mean or mixed mean
    # --------------------------------------
    if full_mode:
        # Whole-batch mean over all B*T tokens (includes self).
        current_mean = Xg.mean(dim=(0, 1), keepdim=True)  # (1, 1, ng, gs)
    elif training and B > 1:
        # Calculate leave-one-out mean (averaged over T)
        sum_all_flat = Xg.sum(dim=(0, 1), keepdim=True)  # (1, 1, ng, gs)
        sum_per_seq = Xg.sum(dim=1, keepdim=True)  # (B, 1, ng, gs)
        count_other = (B - 1) * T
        mean_other = (sum_all_flat - sum_per_seq) / count_other

        if center_mode in ("leave_one_out", "running_mean") or (
                center_mode == "calibrated" and calib_center == "loo"):
            # running_mean also centers with the full-gradient leave-one-out
            # mean; only its covariance is pooled over iterations. A detached
            # (blended) centering mean makes the output c*L^-1 x - L^-1 m under
            # x -> c*x, and the fixed L^-1 m term survives the LayerNorm that
            # follows, so the backward pass is not scale-invariant and the
            # residual stream feeding this layer grows without bound. The
            # detached covariance alone is harmless: it only rescales, which the
            # LayerNorm cancels.
            current_mean = mean_other
            center_for_cov = mean_other
        elif center_mode == "calibrated":
            current_mean = (
                1 - momentum
            ) * running_mean.view(
                1, 1, n_groups, group_size
            ) + momentum * mean_other
            # calib_center="blend" (not the default): the covariance is taken
            # about the same mean that centers the data. Although calibrated
            # overwrites running_cov instead of accumulating an EMA, this runs
            # away in full training at every boost level; use "loo".
            # Un-boosted on purpose: boost_grad is forward-identity, so reusing
            # the boosted current_mean would leave the forward pass unchanged
            # while routing the covariance gradient through the mean's gradient
            # scaling as well.
            center_for_cov = mean_other if calib_center == "blend_loo_cov" else current_mean
            if not cov_warmup:
                current_mean = boost_grad(current_mean, stat_grad_scale)
        else:
            raise ValueError(f"Unknown center_mode: {center_mode}")
    else:
        current_mean = running_mean.view(1, 1, n_groups, group_size)
        center_for_cov = current_mean

    Xc_self = Xg - current_mean

    if full_mode:
        # Whole-batch covariance over all B*T tokens (single shared matrix).
        # Whitening with this in-sample covariance yields an exactly-identity
        # output covariance (up to eps and fix_factor).
        N = B * T
        if N <= 1:
            raise ValueError("center_mode='full' requires B*T > 1")
        Xc_flat = Xc_self.reshape(N, n_groups, group_size)
        cov_full = torch.einsum("ngc,ngd->gcd", Xc_flat, Xc_flat) / (N - 1)
        cov_full = cov_full.unsqueeze(0)  # (1, n_groups, gs, gs)
        cov_full = cov_full + eps * torch.eye(
            group_size, device=device, dtype=dtype
        ).view(1, 1, group_size, group_size)

    # Scale to correct for variance bias from leave-one-out component.
    # With mixed mean (1-m)*running + m*mean_other, only the momentum fraction
    # contributes to the bias. When momentum=1, this gives sqrt((B-1)/B).
    '''if training and B > 1:
        scale = (1 - momentum**2 / B) ** 0.5
        Xc_self = Xc_self * scale'''

    if training and not full_mode:
        if B <= 1:
            raise ValueError(
                "Batch size must be greater than 1 during training"
            )
        # --------------------------------------
        # 3. Compute leave-one-out covariance (other sequences only)
        # --------------------------------------
        # Center about the same mean that centers Xc_self above. The second
        # moment of the centered data is Sigma + dd^T with d the centering
        # error, so a covariance built about a *different* mean omits that
        # rank-1 term and under-whitens along d. Identical to mean_other in
        # leave_one_out mode; the term only bites when the running mean is
        # stale. A per-token leave-one-out mean instead shrinks the covariance
        # and over-scales the whitened output.
        # Built from per-sequence Gram matrices instead of a (B, B, T, C)
        # tensor: for sequence b with center c_b and the other sequences' sum
        # S_b and Gram G_b, sum_{s!=b} (x - c_b)(x - c_b)^T
        # = G_b - c_b S_b^T - S_b c_b^T + N c_b c_b^T. Shifting by the batch
        # mean first (exact, the expression is shift-invariant) keeps the
        # subtraction well conditioned in fp32.
        sample_count = (B - 1) * T
        ref = Xg.mean(dim=(0, 1), keepdim=True)
        Xs = Xg - ref  # (B, T, ng, gs)
        c = (center_for_cov - ref).expand(B, 1, n_groups, group_size).squeeze(1)  # (B, ng, gs)
        G_seq = torch.einsum("s t g c, s t g d -> s g c d", Xs, Xs)
        G_oth = G_seq.sum(dim=0, keepdim=True) - G_seq
        S_seq = Xs.sum(dim=1)
        S_oth = S_seq.sum(dim=0, keepdim=True) - S_seq
        cross = c.unsqueeze(-1) * S_oth.unsqueeze(-2)
        cov = (
            G_oth - cross - cross.transpose(-1, -2)
            + sample_count * c.unsqueeze(-1) * c.unsqueeze(-2)
        ) / sample_count
        cov = cov + eps * torch.eye(
            group_size,
            device=device,
            dtype=dtype,
        ).view(1, 1, group_size, group_size)

    # --------------------------------------
    # 6. Build the whitening covariance (with gradient) and update buffers
    # --------------------------------------
    # The whitening covariance mirrors the mean: a *constant* history term
    # (running_cov from prior iterations, or the frozen calibrated cov) plus the
    # *current batch* covariance, which carries the gradient. Reading the
    # detached running_cov buffer directly (as before) let no gradient flow
    # through the covariance/whitening transform at all.
    if training and not full_mode:
        # History from prior iters / calibration: constant (no gradient).
        cov_history = running_cov.detach().clone()

        if center_mode == "leave_one_out":
            # Full current-batch covariance, coefficient 1.0 (full gradient).
            cov_for_whitening = cov if stat_grad_scale == 1.0 else boost_grad(cov, stat_grad_scale)
        elif cov_warmup:
            # Warmup: diagonally-dominant blend of the current-batch covariance.
            eye4 = torch.eye(group_size, device=device, dtype=dtype).view(
                1, 1, group_size, group_size
            )
            x_var = eye4 * cov
            cov_for_whitening = (1 - momentum) * x_var + momentum * cov
        else:
            # running_mean / calibrated: constant history + current batch (grad),
            # current batch weighted by momentum, identical to the mean mix.
            cov_for_whitening = boost_grad(
                (1 - momentum) * cov_history + momentum * cov, stat_grad_scale
            )

        # Update running buffers (no gradient) so they act as the constant
        # history next iter and as the stats used at inference time.
        with torch.no_grad():
            # Symmetric with the covariance buffer: accumulate the *leave-one-out*
            # mean averaged over the batch (mirroring how running_cov stores the
            # batch-averaged leave-one-out covariance). Averaging the per-sequence
            # leave-one-out means over b is provably identical to the full mean over
            # all B*T tokens -- the excluded sequence returns in the average -- so
            # this is numerically the same as the previous Xg.mean(dim=(0,1)) form,
            # just written to make the mean/cov handling parallel.
            mean_loo = mean_other.mean(dim=(0, 1))  # (n_groups, G)
            # Only update running_mean if enabled (disabled when using calibration)
            copy_last_batch = center_mode == "leave_one_out" and not inference_ema
            if copy_last_batch:
                running_mean.copy_(mean_loo.view(1, n_groups, group_size))
            elif update_running_stats:
                running_mean.mul_(1 - momentum).add_(
                    momentum * mean_loo.view(1, n_groups, group_size)
                )

            # Only update running_cov if enabled (disabled when using calibration).
            if copy_last_batch:
                running_cov.copy_(cov)
            elif update_running_cov_stats:
                if (cov_warmup and not warmup_ema_buffer
                        and center_mode != "leave_one_out"):
                    diag = torch.eye(group_size, device=device, dtype=dtype)
                    x_var = diag.view(1, 1, group_size, group_size) * cov
                    running_cov.copy_(
                        (1 - momentum) * x_var + momentum * cov
                    )

                else:
                    running_cov.mul_(1 - momentum).add_(momentum * cov)

        cov_for_fix = cov_for_whitening
    else:
        # Inference (and the 'full' oracle, handled via cov_full) use the buffer.
        cov_for_fix = running_cov

    uncorrelated_running_cov = shrink_cov(cov_full if full_mode else cov_for_fix, fix_factor, shrink_target)

    # --------------------------------------
    # 7. Cholesky whitening (batched)
    # --------------------------------------
    Xc_perm = Xc_self.permute(0, 2, 3, 1)  # -> (B, n_groups, G, T)
    if whiten != "cholesky":
        if (whiten == "ns_warm" and renorm_cache is not None
                and renorm_cache.get("key") == (running_cov.data_ptr(), running_cov._version)):
            if "event" in renorm_cache:
                torch.cuda.current_stream().wait_event(renorm_cache["event"])
            W = renorm_cache["W"]
        else:
            W = ns_whiten(uncorrelated_running_cov, NS_COLD_ITERS if whiten == "ns_warm" else ns_iters)[0]
        Y = (W @ Xc_perm).permute(0, 3, 1, 2).reshape(B, T, C)
        if gamma is not None:
            Y = Y * gamma.view(1, 1, -1)
        if beta is not None and bias:
            Y = Y + beta.view(1, 1, -1)
        return Y, running_mean.detach(), running_cov.detach(), gamma, beta
    L = torch.linalg.cholesky(uncorrelated_running_cov)  # (B or 1, n_groups, G, G)
    # NOTE: torch.linalg.solve_triangular on the MPS backend miscomputes the
    # result when the triangular factor has a leading batch dim of 1 that must
    # broadcast against B (the inference path uses a shared (1, n_groups, G, G)
    # running_cov). Explicitly expanding L to the batch size avoids the broadcast
    # and is a no-op on CPU/CUDA.
    if L.shape[0] == 1 and Xc_perm.shape[0] > 1:
        L = L.expand(Xc_perm.shape[0], -1, -1, -1).contiguous()
    Y_perm = torch.linalg.solve_triangular(L, Xc_perm, upper=False)

    if renorm:
        L_R = torch.linalg.cholesky(shrink_cov(prev_cov, fix_factor, shrink_target))
        L_R = L_R.expand(B, -1, -1, -1).contiguous()
        with torch.no_grad():
            R = torch.linalg.solve_triangular(L_R, L, upper=False)
            d = torch.linalg.solve_triangular(
                L_R, (current_mean - prev_mean).permute(0, 2, 3, 1), upper=False)
            Y_R = torch.linalg.solve_triangular(
                L_R, (Xg - prev_mean).permute(0, 2, 3, 1), upper=False)
        Z = R @ Y_perm + d
        # Forward value exactly Y_R, so no current-batch statistic enters it,
        # not even as rounding error; the gradient is that of Z.
        Y_perm = Y_R + (Z - Z.detach())

    # Restore shape (B, T, C)
    Y = Y_perm.permute(0, 3, 1, 2).reshape(B, T, C)

    # --------------------------------------
    # 8. Optional affine transform
    # --------------------------------------
    if gamma is not None:
        Y = Y * gamma.view(1, 1, -1)
    if beta is not None and bias:
        Y = Y + beta.view(1, 1, -1)

    # Approximate full-batch cov (cov_all) by averaging per-sequence covs;
    # avoids an extra all-token covariance pass during training.
    if training and not full_mode:
        running_cov = running_cov.mean(dim=0, keepdim=True)

    return Y, running_mean.detach(), running_cov.detach(), gamma, beta


class BatchCenterBlock(nn.Module):
    """
    Mean centering only (no covariance whitening) for debugging.
    
    center_mode options:
    - "leave_one_out": Simple leave-one-out mean. At inference, uses last training mean.
    - "running_mean": Mix running_mean with leave-one-out mean. Updates running_mean.
    - "calibrated": Mix calibrated running_mean with leave-one-out. Does NOT update running_mean.
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-8, center_mode="leave_one_out"):
        super().__init__()
        self.n_features = num_features
        self.eps = eps
        self.center_mode = center_mode
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        group_size, mom = get_batch_whitening_config(B=32, T=64, C=num_features, momentum=1-momentum, threshold=0.1)
        self.group_size = group_size
        self.num_groups = num_features // group_size
        self.momentum = 1.0 - mom

        self.register_buffer('running_mean', torch.zeros(1, self.num_groups, self.group_size))
        # Calibration buffers
        self.register_buffer('calib_sum', torch.zeros(1, self.num_groups, self.group_size))
        self.register_buffer('calib_count', torch.zeros((), dtype=torch.long))
        self.collect_calib_stats = False

    @torch.no_grad()
    def reset_calibration(self):
        self.calib_sum.zero_()
        self.calib_count.zero_()

    @torch.no_grad()
    def _accumulate_calibration_from_input(self, X: torch.Tensor):
        B, T, C = X.shape
        Xg = X.reshape(B * T, self.num_groups, self.group_size).to(torch.float32)
        sum_x = Xg.sum(dim=0, keepdim=True)
        self.calib_sum.add_(sum_x.to(self.calib_sum.dtype))
        self.calib_count.add_(Xg.shape[0])

    @torch.no_grad()
    def commit_calibration(self):
        n = int(self.calib_count.item())
        if n <= 0:
            return
        mean = self.calib_sum / float(n)
        self.running_mean.copy_(mean.to(self.running_mean.dtype))

    def forward(self, X):
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.calib_sum = self.calib_sum.to(X.device)
            self.calib_count = self.calib_count.to(X.device)

        if self.collect_calib_stats:
            self._accumulate_calibration_from_input(X)

        Y, self.running_mean, self.gamma, self.beta = batch_center_only(
            X, self.gamma, self.beta, self.running_mean,
            eps=self.eps, momentum=self.momentum, n_groups=self.num_groups,
            center_mode=self.center_mode,
            training_mode=True if self.collect_calib_stats else None)
        
        return Y


class BatchWhiteningBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,momentum=0.1,eps=1e-8,pre_bias_block=None,num_bias_features=None,center_mode="running_mean",group_size=None,
                 fix_factor=0.9, warmup_ema_buffer=False, inference_ema=False, grad_scale=None,
                 calib_center="loo", renorm=False, renorm_stats="loo", stat_grad="full",
                 renorm_cache=True, shrink_target="diag", renorm_fused=False,
                 whiten="cholesky", ns_iters=5, ns_refresh=0, fast=False, side_stream=False):
        super().__init__()
        # Set (to its eps) by the owner when BW is followed by a LayerNorm without
        # gain or bias, which is then applied here (fused in the fast path).
        self.post_norm_eps = None
        self.renorm_fused = renorm_fused
        self.whiten, self.ns_iters, self.ns_refresh = whiten, ns_iters, ns_refresh
        self.fast, self.side_stream = fast, side_stream
        self.stat_grad = stat_grad
        self.shrink_target = shrink_target
        self.renorm_cache = {} if renorm_cache else None
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.eps = eps
        self.center_mode = center_mode
        self.cov_warmup=True
        self.fix_factor = fix_factor
        self.warmup_ema_buffer = warmup_ema_buffer
        self.inference_ema = inference_ema
        self.grad_scale = grad_scale
        self.calib_center = calib_center
        self.renorm = renorm
        self.renorm_stats = renorm_stats
        self.gamma = nn.Parameter(torch.ones(num_features))
        # The variables that are not model parameters are initialized to 0 and 1

        # TODO: how do we know B and T here?
        derived_group_size, mom = get_batch_whitening_config(B=32, T=64, C=num_features, momentum=1-momentum, threshold=0.1)
        if group_size is None:
            group_size = derived_group_size
        elif num_features % group_size != 0:
            raise ValueError(
                f"group_size={group_size} must divide num_features={num_features}"
            )
        # TODO: remove!!!!!!!!!!!!!!!!!! Group size for nanoGPT with momentum 0.1 is 16
        #group_size = 128
        self.group_size = group_size
        self.num_groups = num_features // group_size
        self.momentum = 1.0 - mom
        self.momentum = momentum  # Use the explicitly passed value, not the derived one from get_batch_whitening_config

        self.register_buffer('running_mean', torch.zeros(1, self.num_groups, self.group_size))
        self.register_buffer('running_cov', torch.eye(self.group_size).view(1, 1, self.group_size, self.group_size).repeat(1, self.num_groups, 1, 1))
        # Per-iteration calibration buffers for pooled mean/cov statistics.
        # We accumulate a numerically stable *centered* co-moment (Welford/Chan)
        # rather than raw first/second moments. Computing the covariance as
        # E[XX^T] - mu mu^T subtracts two large near-equal matrices and, in fp32,
        # loses the small eigenvalues to catastrophic cancellation (producing a
        # non-PSD matrix that crashes the Cholesky). The centered co-moment
        # M2 = sum (x-mu)(x-mu)^T is PSD by construction, so cov = M2/n is stable.
        self.register_buffer('calib_mean', torch.zeros(1, self.num_groups, self.group_size))
        self.register_buffer('calib_M2', torch.zeros(1, self.num_groups, self.group_size, self.group_size))
        self.register_buffer('calib_count', torch.zeros((), dtype=torch.long))
        self.collect_calib_stats = False
        self.update_running_stats = True
        self.update_running_cov_stats = True
        self.pre_bias_block=pre_bias_block

        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    @torch.no_grad()
    def reset_calibration(self):
        self.calib_mean.zero_()
        self.calib_M2.zero_()
        self.calib_count.zero_()

    @torch.no_grad()
    def _accumulate_calibration_from_input(self, X: torch.Tensor):
        if X.dim() != 3:
            raise ValueError(f"Expected X to be 3D (B,T,C), got {tuple(X.shape)}")
        B, T, C = X.shape
        if C != self.n_features:
            raise ValueError(
                f"Expected C={self.n_features} features, got C={C}"
            )
        # (N, n_groups, group_size) in fp32 for numerical stability
        Xg = X.reshape(B * T, self.num_groups, self.group_size).to(torch.float32)
        n_b = Xg.shape[0]

        # Per-batch centered statistics (stable: no large-minus-large subtraction).
        mean_b = Xg.mean(dim=0, keepdim=True)  # (1, n_groups, G)
        Xc = Xg - mean_b
        m2_b = torch.einsum("ngc,ngd->gcd", Xc, Xc).unsqueeze(0)  # (1, ng, G, G)

        # Chan parallel merge of (count, mean, M2) into the running buffers.
        n_a = int(self.calib_count.item())
        if n_a == 0:
            self.calib_mean.copy_(mean_b.to(self.calib_mean.dtype))
            self.calib_M2.copy_(m2_b.to(self.calib_M2.dtype))
        else:
            n_total = n_a + n_b
            delta = mean_b - self.calib_mean  # (1, n_groups, G)
            self.calib_mean.add_((delta * (n_b / n_total)).to(self.calib_mean.dtype))
            # Cross term (n_a*n_b/n_total) * delta delta^T captures the
            # between-batch mean spread so M2 stays the exact total co-moment.
            cross = (delta.unsqueeze(-1) * delta.unsqueeze(-2)) * (n_a * n_b / n_total)
            self.calib_M2.add_((m2_b + cross).to(self.calib_M2.dtype))
        self.calib_count.add_(n_b)

    @torch.no_grad()
    def commit_calibration(self):
        """Copy calibrated mean/cov into the running buffers.

        cov = M2 / n is the population covariance built from the centered
        co-moment, mathematically identical to E[XX^T] - mu mu^T but computed
        without catastrophic cancellation.
        """
        n = int(self.calib_count.item())
        if n <= 0:
            return
        mean = self.calib_mean  # (1, n_groups, G)
        cov = self.calib_M2 / float(n)  # (1, n_groups, G, G)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        eye = torch.eye(
            self.group_size,
            device=cov.device,
            dtype=cov.dtype,
        ).view(1, 1, self.group_size, self.group_size)
        cov = cov + self.eps * eye
        self.running_mean.copy_(mean.to(self.running_mean.dtype))
        self.running_cov.copy_(cov.to(self.running_cov.dtype))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
            self.calib_mean = self.calib_mean.to(X.device)
            self.calib_M2 = self.calib_M2.to(X.device)
            self.calib_count = self.calib_count.to(X.device)

        if X.is_cuda and self.renorm_cache and "event" in self.renorm_cache:
            torch.cuda.current_stream().wait_event(self.renorm_cache["event"])  # buffers written on the side stream
        if self.collect_calib_stats:
            self._accumulate_calibration_from_input(X)

        if (self.fast and torch.is_grad_enabled() and not self.collect_calib_stats and self.renorm
                and self.renorm_stats == "batch" and self.renorm_fused and self.num_groups == 1
                and self.center_mode in ("running_mean", "leave_one_out") and not self.cov_warmup
                and self.stat_grad == "full" and self.grad_scale in (None, 1.0)
                and self.pre_bias_block is None and X.dtype == torch.float32):
            return renorm_fast_whiten(
                X, self.gamma, self.beta, self.running_mean, self.running_cov, self.eps, self.momentum,
                self.fix_factor, self.center_mode, self.update_running_stats, self.update_running_cov_stats,
                self.renorm_cache, self.shrink_target, self.whiten, self.ns_iters, self.ns_refresh,
                self.side_stream, self.post_norm_eps)

        # Save the updated running_mean and moving_var
        Y, self.running_mean, self.running_cov, self.gamma, self.beta = batch_orthonorm(
            X, self.gamma, self.beta, self.running_mean,
            self.running_cov, eps=self.eps, momentum=self.momentum,cov_warmup=self.cov_warmup, bias=False, n_groups=self.num_groups,
            fix_factor=self.fix_factor,
            update_running_stats=self.update_running_stats,
            update_running_cov_stats=self.update_running_cov_stats,
            center_mode=self.center_mode,
            warmup_ema_buffer=self.warmup_ema_buffer,
            inference_ema=self.inference_ema,
            grad_scale=self.grad_scale,
            calib_center=self.calib_center,
            renorm=self.renorm,
            renorm_stats=self.renorm_stats,
            stat_grad=self.stat_grad,
            renorm_cache=self.renorm_cache,
            shrink_target=self.shrink_target,
            renorm_fused=self.renorm_fused,
            whiten=self.whiten,
            ns_iters=self.ns_iters,
            ns_refresh=self.ns_refresh,
            training_mode=True if self.collect_calib_stats else None)
        if self.pre_bias_block is not None:
            Y=self.pre_bias_block(Y)
        # add the bias
        shape = (1, 1, self.n_bias_features)
        Y = Y + self.beta.view(shape)
        if self.post_norm_eps is not None:
            Y = F.layer_norm(Y, (Y.size(-1),), eps=self.post_norm_eps)
        return Y


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Per-token scale normalization with a learned gain, without LayerNorm's centering."""

    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def _make_norm_layer(config, layer_norm_fallback=True):
    """Create a normalization layer based on config.

    ``layer_norm_fallback`` controls what the ``"layer_norm"`` baseline returns:
    - ``True`` (default): return a real ``LayerNorm``. Use this where the layer
      *is* the normalization (e.g. ``ln_f``, or a block that has no separate
      LayerNorm), so the baseline matches the original nano-GPT.
    - ``False``: return ``nn.Identity()``. Use this where the layer is an *extra*
      normalization applied on top of an existing ``LayerNorm`` (e.g. ``bw_2``,
      used as ``ln_2(bw_2(x))``). Returning a LayerNorm there would stack two
      LayerNorms (``LN(LN(x))``), which is not the original architecture.
    """
    if config.normalization_type == "full_bw":
        return BatchWhiteningBlock(
            config.n_embd,
            momentum=config.bw_momentum,
            center_mode=config.batch_center_mode,
            group_size=config.bw_group_size,
            fix_factor=config.bw_fix_factor,
            warmup_ema_buffer=(config.bw_warmup == "diag_ema"),
            inference_ema=config.bw_inference_ema,
            grad_scale=config.bw_grad_scale or None,
            calib_center=config.bw_calib_center,
            renorm=config.bw_renorm,
            renorm_stats=config.bw_renorm_stats,
            stat_grad=config.bw_stat_grad,
            renorm_cache=config.bw_renorm_cache,
            shrink_target=config.bw_shrink_target,
            renorm_fused=config.bw_renorm_fused,
            # ns_chol needs renorm with batch statistics; without them it is Cholesky
            whiten=("cholesky" if config.bw_whiten == "ns_chol" and not (config.bw_renorm and config.bw_renorm_stats == "batch")
                    else config.bw_whiten),
            ns_iters=config.bw_ns_iters,
            ns_refresh=config.bw_ns_refresh,
            fast=config.bw_fast,
            side_stream=config.bw_side_stream,
        )
    elif config.normalization_type == "center_only":
        return BatchCenterBlock(config.n_embd, center_mode=config.batch_center_mode)
    elif config.normalization_type == "layer_norm":
        return LayerNorm(config.n_embd, bias=config.bias) if layer_norm_fallback else nn.Identity()
    else:
        raise ValueError(f"Unknown normalization_type: {config.normalization_type}")


class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        if config.bw_where:
            in_block = layer_idx in {int(i) for i in config.bw_where.split(",")}
        else:
            in_block = config.bw_layers <= 0 or layer_idx < config.bw_layers
        norm = lambda: _make_norm_layer(config, layer_norm_fallback=False)
        self.bw_1 = norm() if in_block and config.bw_sites in ("attn", "both") else nn.Identity()
        self.bw_2 = norm() if in_block and config.bw_sites in ("mlp", "both") else nn.Identity()
        post = config.bw_post_norm if config.bw_ln_after else "none"
        C = config.n_embd

        def pre(bw):  # normalization before BW
            return LayerNorm(C, bias=config.bias) if post == "ln_before" and not isinstance(bw, nn.Identity) else nn.Identity()

        def after(bw):  # normalization between BW (or nothing) and the sublayer
            if post == "ln_noaffine" and config.bw_fuse_post_norm and isinstance(bw, BatchWhiteningBlock):
                bw.post_norm_eps = 1e-5  # nn.LayerNorm's default
                return nn.Identity()
            if isinstance(bw, nn.Identity) or post == "ln":
                return LayerNorm(C, bias=config.bias)
            return {"rms": lambda: RMSNorm(C), "ln_noaffine": lambda: nn.LayerNorm(C, elementwise_affine=False),
                    "none": nn.Identity, "ln_before": nn.Identity}[post]()

        self.pre_1, self.ln_1 = pre(self.bw_1), after(self.bw_1)
        self.attn = CausalSelfAttention(config)
        self.pre_2, self.ln_2 = pre(self.bw_2), after(self.bw_2)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(self.bw_1(self.pre_1(x))))
        x = x + self.mlp(self.ln_2(self.bw_2(self.pre_2(x))))
        return x

class FirstBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm_layer(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # Per-sublayer dropout overrides. None -> fall back to `dropout`.
    # Lets you e.g. keep MLP dropout while disabling attention dropout.
    attn_dropout: float = None
    mlp_dropout: float = None
    # Dropout on the attention output projection (residual add).
    # None -> fall back to `attn_dropout`.
    resid_dropout: float = None
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # Which normalization implementation to use:
    # - "full_bw": BatchWhiteningBlock (mean + covariance whitening)
    # - "center_only": BatchCenterBlock (mean centering only)
    # - "layer_norm": standard LayerNorm baseline
    normalization_type: str = None
    # Centering/statistics mode for BatchWhiteningBlock or BatchCenterBlock.
    # Options: "leave_one_out", "running_mean", "calibrated", "full"
    batch_center_mode: str = None
    # Optional override for the BW group size. None -> derive automatically via
    # get_batch_whitening_config(). Set to n_embd to disable grouping (a single
    # full C x C covariance), which is useful for whitening-correctness tests.
    bw_group_size: int = None
    # BW momentum: weight of the current batch in the running statistics.
    bw_momentum: float = 0.1
    # Off-diagonal covariance shrinkage (1 - epsilon in the paper).
    bw_fix_factor: float = 0.9
    # Covariance warmup until the first eval: "diag" (diagonal-dominant whitening,
    # buffer overwritten), "diag_ema" (same whitening, buffer EMA-accumulates the
    # full covariance), or "none".
    bw_warmup: str = "diag"
    # leave_one_out only: EMA buffers for inference instead of the last batch.
    bw_inference_ema: bool = False
    # Gradient boost on the current-batch BW statistics. 0 -> per-mode default
    # (1/momentum for calibrated, 1 otherwise).
    bw_grad_scale: float = 0.0
    # calibrated only: "loo", "blend_loo_cov", or "blend" (see batch_orthonorm).
    # "blend" makes the activations run away; kept only for comparison.
    bw_calib_center: str = "loo"
    # Causal training forward: whiten with the previous iterations' statistics,
    # gradient through the current batch (see batch_orthonorm). Without it,
    # stacked BW layers leak each sequence's future tokens in training mode.
    bw_renorm: bool = False
    # renorm gradient statistics: "loo" (per sequence) or "batch" (shared, cheaper).
    bw_renorm_stats: str = "loo"
    # BW only in the first bw_layers blocks (0 -> all); later blocks keep plain LayerNorm.
    bw_layers: int = 0
    # Comma-separated block indices that get BW (e.g. "0,1"); overrides bw_layers.
    bw_where: str = ""
    # Which sublayer inputs get BW: "mlp" (ln_2(bw_2(x))), "attn" (ln_1(bw_1(x))) or "both".
    bw_sites: str = "mlp"
    # False drops the LayerNorm that follows each BW layer (BW feeds the linear layer directly).
    bw_ln_after: bool = True
    # Normalization around each BW layer: "ln" (BW -> LayerNorm), "rms" (BW -> RMSNorm),
    # "ln_noaffine" (BW -> LayerNorm without gain/bias), "none" (BW only), "ln_before" (LayerNorm -> BW).
    bw_post_norm: str = "ln"
    bw_fuse_post_norm: bool = True  # "ln_noaffine": apply it inside BW (same values, fewer passes)
    # renorm "batch" only: current-batch statistics with gradient, "full", "mean" or "none".
    bw_stat_grad: str = "full"
    # renorm "batch" only: reuse last step's whitening matrix when it equals the buffers' (exact).
    bw_renorm_cache: bool = True
    # renorm "batch" only: shrink the covariance toward its "diag" (fix_factor) or "identity".
    bw_shrink_target: str = "diag"
    # renorm "batch" only: hand-written backward (same value and gradient, faster).
    bw_renorm_fused: bool = True
    bw_whiten: str = "cholesky"  # "cholesky", "ns", "ns_warm" or "ns_chol" (NS forward, Cholesky backward; renorm batch only)
    bw_ns_iters: int = 5
    bw_fast: bool = True  # renorm batch, one group: fewer passes over the activations (same math)
    bw_side_stream: bool = True  # bw_fast: whitening-matrix math on a side CUDA stream
    bw_ns_refresh: int = 0  # ns_warm: cold restart every this many steps (0: never)
    # Deprecated compatibility field for older checkpoints/configs.
    batch_whitening: bool = None

    def __post_init__(self):
        # Resolve per-sublayer dropout overrides against the general `dropout`.
        if self.attn_dropout is None:
            self.attn_dropout = self.dropout
        if self.mlp_dropout is None:
            self.mlp_dropout = self.dropout
        if self.resid_dropout is None:
            self.resid_dropout = self.attn_dropout

        if self.normalization_type is None:
            if self.batch_whitening is True:
                self.normalization_type = (
                    "center_only" if self.batch_center_mode is not None else "full_bw"
                )
            elif self.batch_whitening is False and self.batch_center_mode is not None:
                self.normalization_type = "center_only"
            else:
                self.normalization_type = "layer_norm"

        if self.batch_center_mode is None:
            self.batch_center_mode = "running_mean"

        valid_normalization_types = {"layer_norm", "center_only", "full_bw"}
        if self.normalization_type not in valid_normalization_types:
            raise ValueError(
                f"normalization_type must be one of {sorted(valid_normalization_types)}, "
                f"got {self.normalization_type!r}"
            )

        if self.bw_sites not in ("mlp", "attn", "both"):
            raise ValueError(f"bw_sites must be 'mlp', 'attn' or 'both', got {self.bw_sites!r}")
        if self.bw_post_norm not in ("ln", "rms", "ln_noaffine", "none", "ln_before"):
            raise ValueError(f"Unknown bw_post_norm: {self.bw_post_norm!r}")

        # "full" is a verification-only oracle (whole-batch, breaks causality).
        valid_center_modes = {"leave_one_out", "running_mean", "calibrated", "full"}
        if self.batch_center_mode not in valid_center_modes:
            raise ValueError(
                f"batch_center_mode must be one of {sorted(valid_center_modes)}, "
                f"got {self.batch_center_mode!r}"
            )

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),  # h = nn.ModuleList([FirstBlock(config)] + [Block(config) for _ in range(config.n_layer - 1)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            bw_f = _make_norm_layer(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # create a list of all BW layers in the model
        self.bw_layers = self._get_bw_layers()
        # cov_warmup is only meaningful for BatchWhiteningBlock (full-cov path).
        # When the model only has BatchCenterBlock layers (centering-only modes),
        # there is no warmup phase, so all batch_center_mode options behave
        # consistently from iter 0 (no jump in the calibrated mode at iter == eval_interval).
        has_cov_warmup_layers = (
            any(hasattr(layer, 'cov_warmup') for layer in self.bw_layers)
            and config.bw_warmup != "none"
        )
        self.curr_cov_warmup = None  # force the layers to take the initial value
        self.set_bw_cov_warmup(has_cov_warmup_layers)

    def _get_bw_layers(self):
        bw_layers = []

        def _extract_layers_recursive(module):
            for name, submodule in module.named_children():
                # Include both BatchWhiteningBlock and BatchCenterBlock
                if isinstance(submodule, (BatchWhiteningBlock, BatchCenterBlock)):
                    bw_layers.append(submodule)
                # If the submodule has children, recursively call this function
                if len(list(submodule.children())) > 0:
                    _extract_layers_recursive(submodule)

        _extract_layers_recursive(self)
        return bw_layers

    def set_bw_cov_warmup(self,cov_warmup):
        if self.curr_cov_warmup != cov_warmup:
            for layer in self.bw_layers:
                # Only BatchWhiteningBlock has cov_warmup
                if hasattr(layer, 'cov_warmup'):
                    layer.cov_warmup = cov_warmup
            self.curr_cov_warmup = cov_warmup
        return

    def causal_leak(self, idx):
        """Max change of sequence 0's first-half logits, relative to their
        scale, when only sequence 0's second half is replaced. Runs the
        training-mode forward (batch statistics, no dropout) in both warmup
        states and restores every buffer, so the model is left untouched. A
        causal model gives exactly 0; any batch statistic shared across stacked
        layers gives a positive value.
        """
        B, T = idx.shape
        H = T // 2
        perturbed = idx.clone()
        perturbed[0, H:] = torch.roll(idx[0, H:], 1)
        saved = {k: v.clone() for k, v in self.named_buffers()}
        was_training, warmup = self.training, self.curr_cov_warmup
        self.eval()  # dropout off; BW takes the training path while grad is on
        leak = 0.0
        try:
            for cov_warmup in (True, False):
                self.set_bw_cov_warmup(cov_warmup)
                outs = []
                for x in (idx, perturbed):
                    with torch.enable_grad():
                        logits, _ = self(x, targets=x)
                    outs.append(logits[0, :H].detach().float())
                    for k, v in self.named_buffers():
                        v.copy_(saved[k])
                diff = (outs[0] - outs[1]).abs().max() / outs[0].abs().max()
                leak = max(leak, diff.item())
        finally:
            self.set_bw_cov_warmup(warmup)
            self.train(was_training)
        return leak

    def bw_calibration_reset(self):
        for layer in self.bw_layers:
            layer.reset_calibration()

    def bw_calibration_enable(self, enabled: bool):
        for layer in self.bw_layers:
            layer.collect_calib_stats = bool(enabled)

    def bw_calibration_commit(self):
        for layer in self.bw_layers:
            layer.commit_calibration()

    def set_bw_update_running_stats(self, enabled: bool):
        for layer in self.bw_layers:
            # Only BatchWhiteningBlock has update_running_stats
            if hasattr(layer, 'update_running_stats'):
                layer.update_running_stats = bool(enabled)

    def set_bw_update_running_cov_stats(self, enabled: bool):
        for layer in self.bw_layers:
            # Only BatchWhiteningBlock has update_running_cov_stats
            if hasattr(layer, 'update_running_cov_stats'):
                layer.update_running_cov_stats = bool(enabled)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # x = self.transformer.bw_f(x)  # Remove this

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
