"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

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
    a = torch.ones_like(covmat) * fix_factor

    if covmat.dim() == 2:
        # 2D covariance
        a.fill_diagonal_(1.0)
    elif covmat.dim() == 3:
        # 3D: [B,D,D]
        eye = torch.eye(covmat.size(-1), device=covmat.device)
        a = a * (1 - eye) + eye
    elif covmat.dim() == 4:
        # 4D: [B,n_groups,D,D]
        eye = torch.eye(
            covmat.size(-1),
            device=covmat.device,
        ).view(1, 1, covmat.size(-1), covmat.size(-1))
        a = a * (1 - eye) + eye
    else:
        raise ValueError(f"Unsupported covmat.dim()={covmat.dim()}")

    return a * covmat


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
):
    """
    Fully vectorized block whitening for X of shape (B, T, C).
    No loops. Running stats updated correctly. Whitening is correct.
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
    if training:
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
    if training and B > 1:
        # Calculate leave-one-out mean (averaged over T)
        sum_all_flat = Xg.sum(dim=(0, 1), keepdim=True)  # (1, 1, ng, gs)
        sum_per_seq = Xg.sum(dim=1, keepdim=True)  # (B, 1, ng, gs)
        count_other = (B - 1) * T
        mean_other = (sum_all_flat - sum_per_seq) / count_other

        # Mix with running_mean
        # alpha = 1 - momentum
        current_mean = (
            1 - momentum
        ) * running_mean.view(
            1, 1, n_groups, group_size
        ) + momentum * mean_other
    else:
        current_mean = running_mean.view(1, 1, n_groups, group_size)

    Xc_self = Xg - current_mean

    # Scale to correct for variance bias from leave-one-out component.
    # With mixed mean (1-m)*running + m*mean_other, only the momentum fraction
    # contributes to the bias. When momentum=1, this gives sqrt((B-1)/B).
    '''if training and B > 1:
        scale = (1 - momentum**2 / B) ** 0.5
        Xc_self = Xc_self * scale'''

    if training:
        if B <= 1:
            raise ValueError(
                "Batch size must be greater than 1 during training"
            )
        # --------------------------------------
        # 3. Compute leave-one-out covariance (other sequences only)
        # --------------------------------------
        sum_all = Xg.sum(dim=0, keepdim=True)
        other_count = B - 1
        mean_other = (sum_all - Xg) / other_count  # (B, T, n_groups, G)

        X_other_centered = Xg.unsqueeze(0) - mean_other.unsqueeze(1)
        diag_mask = torch.eye(
            B, dtype=torch.bool, device=device
        ).view(B, B, 1, 1, 1)
        X_other_centered = X_other_centered.masked_fill(diag_mask, 0.0)
        sample_count = (B - 1) * T
        cov = torch.einsum(
            "b s t g c, b s t g d -> b g c d",
            X_other_centered,
            X_other_centered,
        ) / sample_count
        cov = cov + eps * torch.eye(
            group_size,
            device=device,
            dtype=dtype,
        ).view(1, 1, group_size, group_size)

    # --------------------------------------
    # 6. Update running stats
    # --------------------------------------
    if training:
        with torch.no_grad():
            # Only update running_mean if enabled (disabled when using calibration)
            if update_running_stats:
                mean_all = Xg.mean(dim=(0, 1))  # (n_groups, G) over all tokens
                running_mean.mul_(1 - momentum).add_(
                    momentum * mean_all.view(1, n_groups, group_size)
                )

            # Always update running_cov during training
            if cov_warmup:
                diag = torch.eye(group_size, device=device, dtype=dtype)
                x_var = diag.view(1, 1, group_size, group_size) * cov
                running_cov.copy_(
                    (1 - momentum) * x_var + momentum * cov
                )

            else:
                running_cov.mul_(1 - momentum).add_(momentum * cov)

    uncorrelated_running_cov = fix_cov(running_cov, fix_factor)

    # --------------------------------------
    # 7. Cholesky whitening (batched)
    # --------------------------------------
    L = torch.linalg.cholesky(uncorrelated_running_cov)  # (B or 1, n_groups, G, G)
    Xc_perm = Xc_self.permute(0, 2, 3, 1)  # -> (B, n_groups, G, T)
    Y_perm = torch.linalg.solve_triangular(L, Xc_perm, upper=False)

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
    if training:
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
            center_mode=self.center_mode)
        
        return Y


class BatchWhiteningBlock(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features,momentum=0.1,eps=1e-8,pre_bias_block=None,num_bias_features=None):
        super().__init__()
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.n_features=num_features
        self.n_bias_features = num_features if pre_bias_block is None else num_bias_features
        self.eps = eps
        self.cov_warmup=True
        self.gamma = nn.Parameter(torch.ones(num_features))
        # The variables that are not model parameters are initialized to 0 and 1

        # TODO: how do we know B and T here?
        group_size, mom = get_batch_whitening_config(B=32, T=64, C=num_features, momentum=1-momentum, threshold=0.1)
        self.group_size = group_size
        self.num_groups = num_features // group_size
        self.momentum = 1.0 - mom
        self.momentum = momentum  # Use the explicitly passed value, not the derived one from get_batch_whitening_config

        self.register_buffer('running_mean', torch.zeros(1, self.num_groups, self.group_size))
        self.register_buffer('running_cov', torch.eye(self.group_size).view(1, 1, self.group_size, self.group_size).repeat(1, self.num_groups, 1, 1))
        # Per-iteration calibration buffers for mean only.
        self.register_buffer('calib_sum', torch.zeros(1, self.num_groups, self.group_size))
        self.register_buffer('calib_count', torch.zeros((), dtype=torch.long))
        self.collect_calib_stats = False
        self.update_running_stats = True
        self.pre_bias_block=pre_bias_block

        self.beta = nn.Parameter(torch.zeros(self.n_bias_features))

    @torch.no_grad()
    def reset_calibration(self):
        self.calib_sum.zero_()
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
        # sum_x: (1, n_groups, G)
        sum_x = Xg.sum(dim=0, keepdim=True)

        # Accumulate in buffer dtype (typically fp32)
        self.calib_sum.add_(sum_x.to(self.calib_sum.dtype))
        self.calib_count.add_(Xg.shape[0])

    @torch.no_grad()
    def commit_calibration(self):
        """Copy calibrated mean into the running_mean buffer."""
        n = int(self.calib_count.item())
        if n <= 0:
            return
        denom = float(n)
        mean = self.calib_sum / denom  # (1, n_groups, G)
        self.running_mean.copy_(mean.to(self.running_mean.dtype))

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_cov = self.running_cov.to(X.device)
            self.calib_sum = self.calib_sum.to(X.device)
            self.calib_count = self.calib_count.to(X.device)

        if self.collect_calib_stats:
            self._accumulate_calibration_from_input(X)

        # Save the updated running_mean and moving_var
        Y, self.running_mean, self.running_cov, self.gamma, self.beta = batch_orthonorm(
            X, self.gamma, self.beta, self.running_mean,
            self.running_cov, eps=self.eps, momentum=self.momentum,cov_warmup=self.cov_warmup, bias=False, n_groups=self.num_groups,
            update_running_stats=self.update_running_stats)
        if self.pre_bias_block is not None:
            Y=self.pre_bias_block(Y)
        # add the bias
        shape = (1, 1, self.n_bias_features)
        Y = Y + self.beta.view(shape)
        return Y


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def _make_norm_layer(config):
    """Create normalization layer based on config."""
    if config.batch_center_mode is not None:
        # Debug mode: mean centering only, no covariance
        return BatchCenterBlock(config.n_embd, center_mode=config.batch_center_mode)
    elif config.batch_whitening:
        return BatchWhiteningBlock(config.n_embd)  # For leave_one_out use momentum=1.0
    else:
        return LayerNorm(config.n_embd, bias=config.bias)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.bw_1 = _make_norm_layer(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.bw_2 = _make_norm_layer(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # I droped bw from here
        x = x + self.mlp(self.ln_2(self.bw_2(x)))  # Try to switch order and do BW before LN
        return x

class FirstBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = BatchWhiteningBlock(config.n_embd) if config.batch_whitening else LayerNorm(config.n_embd, bias=config.bias)
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
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    batch_whitening: bool = False
    # For debugging: use BatchCenterBlock (mean only, no cov) instead of BatchWhiteningBlock
    # Options: None (use BW), "leave_one_out", "running_mean", "calibrated"
    batch_center_mode: str = None

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # h = nn.ModuleList([FirstBlock(config)] + [Block(config) for _ in range(config.n_layer - 1)]),
            ln_ttt = LayerNorm(config.n_embd, bias=config.bias),
            ln_f = _make_norm_layer(config),
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
        has_cov_warmup_layers = any(hasattr(layer, 'cov_warmup') for layer in self.bw_layers)
        self.curr_cov_warmup = has_cov_warmup_layers
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
        x = self.transformer.ln_ttt(x)
        x = self.transformer.ln_f(x)

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
