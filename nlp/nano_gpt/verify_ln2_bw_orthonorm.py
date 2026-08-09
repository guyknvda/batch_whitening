import argparse
import os
import sys
import pickle

import numpy as np
import torch

from model import BatchWhiteningBlock, GPT, GPTConfig, fix_cov


def covariance_stats(x: torch.Tensor, n_groups: int, group_size: int):
    """Measure grouped covariance closeness to identity for a (B, T, C) tensor."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, T, C), got {tuple(x.shape)}")

    b, t, c = x.shape
    if c != n_groups * group_size:
        raise ValueError(
            f"Expected C={n_groups * group_size} from groups, got C={c}"
        )

    samples = x.detach().float().reshape(b * t, n_groups, group_size)
    sample_mean = samples.mean(dim=0)
    centered = samples - sample_mean
    cov = torch.einsum("ngc,ngd->gcd", centered, centered) / (samples.shape[0] - 1)

    eye = torch.eye(group_size, device=x.device).expand(n_groups, -1, -1)
    err = cov - eye
    off_diag = err.masked_fill(torch.eye(group_size, device=x.device).bool(), 0.0)
    diag = cov.diagonal(dim1=-2, dim2=-1)
    diag_err = diag - 1.0
    off_diag_count = n_groups * group_size * (group_size - 1)

    token_mean = x.detach().float().mean(dim=-1)
    token_var = x.detach().float().var(dim=-1, unbiased=False)

    return {
        "sample_mean_abs_max": sample_mean.abs().max().item(),
        "cov_err_abs_max": err.abs().max().item(),
        "cov_err_fro_mean": torch.linalg.matrix_norm(err, ord="fro").mean().item(),
        "offdiag_abs_max": off_diag.abs().max().item(),
        "offdiag_abs_mean": (off_diag.abs().sum() / off_diag_count).item(),
        "diag_min": diag.min().item(),
        "diag_max": diag.max().item(),
        "diag_err_abs_max": diag_err.abs().max().item(),
        "diag_err_abs_mean": diag_err.abs().mean().item(),
        "token_mean_abs_max": token_mean.abs().max().item(),
        "token_var_mean": token_var.mean().item(),
    }


def gram_stats(x: torch.Tensor, n_groups: int, group_size: int):
    """Measure grouped raw second moment X^T X / N, without sample-mean removal."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, T, C), got {tuple(x.shape)}")

    b, t, c = x.shape
    if c != n_groups * group_size:
        raise ValueError(
            f"Expected C={n_groups * group_size} from groups, got C={c}"
        )

    samples = x.detach().float().reshape(b * t, n_groups, group_size)
    gram = torch.einsum("ngc,ngd->gcd", samples, samples) / samples.shape[0]

    eye = torch.eye(group_size, device=x.device).expand(n_groups, -1, -1)
    err = gram - eye
    off_diag = err.masked_fill(torch.eye(group_size, device=x.device).bool(), 0.0)
    diag = gram.diagonal(dim1=-2, dim2=-1)
    diag_err = diag - 1.0
    off_diag_count = n_groups * group_size * (group_size - 1)

    return {
        "gram_err_abs_max": err.abs().max().item(),
        "gram_err_fro_mean": torch.linalg.matrix_norm(err, ord="fro").mean().item(),
        "gram_offdiag_abs_max": off_diag.abs().max().item(),
        "gram_offdiag_abs_mean": (off_diag.abs().sum() / off_diag_count).item(),
        "gram_diag_min": diag.min().item(),
        "gram_diag_max": diag.max().item(),
        "gram_diag_err_abs_max": diag_err.abs().max().item(),
        "gram_diag_err_abs_mean": diag_err.abs().mean().item(),
    }


def matrix_identity_stats(matrix: torch.Tensor, label_prefix: str):
    """Summarize closeness of a square matrix, or batch of matrices, to identity."""
    dim = matrix.size(-1)
    eye_values = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    eye_bool = eye_values.bool()
    err = matrix - eye_values
    off_diag = err.masked_fill(eye_bool, 0.0)
    diag = matrix.diagonal(dim1=-2, dim2=-1)
    diag_err = diag - 1.0
    off_diag_count = matrix.numel() - diag.numel()

    return {
        f"{label_prefix}_err_abs_max": err.abs().max().item(),
        f"{label_prefix}_err_fro": torch.linalg.matrix_norm(err, ord="fro").item(),
        f"{label_prefix}_offdiag_abs_max": off_diag.abs().max().item(),
        f"{label_prefix}_offdiag_abs_mean": (off_diag.abs().sum() / off_diag_count).item(),
        f"{label_prefix}_diag_min": diag.min().item(),
        f"{label_prefix}_diag_max": diag.max().item(),
        f"{label_prefix}_diag_err_abs_max": diag_err.abs().max().item(),
        f"{label_prefix}_diag_err_abs_mean": diag_err.abs().mean().item(),
    }


def full_channel_stats(x: torch.Tensor):
    """Measure full-channel covariance and raw Gram over all C channels."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, T, C), got {tuple(x.shape)}")

    samples = x.detach().float().reshape(-1, x.shape[-1])
    sample_mean = samples.mean(dim=0, keepdim=True)
    centered = samples - sample_mean
    cov = centered.T @ centered / (samples.shape[0] - 1)
    gram = samples.T @ samples / samples.shape[0]

    stats = {}
    stats.update(matrix_identity_stats(cov, "full_cov"))
    stats.update(matrix_identity_stats(gram, "full_gram"))
    return stats


def ln_vector_constraint_stats(x: torch.Tensor):
    """Check full-token LayerNorm constraints over the complete channel vector."""
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, T, C), got {tuple(x.shape)}")

    samples = x.detach().float().reshape(-1, x.shape[-1])
    c = samples.shape[-1]
    unit_ones = torch.ones(c, device=samples.device, dtype=samples.dtype) / (c ** 0.5)
    projection = samples @ unit_ones
    norm2_per_dim = samples.pow(2).sum(dim=-1) / c
    gram_diag_mean = samples.pow(2).mean(dim=0).mean()

    return {
        "ones_dir_second_moment": projection.pow(2).mean().item(),
        "ones_dir_abs_max": projection.abs().max().item(),
        "token_norm2_per_dim_mean": norm2_per_dim.mean().item(),
        "token_norm2_per_dim_min": norm2_per_dim.min().item(),
        "token_norm2_per_dim_max": norm2_per_dim.max().item(),
        "full_gram_trace_per_dim": gram_diag_mean.item(),
    }


def format_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label}: "
        f"max|cov-I|={stats['cov_err_abs_max']:.4g}, "
        f"mean||cov-I||_F={stats['cov_err_fro_mean']:.4g}, "
        f"max|offdiag|={stats['offdiag_abs_max']:.4g}, "
        f"mean|offdiag|={stats['offdiag_abs_mean']:.4g}, "
        f"diag=[{stats['diag_min']:.4g}, {stats['diag_max']:.4g}], "
        f"mean|diag-1|={stats['diag_err_abs_mean']:.4g}, "
        f"max|sample_mean|={stats['sample_mean_abs_max']:.4g}, "
        f"max|token_mean|={stats['token_mean_abs_max']:.4g}, "
        f"mean token var={stats['token_var_mean']:.4g}"
    )


def format_gram_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label} raw Gram X^T X/N: "
        f"max|G-I|={stats['gram_err_abs_max']:.4g}, "
        f"mean||G-I||_F={stats['gram_err_fro_mean']:.4g}, "
        f"max|offdiag|={stats['gram_offdiag_abs_max']:.4g}, "
        f"mean|offdiag|={stats['gram_offdiag_abs_mean']:.4g}, "
        f"diag=[{stats['gram_diag_min']:.4g}, {stats['gram_diag_max']:.4g}], "
        f"mean|diag-1|={stats['gram_diag_err_abs_mean']:.4g}"
    )


def format_full_channel_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label} full covariance: "
        f"max|Cov-I|={stats['full_cov_err_abs_max']:.4g}, "
        f"||Cov-I||_F={stats['full_cov_err_fro']:.4g}, "
        f"max|offdiag|={stats['full_cov_offdiag_abs_max']:.4g}, "
        f"mean|offdiag|={stats['full_cov_offdiag_abs_mean']:.4g}, "
        f"diag=[{stats['full_cov_diag_min']:.4g}, {stats['full_cov_diag_max']:.4g}], "
        f"mean|diag-1|={stats['full_cov_diag_err_abs_mean']:.4g}"
    )


def format_full_gram_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label} full raw Gram X^T X/N: "
        f"max|G-I|={stats['full_gram_err_abs_max']:.4g}, "
        f"||G-I||_F={stats['full_gram_err_fro']:.4g}, "
        f"max|offdiag|={stats['full_gram_offdiag_abs_max']:.4g}, "
        f"mean|offdiag|={stats['full_gram_offdiag_abs_mean']:.4g}, "
        f"diag=[{stats['full_gram_diag_min']:.4g}, {stats['full_gram_diag_max']:.4g}], "
        f"mean|diag-1|={stats['full_gram_diag_err_abs_mean']:.4g}"
    )


def format_ln_constraint_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label} full-token LN constraint: "
        f"E[(x·1/sqrt(C))^2]={stats['ones_dir_second_moment']:.4g}, "
        f"max|x·1/sqrt(C)|={stats['ones_dir_abs_max']:.4g}, "
        f"mean||x||^2/C={stats['token_norm2_per_dim_mean']:.4g}, "
        f"||x||^2/C=[{stats['token_norm2_per_dim_min']:.4g}, "
        f"{stats['token_norm2_per_dim_max']:.4g}], "
        f"trace(G)/C={stats['full_gram_trace_per_dim']:.4g}"
    )


def covariance_matrix_stats(cov: torch.Tensor, label: str) -> str:
    """Summarize a covariance tensor ending in (..., group_size, group_size)."""
    group_size = cov.size(-1)
    eye = torch.eye(group_size, device=cov.device, dtype=torch.bool)
    diag = cov.diagonal(dim1=-2, dim2=-1)
    offdiag = cov.masked_fill(eye, 0.0)
    offdiag_count = cov.numel() - diag.numel()
    return (
        f"{label}: diag=[{diag.min().item():.4g}, {diag.max().item():.4g}], "
        f"mean|diag-1|={(diag - 1).abs().mean().item():.4g}, "
        f"mean|offdiag|={(offdiag.abs().sum() / offdiag_count).item():.4g}, "
        f"max|offdiag|={offdiag.abs().max().item():.4g}"
    )


def compute_bw_internal_tensors(x: torch.Tensor, layer: BatchWhiteningBlock):
    """Recompute the tensors used by batch_orthonorm for diagnostics."""
    b, t, c = x.shape
    n_groups = layer.num_groups
    group_size = layer.group_size
    xg = x.detach().float().reshape(b, t, n_groups, group_size)

    sum_all_flat = xg.sum(dim=(0, 1), keepdim=True)
    sum_per_seq = xg.sum(dim=1, keepdim=True)
    mean_other_flat = (sum_all_flat - sum_per_seq) / ((b - 1) * t)

    if layer.center_mode == "leave_one_out":
        current_mean = mean_other_flat
    elif layer.center_mode in ("running_mean", "calibrated"):
        # Use the running_mean snapshotted *before* the forward updated it (the
        # constant history the model actually used); fall back to the buffer.
        prev_mean = getattr(layer, "_debug_prev_mean", None)
        if prev_mean is None:
            prev_mean = layer.running_mean
        running_mean = prev_mean.detach().float().view(1, 1, n_groups, group_size)
        current_mean = (1 - layer.momentum) * running_mean + layer.momentum * mean_other_flat
    else:
        raise ValueError(f"Unknown center_mode: {layer.center_mode}")

    xc_self = xg - current_mean

    # This mirrors the covariance used by batch_orthonorm: per excluded sequence,
    # center all other sequences by the same global leave-one-out mean used for Xc_self.
    x_other_centered = xg.unsqueeze(0) - mean_other_flat.unsqueeze(1)
    diag_mask = torch.eye(b, dtype=torch.bool, device=x.device).view(b, b, 1, 1, 1)
    x_other_centered = x_other_centered.masked_fill(diag_mask, 0.0)
    internal_cov = torch.einsum(
        "b s t g c, b s t g d -> b g c d",
        x_other_centered,
        x_other_centered,
    ) / ((b - 1) * t)
    internal_cov = internal_cov + layer.eps * torch.eye(
        group_size, device=x.device, dtype=internal_cov.dtype
    ).view(1, 1, group_size, group_size)

    if layer.center_mode == "leave_one_out":
        # Full current-batch covariance (coefficient 1.0), same as the model.
        used_cov = internal_cov
    elif layer.cov_warmup:
        # Warmup: diagonally-dominant blend of the current-batch covariance
        # (no history term), mirroring cov_for_whitening during warmup.
        diag = torch.eye(group_size, device=x.device, dtype=internal_cov.dtype)
        used_cov = (
            1 - layer.momentum
        ) * diag.view(1, 1, group_size, group_size) * internal_cov + layer.momentum * internal_cov
    else:
        # running_mean / calibrated: (1 - m) * constant history + m * current batch,
        # mirroring cov_for_whitening in batch_orthonorm. The history is the
        # running_cov snapshotted *before* the forward updated it (see the
        # forward_pre_hook in main); fall back to the current buffer otherwise.
        prev_cov = getattr(layer, "_debug_prev_cov", None)
        if prev_cov is None:
            prev_cov = layer.running_cov
        prev_cov = (
            prev_cov.detach()
            .to(dtype=internal_cov.dtype, device=internal_cov.device)
            .reshape(-1, n_groups, group_size, group_size)
        )
        used_cov = (1 - layer.momentum) * prev_cov + layer.momentum * internal_cov

    fixed_cov = fix_cov(used_cov, layer.fix_factor)
    l = torch.linalg.cholesky(fixed_cov)
    xc_perm = xc_self.permute(0, 2, 3, 1)
    y_perm = torch.linalg.solve_triangular(l, xc_perm, upper=False)
    manual_y = y_perm.permute(0, 3, 1, 2).reshape(b, t, c)

    pooled_samples = xc_self.reshape(b * t, n_groups, group_size)
    pooled_cov = torch.einsum(
        "ngc,ngd->gcd",
        pooled_samples,
        pooled_samples,
    ) / (pooled_samples.shape[0] - 1)
    pooled_cov = pooled_cov + layer.eps * torch.eye(
        group_size, device=x.device, dtype=pooled_cov.dtype
    ).view(1, group_size, group_size)
    pooled_l = torch.linalg.cholesky(pooled_cov)
    pooled_perm = xc_self.permute(2, 3, 0, 1).reshape(n_groups, group_size, b * t)
    pooled_y_perm = torch.linalg.solve_triangular(pooled_l, pooled_perm, upper=False)
    pooled_manual_y = pooled_y_perm.reshape(n_groups, group_size, b, t).permute(2, 3, 0, 1).reshape(b, t, c)

    return xc_self.reshape(b, t, c), internal_cov, used_cov, fixed_cov, manual_y, pooled_cov, pooled_manual_y


def print_bw_debug(block_idx: int, layer: BatchWhiteningBlock, bw_input: torch.Tensor, bw_output: torch.Tensor):
    if bw_input.size(0) <= 1:
        print(f"  BW DEBUG block {block_idx}: skipped because batch size <= 1")
        return

    (
        centered_input,
        internal_cov,
        used_cov,
        fixed_cov,
        manual_y,
        pooled_cov,
        pooled_manual_y,
    ) = compute_bw_internal_tensors(
        bw_input.to(bw_output.device),
        layer,
    )
    manual_y = manual_y.detach().float().cpu()
    bw_output = bw_output.detach().float().cpu()
    max_abs_diff = (manual_y - bw_output).abs().max().item()

    print(
        f"  BW DEBUG block {block_idx}: "
        f"cov_warmup={layer.cov_warmup}, momentum={layer.momentum}, "
        f"fix_factor={layer.fix_factor}, center_mode={layer.center_mode}"
    )
    print("  " + format_stats(
        "centered input measured with pooled covariance",
        covariance_stats(centered_input.cpu(), layer.num_groups, layer.group_size),
    ))
    print("  " + covariance_matrix_stats(internal_cov.cpu(), "internal leave-one-out cov"))
    print("  " + covariance_matrix_stats(used_cov.cpu(), "cov before fix_cov"))
    print("  " + covariance_matrix_stats(fixed_cov.cpu(), "cov used for Cholesky"))
    print("  " + covariance_matrix_stats(pooled_cov.cpu(), "pooled cov of centered input"))
    print("  " + format_stats(
        "if whitened with pooled centered-input cov",
        covariance_stats(pooled_manual_y.cpu(), layer.num_groups, layer.group_size),
    ))
    print(f"  manual whitening vs layer output max|diff|={max_abs_diff:.4g}")


def orthogonality_verdict(stats: dict[str, float], args) -> tuple[bool, list[str]]:
    """Return whether channels are decorrelated: off-diagonal covariance near 0."""
    checks = [
        ("mean|offdiag|", stats["offdiag_abs_mean"], args.offdiag_mean_tol),
        ("max|offdiag|", stats["offdiag_abs_max"], args.offdiag_max_tol),
    ]
    failures = [
        f"{name}={value:.4g} > {limit:.4g}"
        for name, value, limit in checks
        if value > limit
    ]
    return not failures, failures


def orthonormality_verdict(stats: dict[str, float], args) -> tuple[bool, list[str]]:
    """Return whether channels are decorrelated and unit-variance."""
    checks = [
        ("mean|offdiag|", stats["offdiag_abs_mean"], args.offdiag_mean_tol),
        ("max|offdiag|", stats["offdiag_abs_max"], args.offdiag_max_tol),
        ("mean|diag-1|", stats["diag_err_abs_mean"], args.diag_mean_tol),
        ("max|diag-1|", stats["diag_err_abs_max"], args.diag_max_tol),
    ]
    failures = [
        f"{name}={value:.4g} > {limit:.4g}"
        for name, value, limit in checks
        if value > limit
    ]
    return not failures, failures


def apply_train_like_defaults(args, explicit=None):
    """Match the Shakespeare-char override used in train.py unless explicitly set.

    Any option the user passed explicitly on the command line (collected in
    ``explicit``) is preserved rather than being overwritten by the train-like
    default. This matters for e.g. ``--train_like --dropout=0.0``.
    """
    explicit = explicit or set()
    defaults = {
        "batch_size": 64,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.0,
        "bias": False,
        "dataset": "shakespeare_char",
    }
    for key, value in defaults.items():
        if key not in explicit:
            setattr(args, key, value)


def build_batch_getter(args):
    """Return a get_batch(split) function using either real nanoGPT data or synthetic tokens."""
    if not args.real_data:
        def get_synthetic_batch(_split: str):
            return torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.block_size),
                device=args.device,
            )

        return get_synthetic_batch

    data_dir = os.path.join("data", args.dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        args.vocab_size = meta["vocab_size"]

    def get_real_batch(split: str):
        path = os.path.join(data_dir, f"{split}.bin")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([
            torch.from_numpy((data[i:i + args.block_size]).astype(np.int64))
            for i in ix
        ])
        return x.to(args.device)

    return get_real_batch


@torch.no_grad()
def calibrate_model(model: GPT, args, get_batch):
    """Populate calibrated running stats before measurement."""
    was_training = model.training
    try:
        # Match train.py calibration: train-mode module behavior (notably
        # dropout), but no gradient recording.
        model.train()
        model.bw_calibration_reset()
        model.bw_calibration_enable(True)
        for _ in range(args.calib_iters):
            idx = get_batch("train")
            model(idx)
    finally:
        model.bw_calibration_enable(False)
        model.bw_calibration_commit()
        model.set_bw_update_running_stats(False)
        model.set_bw_update_running_cov_stats(False)
        if was_training:
            model.train()


def check_causality(model, config, device, dtype):
    """Verify whether block0.bw_2 leaks information across token positions.

    A causal BW layer's output at token t0 must NOT depend on other token
    positions t != t0 within the same sequence (cross-sequence dependence via
    batch statistics is allowed). The real modes (leave_one_out / running_mean /
    calibrated) are causal. The 'full' oracle uses whole-batch statistics that
    include the token's own sequence, so it is EXPECTED to break causality.
    """
    bw = model.transformer.h[0].bw_2
    if not isinstance(bw, BatchWhiteningBlock):
        print("\n## causality check: skipped (bw_2 is not a BatchWhiteningBlock)")
        return

    C = config.n_embd
    B = 4
    # Ensure B*T > C so the whole-batch covariance is full rank in 'full' mode.
    T = max(16, (2 * C) // B + 4)
    torch.manual_seed(1234)
    x = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)

    was_training = bw.training
    bw.train()
    saved_mean = bw.running_mean.clone()
    saved_cov = bw.running_cov.clone()
    with torch.enable_grad():
        y = bw(x)
    t0 = T // 2
    grad = torch.autograd.grad(y[0, t0, :].sum(), x)[0]  # (B, T, C)
    # Restore buffers so the check does not pollute measured stats.
    bw.running_mean = saved_mean
    bw.running_cov = saved_cov
    if not was_training:
        bw.eval()

    same_seq = grad[0].norm(dim=-1)  # (T,)
    mask = torch.ones(T, dtype=torch.bool, device=grad.device)
    mask[t0] = False
    within_seq_leak = same_seq[mask].max().item()
    cross_seq = grad[1:].norm(dim=-1).max().item() if B > 1 else 0.0
    causal = within_seq_leak < 1e-6

    print("\n## causality check (block0.bw_2)")
    print(f"- within-sequence cross-token leak (should be ~0 if causal): {within_seq_leak:.3e}")
    print(f"- cross-sequence dependence (batch stats, expected > 0):     {cross_seq:.3e}")
    if config.batch_center_mode == "full":
        if not causal:
            print("  VERDICT: NON-CAUSAL as expected for the 'full' oracle "
                  "(other tokens in the same sequence affect the output).")
        else:
            print("  VERDICT: UNEXPECTED - 'full' oracle should break causality "
                  "but the within-sequence leak is ~0.")
    else:
        if causal:
            print(f"  VERDICT: CAUSAL - mode '{config.batch_center_mode}' does not leak "
                  "across token positions within a sequence.")
        else:
            print(f"  VERDICT: UNEXPECTED - mode '{config.batch_center_mode}' should be "
                  f"causal but leaked {within_seq_leak:.3e}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check covariance of bw_2 and ln_2(bw_2(x)) over a few forwards."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--iters", type=int, default=1, help="Immediate train-mode measurement forwards.")
    parser.add_argument("--inference_iters", type=int, default=1, help="Immediate eval-mode measurement forwards.")
    parser.add_argument(
        "--burn_in_iters",
        type=int,
        default=0,
        help="Training-mode forwards before measurement; useful for running_mean EMA stats.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0, help="Base/embedding dropout.")
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=None,
        help="Dropout on attention softmax weights. None -> falls back to --dropout.",
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=None,
        help="Dropout after the MLP (writes to residual, upstream of the next block's BW). None -> --dropout.",
    )
    parser.add_argument(
        "--resid_dropout",
        type=float,
        default=None,
        help="Dropout on the attention output projection (writes to residual, directly upstream of bw_2). None -> --attn_dropout.",
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--real_data", action="store_true", help="Use data/<dataset>/*.bin batches like train.py.")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument(
        "--train_like",
        action="store_true",
        help="Use train.py's Shakespeare-char model/data defaults: B=64,T=256,L=6,H=6,C=384,dropout=0.0,bias=False.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate all measurement iterations before reporting instead of reporting each forward immediately.",
    )
    parser.add_argument(
        "--calib_iters",
        type=int,
        default=5,
        help="Random no-grad calibration forwards used only when batch_center_mode=calibrated.",
    )
    parser.add_argument(
        "--batch_center_mode",
        default="leave_one_out",
        choices=("leave_one_out", "running_mean", "calibrated", "full"),
        help="'full' is a verification oracle: whole-batch in-sample whitening "
        "(breaks causality). With --group_size=n_embd and --fix_factor=1.0 it "
        "should make bw_2 output an exactly-identity covariance.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=0,
        help="BW group size override. 0 = auto (get_batch_whitening_config). "
        "Set equal to n_embd to disable grouping (single full C x C covariance).",
    )
    parser.add_argument("--mean_tol", type=float, default=0.05)
    parser.add_argument("--diag_mean_tol", type=float, default=0.10)
    parser.add_argument("--diag_max_tol", type=float, default=0.35)
    parser.add_argument("--offdiag_mean_tol", type=float, default=0.08)
    parser.add_argument("--offdiag_max_tol", type=float, default=0.35)
    parser.add_argument(
        "--fix_factor",
        type=float,
        default=0.9,
        help="Off-diagonal covariance shrink factor used before Cholesky. Use 1.0 to disable fix_cov.",
    )
    parser.add_argument(
        "--debug_bw",
        action="store_true",
        help="Print internals for the last captured bw_2 batch.",
    )
    parser.add_argument(
        "--wandb_log",
        default=False,
        help="Accepted for train.py command compatibility; this verification script does not log to W&B.",
    )
    parser.add_argument(
        "--stable_cov",
        action="store_true",
        help="Disable BW covariance warmup before measuring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_like:
        # Preserve any option the user set explicitly on the command line.
        explicit = {
            tok.lstrip("-").split("=", 1)[0].replace("-", "_")
            for tok in sys.argv[1:]
            if tok.startswith("--")
        }
        apply_train_like_defaults(args, explicit)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    get_batch = build_batch_getter(args)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout,
        resid_dropout=args.resid_dropout,
        bias=args.bias,
        normalization_type="full_bw",
        batch_center_mode=args.batch_center_mode,
        bw_group_size=(args.group_size if args.group_size > 0 else None),
    )
    model = GPT(config).to(device=args.device, dtype=dtype)
    model.train()
    for layer in model.bw_layers:
        if hasattr(layer, "fix_factor"):
            layer.fix_factor = args.fix_factor
    if args.stable_cov:
        model.set_bw_cov_warmup(False)
    if args.batch_center_mode == "calibrated":
        calibrate_model(model, args, get_batch)

    captured = {}

    def save_hook(name):
        def hook(_module, _inputs, output):
            captured.setdefault(name, []).append(output.detach().float().cpu())

        return hook

    def save_input_output_hook(name):
        def hook(_module, inputs, output):
            captured.setdefault(f"{name}.input", []).append(inputs[0].detach().float().cpu())
            captured.setdefault(name, []).append(output.detach().float().cpu())

        return hook

    def save_prev_stats_hook(module, _inputs):
        # Snapshot the running buffers BEFORE the forward updates them. These are
        # the *constant history* terms used by current_mean / cov_for_whitening in
        # batch_orthonorm, so the --debug_bw reconstruction must use the pre-update
        # values (running_mean is updated with mean_all, running_cov with the
        # current-batch cov during the forward for running_mean mode).
        module._debug_prev_mean = module.running_mean.detach().clone()
        module._debug_prev_cov = module.running_cov.detach().clone()

    hooks = []
    for i, block in enumerate(model.transformer.h):
        if isinstance(block.bw_2, BatchWhiteningBlock):
            hooks.append(block.bw_2.register_forward_hook(save_input_output_hook(f"block{i}.bw_2")))
            hooks.append(block.ln_2.register_forward_hook(save_hook(f"block{i}.ln_2_after_bw_2")))
            if args.debug_bw:
                hooks.append(block.bw_2.register_forward_pre_hook(save_prev_stats_hook))

    def run_forward(split: str):
        idx = get_batch(split)
        context = torch.enable_grad() if model.training else torch.no_grad()
        with context:
            model(idx)

    def print_captured(label: str):
        print(f"\n## {label}")
        for i, block in enumerate(model.transformer.h):
            bw_layer = block.bw_2
            if not isinstance(bw_layer, BatchWhiteningBlock):
                continue

            print(
                f"\nblock {i}: num_groups={bw_layer.num_groups}, "
                f"group_size={bw_layer.group_size}"
            )
            if args.debug_bw:
                print_bw_debug(
                    i,
                    bw_layer,
                    captured[f"block{i}.bw_2.input"][-1],
                    captured[f"block{i}.bw_2"][-1],
                )
            for name in (f"block{i}.bw_2", f"block{i}.ln_2_after_bw_2"):
                output = torch.cat(captured[name], dim=0)
                stats = covariance_stats(output, bw_layer.num_groups, bw_layer.group_size)
                full_stats = full_channel_stats(output)
                print(format_stats(name, stats))
                print(format_gram_stats(name, gram_stats(output, bw_layer.num_groups, bw_layer.group_size)))
                print(format_full_channel_stats(name, full_stats))
                print(format_full_gram_stats(name, full_stats))
                if name.endswith("ln_2_after_bw_2"):
                    print(format_ln_constraint_stats(name, ln_vector_constraint_stats(output)))

                passed, failures = orthonormality_verdict(stats, args)
                label = "ORTHONORMALITY"
                pass_message = "off-diagonal is close to 0 and diagonal is close to 1."
                fail_message = "off-diagonal and/or diagonal are not close enough:"

                if passed:
                    print(f"  {label}: PASS - {pass_message}")
                else:
                    print(f"  {label}: FAIL - {fail_message}")
                    for failure in failures:
                        print(f"    - {failure}")

    print(
        "This script checks grouped and full-channel statistics over each captured B*T token sample set.\n"
        "Interpretation:\n"
        "- grouped covariance/Gram checks use the BW groups, e.g. 16x16 blocks.\n"
        "- full covariance/Gram checks use the complete CxC channel matrix.\n"
        "- covariance stats subtract the sample mean; raw Gram stats use X^T X/N directly.\n"
        "- after bw_2: grouped covariance is the natural BW target because BW is blockwise.\n"
        "- after ln_2(bw_2): LayerNorm centers and normalizes the full token vector, "
        "so the full CxC covariance/Gram cannot be identity; the all-ones direction "
        "should instead have near-zero energy.\n"
        "- LayerNorm also forces each token's channel mean near 0 and variance near 1, "
        "but that is not the same as grouped covariance being identity.\n"
        f"- mode={args.batch_center_mode}"
        + (
            f", calibrated with {args.calib_iters} random train-mode no-grad forwards.\n"
            if args.batch_center_mode == "calibrated"
            else ".\n"
        )
        + f"- burn_in_iters={args.burn_in_iters}\n"
        + f"- train iters={args.iters}, inference iters={args.inference_iters}, aggregate={args.aggregate}\n"
        + f"- real_data={args.real_data}, dataset={args.dataset}, train_like={args.train_like}\n"
        + f"- dropout: base={config.dropout}, attn={config.attn_dropout}, "
        + f"mlp={config.mlp_dropout}, resid={config.resid_dropout}, "
        + "live nn.Dropout p="
        + str(sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}))
        + "\n"
        + f"- fix_factor={args.fix_factor}, "
        + "bw group_size/num_groups="
        + str(sorted({(lyr.group_size, lyr.num_groups) for lyr in model.bw_layers}))
        + "\n"
        + (
            "- ORACLE: center_mode='full' whitens with the whole-batch in-sample "
            "covariance (breaks causality). With group_size=n_embd and "
            "fix_factor=1.0, bw_2 should be an exactly-identity covariance.\n"
            if args.batch_center_mode == "full"
            else ""
        )
    )

    try:
        for _ in range(args.burn_in_iters):
            run_forward("train")
        captured.clear()

        model.train()
        if args.aggregate:
            for _ in range(args.iters):
                run_forward("train")
        else:
            for iter_idx in range(args.iters):
                captured.clear()
                run_forward("train")
                print_captured(f"train forward {iter_idx + 1}/{args.iters}")

        if args.aggregate and args.iters > 0:
            print_captured(f"train aggregate over {args.iters} forwards")

        model.eval()
        if args.inference_iters > 0:
            if args.aggregate:
                captured.clear()
                for _ in range(args.inference_iters):
                    run_forward("val" if args.real_data else "train")
                print_captured(f"inference aggregate over {args.inference_iters} forwards")
            else:
                for iter_idx in range(args.inference_iters):
                    captured.clear()
                    run_forward("val" if args.real_data else "train")
                    print_captured(f"inference forward {iter_idx + 1}/{args.inference_iters}")
    finally:
        for hook in hooks:
            hook.remove()

    check_causality(model, config, args.device, dtype)


if __name__ == "__main__":
    main()
