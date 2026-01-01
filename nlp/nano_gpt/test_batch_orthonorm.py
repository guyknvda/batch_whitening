import torch

from nlp.nano_gpt.model import batch_orthonorm, get_batch_whitening_config


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test parameters
    B, T, C = 32, 64, 128
    eps = 1e-6
    momentum = 0.1
    warmup_iters = 5
    train_iters = 10

    # Optional gamma/beta
    gamma = torch.ones(C)
    beta = torch.zeros(C)

    group_size, mom = get_batch_whitening_config(B, T, C, 1 - momentum, 0.1)
    n_groups = C // group_size
    momentum = 1.0 - mom
    print(
        f'B, T, C, n_groups, group_size, momentum='
        f'{B,T,C,n_groups, group_size, momentum}'
    )

    def analyze_output(label, Y):
        print(f"\n=== {label} mode ===")
        B_local, T_local, D_local = Y.shape
        ng = D_local // group_size
        Ys = Y.reshape(B_local * T_local, ng, group_size).transpose(0, 1)

        means = Ys.mean(dim=1)
        covs = torch.matmul(Ys.transpose(1, 2), Ys) / (Ys.shape[1] - 1)

        # print("Mean per group (should be ~0):")
        # print(means)
        # print("Covariance matrices (should be identity):")
        # print(covs)

        mean_ok = torch.allclose(means, torch.zeros_like(means), atol=5e-2)
        cov_ok = torch.allclose(
            covs,
            torch.eye(group_size, device=covs.device).expand_as(covs),
            atol=7e-2,
        )
        eye_g = torch.eye(group_size, device=covs.device)
        max_cov_err = ((covs - eye_g).abs()).max()
        print(f"Mean check: pass={mean_ok} (max |μ|={means.abs().max():.4f})")
        print(f"Cov check: pass={cov_ok} (max |Σ-I|={max_cov_err:.4f})")

        Y_flat = Y.permute(2, 0, 1).reshape(D_local, -1)
        cov_full = (Y_flat @ Y_flat.T) / (Y_flat.shape[1] - 1)
        cov_full_ok = torch.allclose(
            cov_full,
            torch.eye(D_local, device=Y.device, dtype=Y.dtype),
            atol=9e-2,
        )
        cov_eye = torch.eye(D_local, device=Y.device)
        max_full_err = (cov_full - cov_eye).abs().max()
        print(
            f"Full-cov check: pass={cov_full_ok} "
            f"(max |Σ_full-I|={max_full_err:.4f})"
        )

    def run_mode(
        label,
        X_input,
        running_mean,
        running_cov,
        enable_grad,
        training_mode,
        cov_warmup,
    ):
        ctx = torch.set_grad_enabled(enable_grad)
        with ctx:
            X_autograd = (
                X_input.clone().detach().requires_grad_(True)
                if enable_grad
                else X_input
            )
            (
                Y_mode,
                running_mean_out,
                running_cov_out,
                _,
                _,
            ) = batch_orthonorm(
                X_autograd,
                gamma=gamma,
                beta=beta,
                running_mean=running_mean,
                running_cov=running_cov,
                eps=eps,
                n_groups=n_groups,
                cov_warmup=cov_warmup,
                training_mode=training_mode,
                momentum=momentum,
            )
        analyze_output(label, Y_mode)
        return running_mean_out, running_cov_out, Y_mode, X_autograd

    def causality_check(label, Y_mode, X_input, expect_cross_zero):
        seq_idx = 0
        token_idx = 3
        grad = torch.autograd.grad(
            Y_mode[seq_idx, token_idx, :].sum(),
            X_input,
            retain_graph=False,
            allow_unused=False,
        )[0]
        seq_grad_norms = grad[seq_idx].norm(dim=-1)
        mask = torch.ones(T, dtype=torch.bool)
        mask[token_idx] = False
        max_other_grad = seq_grad_norms[mask].max()
        causal_ok = max_other_grad < 1e-8

        other_seq = 1 if B > 1 else 0
        grad_other_seq = grad[other_seq].norm(dim=-1)
        cross_seq_max = grad_other_seq.max()
        if expect_cross_zero:
            cross_seq_ok = cross_seq_max < 1e-8
        else:
            cross_seq_ok = cross_seq_max > 1e-6

        print(
            f"{label} causality: same seq pass={causal_ok}, "
            f"cross seq pass={cross_seq_ok}"
        )

    running_mean = torch.zeros(1, n_groups, group_size)
    running_cov = (
        torch.eye(group_size)
        .view(1, 1, group_size, group_size)
        .repeat(1, n_groups, 1, 1)
    )

    last_warmup = None
    for step in range(1, warmup_iters + 1):
        label = f"Training (warmup) step {step}/{warmup_iters}"
        running_mean, running_cov, Y_warm, X_warm = run_mode(
            label,
            torch.randn(B, T, C),
            running_mean,
            running_cov,
            enable_grad=True,
            training_mode=True,
            cov_warmup=True,
        )
        last_warmup = (Y_warm, X_warm)

    last_stable = None
    for step in range(1, train_iters + 1):
        label = f"Training (stable) step {step}/{train_iters}"
        running_mean, running_cov, Y_stable, X_stable = run_mode(
            label,
            torch.randn(B, T, C),
            running_mean,
            running_cov,
            enable_grad=True,
            training_mode=True,
            cov_warmup=False,
        )
        last_stable = (Y_stable, X_stable)

    if last_warmup is not None:
        causality_check("Training (warmup final)", *last_warmup, False)
    if last_stable is not None:
        causality_check("Training (stable final)", *last_stable, False)

    # Inference mode (no grad, reuse training stats)
    X_infer = torch.randn(B, T, C)
    _, _, Y_infer, X_infer = run_mode(
        "Inference",
        X_infer,
        running_mean.clone(),
        running_cov.clone(),
        enable_grad=True,
        training_mode=False,
        cov_warmup=False,
    )
    causality_check("Inference", Y_infer, X_infer, True)