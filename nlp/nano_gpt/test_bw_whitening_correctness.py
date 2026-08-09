"""Direct correctness tests for batch_orthonorm (no model, float64).

Rules out implementation bugs by checking the whitening in the regime where it
is *mathematically guaranteed* to be exact, then re-introducing the two
approximations (grouping, leave-one-out) and LayerNorm and confirming each
behaves exactly as predicted.

Cholesky/ZCA whitening with the covariance computed over the SAME samples it is
applied to gives an output covariance that is exactly identity. The knobs:
  - center_mode="full"     -> whole-batch in-sample cov (exact; breaks causality)
  - n_groups=1             -> no grouping, whiten the full C x C covariance
  - fix_factor=1.0         -> no off-diagonal shrink in fix_cov
Together these must yield an exactly-identity output covariance.

Run:
    uv run --with torch --with numpy python test_bw_whitening_correctness.py
"""

import torch
import torch.nn.functional as F

from model import batch_orthonorm


torch.manual_seed(0)
DTYPE = torch.float64


def make_correlated(B, T, C):
    """X with correlated channels (cross-group correlation present)."""
    A = torch.randn(C, C, dtype=DTYPE)
    mean = torch.randn(C, dtype=DTYPE) * 3.0
    Z = torch.randn(B, T, C, dtype=DTYPE)
    return Z @ A + mean


def whiten(X, n_groups, center_mode, fix_factor, training=True):
    Y, *_ = batch_orthonorm(
        X, n_groups=n_groups, center_mode=center_mode, fix_factor=fix_factor,
        eps=1e-12, learn_affine=False, bias=False, training_mode=training,
    )
    return Y


def group_cov(Y, n_groups):
    B, T, C = Y.shape
    gs = C // n_groups
    s = Y.reshape(B * T, n_groups, gs)
    s = s - s.mean(dim=0)
    cov = torch.einsum("ngc,ngd->gcd", s, s) / (B * T - 1)
    eye = torch.eye(gs, dtype=Y.dtype)
    return (cov - eye).abs().max().item()


def full_cov(Y):
    B, T, C = Y.shape
    s = Y.reshape(B * T, C)
    s = s - s.mean(dim=0)
    cov = s.T @ s / (B * T - 1)
    eye = torch.eye(C, dtype=Y.dtype)
    off = cov - torch.diag(torch.diagonal(cov))
    return {
        "max|cov-I|": (cov - eye).abs().max().item(),
        "max|offdiag|": off.abs().max().item(),
        "diag[min,max]": (torch.diagonal(cov).min().item(), torch.diagonal(cov).max().item()),
    }


results = []


def check(name, ok, detail):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# (a) EXACT whitening: full mode, no grouping, no fix_cov -> Cov(Y) == I
# ---------------------------------------------------------------------------
B, T, C = 16, 32, 24  # N = 512 >> C=24, so cov is full rank
X = make_correlated(B, T, C)
Y = whiten(X, n_groups=1, center_mode="full", fix_factor=1.0)
fc = full_cov(Y)
check("(a) full + no-group + fix=1 -> exact identity",
      fc["max|cov-I|"] < 1e-6, f"full max|Cov-I|={fc['max|cov-I|']:.2e}")

# ---------------------------------------------------------------------------
# (b) Grouping only block-diagonalizes: per-group blocks are identity, but the
#     full C x C matrix keeps cross-group correlation (not identity).
# ---------------------------------------------------------------------------
n_groups = 4  # gs = 6
Yg = whiten(X, n_groups=n_groups, center_mode="full", fix_factor=1.0)
gmax = group_cov(Yg, n_groups)
fcg = full_cov(Yg)
check("(b) full + grouping -> each group block is identity",
      gmax < 1e-6, f"grouped max|blockCov-I|={gmax:.2e}")
check("(b) full + grouping -> full C x C is NOT identity (cross-group remains)",
      fcg["max|cov-I|"] > 0.05, f"full max|Cov-I|={fcg['max|cov-I|']:.3f}, "
      f"max|offdiag|={fcg['max|offdiag|']:.3f}")

# ---------------------------------------------------------------------------
# (c) Leave-one-out is out-of-sample -> approximately, not exactly, identity.
#     It should be clearly worse than the full-mode oracle.
# ---------------------------------------------------------------------------
Yl = whiten(X, n_groups=1, center_mode="leave_one_out", fix_factor=1.0)
fcl = full_cov(Yl)
check("(c) leave_one_out + no-group -> approx identity (not exact)",
      1e-4 < fcl["max|cov-I|"] < 0.5,
      f"full max|Cov-I|={fcl['max|cov-I|']:.4f} (oracle was {fc['max|cov-I|']:.2e})")

# ---------------------------------------------------------------------------
# (d) LayerNorm after exact whitening: per-token mean~0 & var~1, but the full
#     C x C covariance is no longer identity (all-ones direction loses energy).
# ---------------------------------------------------------------------------
Yln = F.layer_norm(Y, (C,))
tok_mean = Yln.mean(dim=-1)          # per-token channel mean
tok_var = Yln.var(dim=-1, unbiased=False)  # per-token channel var
ones = torch.ones(C, dtype=DTYPE) / (C ** 0.5)
proj = (Yln.reshape(-1, C) @ ones)  # energy along all-ones direction
allones_energy = (proj ** 2).mean().item()
fcln = full_cov(Yln)
check("(d) LN after whitening -> per-token mean ~0",
      tok_mean.abs().max().item() < 1e-6, f"max|token_mean|={tok_mean.abs().max().item():.2e}")
check("(d) LN after whitening -> per-token var ~1",
      (tok_var - 1).abs().max().item() < 1e-3, f"max|token_var-1|={(tok_var-1).abs().max().item():.2e}")
check("(d) LN after whitening -> all-ones direction ~0 energy",
      allones_energy < 1e-6, f"E[(x.1/sqrtC)^2]={allones_energy:.2e}")
print(f"      (info) LN full max|Cov-I|={fcln['max|cov-I|']:.3f}, diag={fcln['diag[min,max]']}")

print()
n_fail = sum(1 for _, ok in results if not ok)
print(f"SUMMARY: {len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    raise SystemExit(1)
