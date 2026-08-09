"""Demonstrate why the calibrated-mode covariance must be accumulated as a
centered co-moment (Welford/Chan) instead of raw first/second moments.

It mirrors, in miniature, what BatchWhiteningBlock.commit_calibration does:
pool statistics over several batches, then form a per-group covariance and run
a Cholesky (exactly what the whitening step needs).

Two estimators are compared, both streamed over `n_batches` batches:

  OLD  (raw moments):    accumulate  sum_x, sum_xx    ->  cov = sum_xx/n - mu mu^T
  NEW  (centered M2):    accumulate  Welford (n,mu,M2) ->  cov = M2/n

The input has a LARGE per-channel mean (like the un-normalized residual stream
that feeds bw_2) and only a tiny true variance, which is the worst case for the
`E[XX^T] - mu mu^T` cancellation.

Expected result:
  * OLD in fp32  -> negative min-eigenvalue, Cholesky FAILS
  * OLD in fp64  -> fine (this is the ground-truth covariance)
  * NEW in fp32  -> fine AND matches OLD-fp64 to ~fp32 precision
"""

import torch

torch.manual_seed(0)

C = 32               # channels in a group
N_PER_BATCH = 16384  # like batch_size(64) * block_size(256)
N_BATCHES = 10       # like bw_iter_calibration_num_batches
MEAN_SCALE = 1.0e3   # large per-channel mean -> forces cancellation
VAR_SCALE = 1.0e-2   # small true std -> tiny eigenvalues


def make_batch(dtype):
    """A batch with a large mean offset and a small, correlated covariance."""
    z = torch.randn(N_PER_BATCH, C, dtype=torch.float64)
    A = torch.randn(C, C, dtype=torch.float64)
    x = z @ A * VAR_SCALE                     # small, correlated variance
    x = x + MEAN_SCALE + torch.randn(C, dtype=torch.float64)  # large per-channel mean
    return x.to(dtype)


# Generate one fixed dataset (in fp64) and reuse the same numbers for every
# estimator/precision so differences are purely numerical, not statistical.
BATCHES_F64 = [make_batch(torch.float64) for _ in range(N_BATCHES)]


def cov_old(dtype):
    """Raw first/second moment accumulation, then cov = sum_xx/n - mu mu^T."""
    sum_x = torch.zeros(C, dtype=dtype)
    sum_xx = torch.zeros(C, C, dtype=dtype)
    n = 0
    for xb in BATCHES_F64:
        x = xb.to(dtype)
        sum_x += x.sum(dim=0)
        sum_xx += x.t() @ x
        n += x.shape[0]
    mu = sum_x / n
    cov = sum_xx / n - torch.outer(mu, mu)
    return 0.5 * (cov + cov.t())


def cov_new(dtype):
    """Welford/Chan centered co-moment accumulation, then cov = M2/n."""
    mean = torch.zeros(C, dtype=dtype)
    M2 = torch.zeros(C, C, dtype=dtype)
    n = 0
    for xb in BATCHES_F64:
        x = xb.to(dtype)
        nb = x.shape[0]
        mean_b = x.mean(dim=0)
        xc = x - mean_b
        m2_b = xc.t() @ xc
        if n == 0:
            mean, M2 = mean_b.clone(), m2_b.clone()
        else:
            n_total = n + nb
            delta = mean_b - mean
            mean = mean + delta * (nb / n_total)
            M2 = M2 + m2_b + torch.outer(delta, delta) * (n * nb / n_total)
        n += nb
    cov = M2 / n
    return 0.5 * (cov + cov.t())


def report(name, cov):
    cov64 = cov.to(torch.float64)
    eig = torch.linalg.eigvalsh(cov64)
    min_eig = eig.min().item()
    max_eig = eig.max().item()
    try:
        torch.linalg.cholesky(cov.to(torch.float64) if cov.dtype == torch.float64 else cov)
        chol = "OK"
    except RuntimeError:
        chol = "FAILED (not positive definite)"
    print(f"{name:28s}  min_eig={min_eig:+.3e}  max_eig={max_eig:.3e}  "
          f"cond={max_eig/abs(min_eig) if min_eig != 0 else float('inf'):.2e}  "
          f"cholesky={chol}")
    return cov64


print("=" * 100)
print(f"Setup: {N_BATCHES} batches x {N_PER_BATCH} samples, C={C}, "
      f"mean~{MEAN_SCALE:g}, true std~{VAR_SCALE:g}")
print(f"  -> E[XX^T] entries ~{MEAN_SCALE**2:g}, true variance ~{VAR_SCALE**2:g}: "
      f"ratio ~{(MEAN_SCALE**2)/(VAR_SCALE**2):.0e} (fp32 has ~7 digits)")
print("=" * 100)

gt = report("OLD raw-moments  fp64", cov_old(torch.float64))   # ground truth
old32 = report("OLD raw-moments  fp32", cov_old(torch.float32))
new64 = report("NEW centered-M2  fp64", cov_new(torch.float64))
new32 = report("NEW centered-M2  fp32", cov_new(torch.float32))

print("-" * 100)
print("Differences vs OLD-fp64 (the ground-truth covariance):")
print(f"  || OLD-fp32 - OLD-fp64 || / ||OLD-fp64|| = "
      f"{(old32 - gt).norm().item() / gt.norm().item():.3e}   (broken)")
print(f"  || NEW-fp32 - OLD-fp64 || / ||OLD-fp64|| = "
      f"{(new32 - gt).norm().item() / gt.norm().item():.3e}   (fixed, ~fp32 eps)")
print(f"  || NEW-fp64 - OLD-fp64 || / ||OLD-fp64|| = "
      f"{(new64 - gt).norm().item() / gt.norm().item():.3e}   (mathematically identical)")
print("=" * 100)
