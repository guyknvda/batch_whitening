"""Small-matrix (G = 384) building blocks of BW on the GPU, and the precision of
the covariance from an uncentered TF32 Gram matrix.

Run: python bench_linalg.py   (needs a GPU)
"""
import torch

from model import ns_whiten

G, N = 384, 64 * 256
d = "cuda"
torch.manual_seed(0)


def bench(fn, reps=50):
    for _ in range(5):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3


X = torch.randn(N, G, device=d) @ (torch.randn(G, G, device=d) * 0.05 + torch.eye(G, device=d))
A = torch.cov(X.T) * 0.8 + 0.2 * torch.diag(torch.cov(X.T).diagonal())
B = torch.randn(G, G, device=d)
I = torch.eye(G, device=d)
L = torch.linalg.cholesky(A)
for tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"--- TF32 {tf32}")
    print(f"{bench(lambda: A @ B):8.1f} us  one 384^3 product")
    print(f"{bench(lambda: ns_whiten(A, 10)):8.1f} us  coupled Newton-Schulz, 10 iterations (eager)")
    print(f"{bench(lambda: ns_whiten(A, 5)):8.1f} us  coupled Newton-Schulz, 5 iterations (eager)")
torch.backends.cuda.matmul.allow_tf32 = True
for lib in ("cusolver", "magma"):
    torch.backends.cuda.preferred_linalg_library(lib)
    print(f"--- {lib}")
    print(f"{bench(lambda: torch.linalg.cholesky_ex(A)):8.1f} us  cholesky_ex")
    print(f"{bench(lambda: torch.linalg.solve_triangular(L, I, upper=False)):8.1f} us  solve_triangular(L, I)")
    print(f"{bench(lambda: torch.linalg.inv_ex(L)):8.1f} us  inv_ex(L)")
    print(f"{bench(lambda: torch.linalg.eigh(A), reps=10):8.1f} us  eigh fp32")
    print(f"{bench(lambda: torch.linalg.eigh(A.double()), reps=10):8.1f} us  eigh fp64")
torch.backends.cuda.preferred_linalg_library("default")

# NS graph replay: launch overhead removed
g = torch.cuda.CUDAGraph()
A_static = A.clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        ns_whiten(A_static, 10)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = ns_whiten(A_static, 10)
print(f"{bench(g.replay):8.1f} us  Newton-Schulz 10 iterations, CUDA graph replay")

# Precision: covariance of activations with sizeable channel means
print("--- covariance precision (relative to float64, centered)")
for mean_scale in (0.0, 1.0, 3.0):
    Xm = X + mean_scale * torch.randn(G, device=d)
    ref = torch.cov(Xm.double().T, correction=0)
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        mu = Xm.sum(0) / N
        unc = Xm.T @ Xm / N - torch.outer(mu, mu)
        Xc = Xm - mu
        cen = Xc.T @ Xc / N
        e1 = ((unc.double() - ref).norm() / ref.norm()).item()
        e2 = ((cen.double() - ref).norm() / ref.norm()).item()
        Ar = ref * 0.8 + 0.2 * torch.diag(ref.diagonal())
        Wr = torch.linalg.inv(torch.linalg.cholesky(Ar))
        Au = unc.double() * 0.8 + 0.2 * torch.diag(unc.double().diagonal())
        Wu = torch.linalg.inv(torch.linalg.cholesky(Au))
        print(f"mean scale {mean_scale}, TF32 {tf32}: cov err uncentered {e1:.1e}, centered {e2:.1e}; "
              f"whitening matrix err (uncentered) {((Wu - Wr).norm() / Wr.norm()).item():.1e}")
