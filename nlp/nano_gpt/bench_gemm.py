"""Time BW's activation matrix products (N = 64 * 256 tokens, C = 384) in the
layouts and call forms cuBLAS could be given.

Run: python bench_gemm.py   (needs a GPU)
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
N, C = 64 * 256, 384
d = "cuda"
X = torch.randn(N, C, device=d)
g = torch.randn(N, C, device=d)
W = torch.randn(C, C, device=d)
Wt = W.t().contiguous().t()  # same values, column-major
b = torch.randn(C, device=d)
out = torch.empty(N, C, device=d)
Xt = X.t().contiguous()
gt = g.t().contiguous()
ones = torch.ones(N, device=d)
csum = torch.compile(lambda A: A.sum(0))


def bench(fn, reps=200):
    for _ in range(10):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3


cases = {
    "X @ W (row-major W)": lambda: X @ W,
    "X @ W (column-major W)": lambda: X @ Wt,
    "addmm(b, X, W) row-major W": lambda: torch.addmm(b, X, W),
    "addmm(b, X, W) column-major W": lambda: torch.addmm(b, X, Wt),
    "mm(X, W, out=) then add_(b)": lambda: torch.mm(X, W, out=out).add_(b),
    "out.addmm_(X, W) (accumulate)": lambda: out.addmm_(X, W),
    "out.addmm_(X, W) column-major W": lambda: out.addmm_(X, Wt),
    "X^T X (Gram, X row-major)": lambda: X.t() @ X,
    "Xt @ Xt^T (Gram, X stored transposed)": lambda: Xt @ Xt.t(),
    "g^T X": lambda: g.t() @ X,
    "gt @ X (g stored transposed)": lambda: gt @ X,
    "[g X]^T [g X] one product, 768 wide": lambda: (lambda Y: Y.t() @ Y)(torch.cat([g, X], 1)),
    "bf16 X @ W": lambda: X.bfloat16() @ W.bfloat16(),
    "X.sum(0)": lambda: X.sum(0),
    "torch.mv(X^T, ones)": lambda: torch.mv(X.t(), ones),
    "ones @ X (1 x N product)": lambda: ones[None] @ X,
    "X.view(64, 256, C).sum((0, 1))": lambda: X.view(64, 256, C).sum((0, 1)),
    "X.view(256, 64, C).sum(1).sum(0)": lambda: X.view(256, 64, C).sum(1).sum(0),
    "compiled X.sum(0)": lambda: csum(X),
}
for name, fn in cases.items():
    print(f"{bench(fn):7.1f} us  {name}")
print(f"{bench(lambda: torch.cat([g, X], 1)):7.1f} us  (cat alone)")
flops = 2 * N * C * C
print(f"ideal at 400 TFLOP/s: {flops / 400e12 * 1e6:.1f} us per product")
