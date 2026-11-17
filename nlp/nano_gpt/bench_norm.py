"""GPU time and CPU launch cost of the per-token normalization after BW,
(N, C) = (64 * 256, 384): native LayerNorm against an RMS scaling of an
already-centered input (compiled, and a raw Triton kernel).

Run: python bench_norm.py   (needs a GPU)
"""
import time

import torch
import triton
import triton.language as tl

N, C, eps = 64 * 256, 384, 1e-5
d = "cuda"
Y = torch.randn(N, C, device=d)
Y = Y - Y.mean(1, keepdim=True)
dZ = torch.randn(N, C, device=d)


@triton.jit
def _rms_fwd(Y, Z, RSTD, n_rows, C, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < C)
    off = rows[:, None] * C + cols[None, :]
    y = tl.load(Y + off, mask=mask, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(y * y, 1) / C + eps)
    tl.store(Z + off, y * rstd[:, None], mask=mask)
    tl.store(RSTD + rows, rstd, mask=rows < n_rows)


@triton.jit
def _rms_bwd(DZ, Z, RSTD, H, n_rows, C, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < C)
    off = rows[:, None] * C + cols[None, :]
    dz = tl.load(DZ + off, mask=mask, other=0.0)
    z = tl.load(Z + off, mask=mask, other=0.0)
    rstd = tl.load(RSTD + rows, mask=rows < n_rows, other=0.0)
    m = tl.sum(dz * z, 1) / C
    tl.store(H + off, rstd[:, None] * (dz - z * m[:, None]), mask=mask)


def tri_fwd(Y, rows=4):
    Z, rstd = torch.empty_like(Y), torch.empty(Y.size(0), device=Y.device)
    _rms_fwd[(triton.cdiv(Y.size(0), rows),)](Y, Z, rstd, Y.size(0), Y.size(1), eps, ROWS=rows,
                                              BLOCK=triton.next_power_of_2(Y.size(1)))
    return Z, rstd


def tri_bwd(dZ, Z, rstd, rows=4):
    H = torch.empty_like(dZ)
    _rms_bwd[(triton.cdiv(dZ.size(0), rows),)](dZ, Z, rstd, H, dZ.size(0), dZ.size(1), ROWS=rows,
                                               BLOCK=triton.next_power_of_2(dZ.size(1)))
    return H


def c_fwd(Y):
    rstd = torch.rsqrt(Y.square().mean(1, keepdim=True) + eps)
    return Y * rstd, rstd


def c_bwd(dZ, Z, rstd):
    return rstd * (dZ - Z * (dZ * Z).mean(1, keepdim=True))


c_fwd, c_bwd = torch.compile(c_fwd), torch.compile(c_bwd)


def bench(fn, reps=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    gpu = s.elapsed_time(e) / reps * 1e3
    torch.cuda._sleep(1_000_000_000)
    t = time.perf_counter()
    for _ in range(20):
        fn()
    cpu = (time.perf_counter() - t) / 20 * 1e6
    torch.cuda.synchronize()
    return gpu, cpu


Zn, mean, rstd_n = torch.native_layer_norm(Y, [C], None, None, eps)
Zt, rstd_t = tri_fwd(Y)
Zc, rstd_c = c_fwd(Y)
print("max |diff| triton vs native:", (Zt - Zn).abs().max().item(),
      (tri_bwd(dZ, Zt, rstd_t) - torch.ops.aten.native_layer_norm_backward(
          dZ, Y, [C], mean, rstd_n, None, None, [True, False, False])[0]).abs().max().item())
cases = {
    "native forward": lambda: torch.native_layer_norm(Y, [C], None, None, eps),
    "native backward": lambda: torch.ops.aten.native_layer_norm_backward(dZ, Y, [C], mean, rstd_n, None, None,
                                                                         [True, False, False]),
    "compiled RMS forward": lambda: c_fwd(Y),
    "compiled RMS backward": lambda: c_bwd(dZ, Zc, rstd_c),
}
for r in (1, 2, 4, 8):
    cases[f"Triton RMS forward, {r} rows/program"] = lambda r=r: tri_fwd(Y, r)
    cases[f"Triton RMS backward, {r} rows/program"] = lambda r=r: tri_bwd(dZ, Zt, rstd_t, r)
for name, fn in cases.items():
    gpu, cpu = bench(fn)
    print(f"GPU {gpu:6.1f} us  CPU {cpu:6.1f} us  {name}")
