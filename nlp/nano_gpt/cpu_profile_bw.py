"""CPU (Python and launch) cost of one fast-path BW layer, forward and backward,
issued while the GPU is kept busy so no call waits for it.

Run: python cpu_profile_bw.py   (needs a GPU)
"""
import time

import torch

from model import BatchWhiteningBlock

torch.backends.cuda.matmul.allow_tf32 = True
B, T, C = 64, 256, 384
bw = BatchWhiteningBlock(C, momentum=0.7, center_mode="running_mean", group_size=C, fix_factor=0.8,
                         renorm=True, renorm_stats="batch", renorm_fused=True, fast=True,
                         side_stream=True).cuda()
bw.cov_warmup = False
bw.post_norm_eps = 1e-5
X = torch.randn(B, T, C, device="cuda", requires_grad=True)
g = torch.randn(B, T, C, device="cuda")


def fwd_bwd():
    bw(X).backward(g)


for _ in range(10):
    fwd_bwd()
torch.cuda.synchronize()

reps = 20
torch.cuda._sleep(3_000_000_000)
t = time.perf_counter()
for _ in range(reps):
    Y = bw(X)
fwd = (time.perf_counter() - t) / reps * 1e3
torch.cuda.synchronize()
Y = bw(X)
torch.cuda.synchronize()
torch.cuda._sleep(3_000_000_000)
t = time.perf_counter()
Y.backward(g)
bwd = (time.perf_counter() - t) * 1e3
torch.cuda.synchronize()
print(f"CPU per layer: forward {fwd:.3f} ms, backward {bwd:.3f} ms")

torch.cuda._sleep(3_000_000_000)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    for _ in range(reps):
        Y = bw(X)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
Y = bw(X)
torch.cuda.synchronize()
torch.cuda._sleep(3_000_000_000)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    Y.backward(g)
torch.cuda.synchronize()
print("backward:")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
