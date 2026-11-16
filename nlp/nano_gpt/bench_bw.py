"""Time one BW layer (forward + backward) at nanoGPT size, per variant, and profile it.

Run: python bench_bw.py   (needs a GPU)
"""
import torch
from torch.profiler import ProfilerActivity, profile

from model import BatchWhiteningBlock, LayerNorm

torch.backends.cuda.matmul.allow_tf32 = True
B, T, C = 64, 256, 384


def make(**kw):
    bw = BatchWhiteningBlock(C, momentum=0.7, center_mode="running_mean", group_size=C, fix_factor=0.8,
                             renorm=True, renorm_stats="batch", **kw).cuda()
    bw.cov_warmup = False
    return bw


def step(layer, x):
    y = layer(x)
    y.square().mean().backward()


def time_ms(layer, n=100):
    x = torch.randn(B, T, C, device="cuda", requires_grad=True)
    for _ in range(10):
        step(layer, x)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        step(layer, x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


variants = {
    "LayerNorm": LayerNorm(C, bias=False).cuda(),
    "BW, no cache": make(renorm_cache=False),
    "BW, cache": make(renorm_cache=True),
    "BW, cache, fused backward": make(renorm_cache=True, renorm_fused=True),
    "BW, cache, stat_grad=mean": make(renorm_cache=True, stat_grad="mean"),
    "BW, cache, stat_grad=none": make(renorm_cache=True, stat_grad="none"),
}
for name, layer in variants.items():
    print(f"{name:28s} {time_ms(layer):6.3f} ms / layer (fwd+bwd)")

layer = variants["BW, cache, fused backward"]
x = torch.randn(B, T, C, device="cuda", requires_grad=True)
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        step(layer, x)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
