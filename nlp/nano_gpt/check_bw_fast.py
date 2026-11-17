"""Check bw_fast (optionally with the side stream) against the reference BW path on the GPU.

Run: python check_bw_fast.py [cpu]   (default: GPU)
"""
import copy
import sys

import torch

from model import BatchWhiteningBlock

torch.manual_seed(0)
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
B, T, C = (64, 256, 384) if dev == "cuda" else (4, 32, 48)
for (whiten, iters), post_eps in [(w, e) for w in (("cholesky", 5), ("ns", 10), ("ns_chol", 10)) for e in (None, 1e-5)]:
    ref = BatchWhiteningBlock(C, momentum=0.7, center_mode="running_mean", group_size=C, fix_factor=0.8,
                              renorm=True, renorm_stats="batch", renorm_fused=True, whiten=whiten,
                              ns_iters=iters).to(dev)
    ref.cov_warmup = False
    ref.post_norm_eps = post_eps
    with torch.no_grad():
        ref.gamma.uniform_(0.5, 1.5)
        ref.beta.normal_()
    fast = copy.deepcopy(ref)
    fast.fast, fast.side_stream = True, dev == "cuda"
    Amix = torch.randn(C, C, device=dev) * 0.05 + torch.eye(C, device=dev)
    worst = 0.0
    for step in range(20):
        X = (torch.randn(B, T, C, device=dev) @ Amix + 0.1).requires_grad_()
        g = torch.randn(B, T, C, device=dev)
        outs = []
        for layer in (ref, fast):
            Y = layer(X)
            outs.append((Y,) + torch.autograd.grad(Y, (X, layer.gamma, layer.beta), g))
        for a, b in zip(*outs):
            worst = max(worst, ((a - b).norm() / b.norm()).item())
        worst = max(worst, ((fast.running_cov - ref.running_cov).norm() / ref.running_cov.norm()).item())
    print(f"{whiten:9s} post_eps {post_eps} fast vs reference, 20 steps, max rel err {worst:.1e}")
