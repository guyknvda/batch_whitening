"""How well would BW, with statistics from small batches, condition the MLP input?

Usage: python diag_whitening_quality.py <ckpt.pt> <data.bin> <batch_size> [num_seqs=256]

Runs a trained model on num_seqs sequences and keeps per-sequence first and
second moments of the residual stream entering each block's bw_2 (the BW input;
Identity in a LayerNorm model). The covariance over all sequences is the
reference. For each group size, fix_factor and momentum it then estimates the
covariance the way BW does in training (one batch of batch_size sequences, or
an EMA over consecutive batches), whitens with it, and reports the condition
number of the whitened reference covariance next to per-channel
standardization (no whitening) and to whitening with the reference itself.
No training; one forward pass per batch.
"""
import math
import sys

import numpy as np
import torch

from model import GPT, GPTConfig, fix_cov

ckpt_path, data_path, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
num_seqs = int(sys.argv[4]) if len(sys.argv) > 4 else 256
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig(**ckpt["model_args"]))
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
model.load_state_dict(sd)
model.to(device).eval()
T = model.config.block_size
C = model.config.n_embd
n_layer = model.config.n_layer

data = np.memmap(data_path, dtype=np.uint16, mode="r")
starts = torch.randint(len(data) - T, (num_seqs,))

S = torch.zeros(n_layer, num_seqs, C, dtype=torch.float64, device=device)
Gm = torch.zeros(n_layer, num_seqs, C, C, dtype=torch.float64, device=device)
hooks, cur = [], {}
for li, block in enumerate(model.transformer.h):
    hooks.append(block.bw_2.register_forward_pre_hook(
        lambda mod, inp, li=li: cur.__setitem__(li, inp[0].detach().double())))

chunk = 16
with torch.no_grad():
    for a in range(0, num_seqs, chunk):
        idx = torch.stack([torch.from_numpy(data[s:s + T].astype(np.int64)) for s in starts[a:a + chunk]]).to(device)
        model(idx, targets=idx)
        for li in range(n_layer):
            x = cur[li]
            S[li, a:a + len(x)] = x.sum(dim=1)
            Gm[li, a:a + len(x)] = torch.einsum("btc,btd->bcd", x, x)


def cov_of(li, seqs):
    n = len(seqs) * T
    mu = S[li, seqs].sum(0) / n
    return Gm[li, seqs].sum(0) / n - torch.outer(mu, mu)


def whitener(cov, gs, ff):
    ng = C // gs
    blocks = cov.view(ng, gs, ng, gs).diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # (ng, gs, gs)
    blocks = blocks + 1e-6 * blocks.diagonal(dim1=-2, dim2=-1).mean() * torch.eye(gs, device=device, dtype=cov.dtype)
    L = torch.linalg.cholesky(fix_cov(blocks, ff))
    Winv = torch.linalg.solve_triangular(L, torch.eye(gs, device=device, dtype=cov.dtype).expand(ng, gs, gs), upper=False)
    return torch.block_diag(*Winv)


def kappa(O):
    e = torch.linalg.eigvalsh(0.5 * (O + O.T))
    return (e[-1] / e[0].clamp_min(1e-12)).item()


GS = [g for g in (16, 64, 128, 192, 384, 768) if C % g == 0 and g <= C]
FF = [0.4, 0.6, 0.8, 0.9, 1.0]
MOM = [1.0, 0.7, 0.3, 0.1]
n_batches = num_seqs // B
res = {}
for li in range(n_layer):
    ref = cov_of(li, list(range(num_seqs)))
    d = ref.diagonal().rsqrt()
    k_std = kappa(d[:, None] * ref * d[None, :])
    res.setdefault("std", []).append(k_std)
    for gs in GS:
        for ff in FF:
            res.setdefault(("oracle", gs, ff), []).append(kappa(whitener(ref, gs, ff) @ ref @ whitener(ref, gs, ff).T))
            for m in MOM:
                ks = []
                ema = None
                for b in range(n_batches):
                    c = cov_of(li, list(range(b * B, (b + 1) * B)))
                    ema = c if ema is None else (1 - m) * ema + m * c
                    if b >= n_batches // 2:
                        W = whitener(ema, gs, ff)
                        ks.append(kappa(W @ ref @ W.T))
                res.setdefault((gs, ff, m), []).append(float(np.median(ks)))

gm = lambda v: math.exp(np.mean(np.log(v)))
print(f"\n{ckpt_path}: B={B}, T={T}, C={C}, {n_layer} layers, {num_seqs} sequences")
print("condition number of the whitened true covariance, geometric mean over layers (lower is better)")
print(f"per-channel standardization (no whitening): {gm(res['std']):.1f}   per layer: "
      + " ".join(f"{k:.0f}" for k in res["std"]))
for gs in GS:
    print(f"\ngroup {gs}")
    print("  fix_factor  oracle  " + "  ".join(f"m={m:<4}" for m in MOM))
    for ff in FF:
        print(f"  {ff:<10}  {gm(res[('oracle', gs, ff)]):6.1f}  "
              + "  ".join(f"{gm(res[(gs, ff, m)]):6.1f}" for m in MOM))
