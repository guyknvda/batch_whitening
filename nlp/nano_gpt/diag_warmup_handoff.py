"""running_mean's activations explode by ~40x starting exactly at iteration 250.

That is the iteration where train.py clears cov_warmup, and for running_mean two
things flip at once:

  1. the whitening covariance switches from the near-diagonal warmup blend
     0.9*diag(cov) + 0.1*cov  to the full blend  0.9*running_cov + 0.1*cov,
     while running_cov itself still holds a ~90%-diagonal matrix (the warmup
     branch writes it with copy_, so no history has accumulated);
  2. the 1/momentum gradient boost switches on, because it is gated on
     `not cov_warmup`.

running_mean is the ONLY mode that crosses this boundary: leave_one_out bypasses
warmup entirely, and calibrated has it disabled from iteration 0. So the
boundary -- not staleness -- is the prime suspect, and the two halves need
separating.

Variants:
  baseline        warmup until 250, boost on after 250   (current behaviour)
  no_warmup       warmup off from iter 0, boost on from iter 0
  no_boost        warmup until 250, boost never          (scale = 1)
  neither         warmup off from iter 0, boost never

Run (from the repo root): sbatch nlp/sbatch_diag_warmup_handoff.sh
"""
import math
import os
import pickle

import numpy as np
import torch

import model as M
from model import GPT, GPTConfig

BATCH, BLOCK, GROUP_SIZE = 64, 256, 16
N_ITERS = 700
PROBE_AT = [100, 200, 249, 251, 260, 275, 300, 350, 400, 500, 600, 699]
LR, MIN_LR, WARMUP = 1e-3, 1e-4, 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'data', 'shakespeare_char')

train_data = np.memmap(os.path.join(DATA, 'train.bin'), dtype=np.uint16, mode='r')
with open(os.path.join(DATA, 'meta.pkl'), 'rb') as f:
    VOCAB = pickle.load(f)['vocab_size']


def get_batch():
    ix = torch.randint(len(train_data) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(train_data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train_data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def lr_at(it):
    if it < WARMUP:
        return LR * it / max(WARMUP, 1)
    r = (it - WARMUP) / (N_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


def run(tag, warmup, boost):
    torch.manual_seed(1337)
    M.STAT_GRAD_SCALE = None if boost else 1.0
    cfg = GPTConfig(
        block_size=BLOCK, vocab_size=VOCAB, n_layer=6, n_head=6, n_embd=384,
        dropout=0.0, attn_dropout=0.0, mlp_dropout=0.0, resid_dropout=0.0,
        bias=False, normalization_type='full_bw',
        batch_center_mode='running_mean', bw_group_size=GROUP_SIZE,
    )
    model = GPT(cfg).to(DEVICE)
    opt = model.configure_optimizers(0.1, LR, (0.9, 0.99), DEVICE.split(':')[0])
    BW = list(model.bw_layers)
    if not warmup:
        model.set_bw_cov_warmup(False)

    cap = []

    def hook(mod, inp, out):
        Xi = inp[0].detach().float()
        Yg = out.detach().float().view(BATCH, BLOCK, mod.num_groups, mod.group_size)
        n = mod.num_groups * mod.group_size
        # per-channel residual mean in sd units (normalise by sqrt(n): the
        # earlier run divided a 384-dim norm by a scalar sd and inflated it ~20x)
        sd = Yg.std(dim=(0, 1)).pow(2).mean().sqrt().clamp_min(1e-12)
        off = Yg.mean(dim=(0, 1)).norm() / (sd * n ** 0.5)
        cap.append((Xi.pow(2).mean().sqrt().item(), sd.item(), off.item()))

    rows = {}
    model.train()
    for it in range(N_ITERS):
        if warmup and it == 250:
            model.set_bw_cov_warmup(False)
        for g in opt.param_groups:
            g['lr'] = lr_at(it)
        if it in PROBE_AT:
            cap.clear()
            hs = [l.register_forward_hook(hook) for l in BW]
            with torch.no_grad():
                X, Y = get_batch()
                _, lp = model(X, targets=Y)
            for h in hs:
                h.remove()
            rows[it] = (lp.item(), np.array(cap))
        X, Y = get_batch()
        _, loss = model(X, targets=Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it % 200 == 0:
            print(f"  [{tag}] iter {it} loss {loss.item():.4f}", flush=True)
    M.STAT_GRAD_SCALE = None
    return rows


VAR = [('baseline', True, True), ('no_warmup', False, True),
       ('no_boost', True, False), ('neither', False, False)]
res = {}
for tag, w, b in VAR:
    print(f"\n### {tag}  (warmup={w}, boost={b})", flush=True)
    res[tag] = run(tag, w, b)

for label, idx in [("INPUT ACTIVATION RMS (runaway?)", 0),
                   ("OUTPUT SCALE rms(sd(Y)) (want ~1)", 1),
                   ("PER-CHANNEL RESIDUAL MEAN in sd", 2)]:
    print(f"\n{'=' * 72}\n{label}")
    print(f"{'iter':>5} | " + " | ".join(f"{t:>10}" for t, _, _ in VAR))
    for it in PROBE_AT:
        print(f"{it:>5} | " + " | ".join(
            f"{res[t][it][1][:, idx].mean():10.4f}" for t, _, _ in VAR))

print(f"\n{'=' * 72}\nLOSS")
print(f"{'iter':>5} | " + " | ".join(f"{t:>10}" for t, _, _ in VAR))
for it in PROBE_AT:
    print(f"{it:>5} | " + " | ".join(f"{res[t][it][0]:10.4f}" for t, _, _ in VAR))
