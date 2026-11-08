"""Is `running_mean`'s problem staleness, or is it just noise?

The two competing explanations for why running_mean underperforms make opposite,
testable predictions about the fresh per-iteration statistics F(t) (the
leave-one-out mean/cov of each BW layer's input at the current weights):

  "pooling helps"  -- F(t) is a stationary population statistic plus sampling
                      noise. Averaging ~1/momentum of them is then strictly
                      better than any single one, and the EMA buffer B(t) should
                      be *closer* to the truth than F(t) is.

  "stats are stale" -- F(t) drifts because the weights move, so the EMA buffer
                      is biased toward what the network looked like several
                      iterations ago.

These are separable. Drift and noise add in quadrature, so with

    noise   = E|| F(t) - F'(t) ||      two INDEPENDENT batches, SAME weights
    D(k)    = E|| F(t) - F(t-k) ||     same batch distribution, k weight updates apart

we get  drift(k) = sqrt(max(D(k)^2 - noise^2, 0)).  If drift(k) ~ 0 for all k
the statistics are stationary and pooling is free. If drift(k) grows with k,
they are not, and the EMA's effective window imports exactly that much bias.

Two further diagnostics:
  * bestLag = argmin_k E|| B(t) - F(t-k) ||. A buffer that is merely noisy is
    closest to the *current* statistics (bestLag = 0). A stale buffer is
    closest to some past network (bestLag > 0). This is a signature noise
    cannot fake.
  * the variance/bias tradeoff: pooling n batches cuts noise to noise/sqrt(n)
    but incurs drift(n). The window minimising total error is the one the
    momentum *should* correspond to; compare it against 1/momentum.

Run (from the repo root): sbatch nlp/sbatch_diag_staleness.sh
"""
import math
import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig

# ----------------------------------------------------------------- config
BATCH, BLOCK = 64, 256
MOMENTUM = 0.1
GROUP_SIZE = 16
N_ITERS = 900           # training iterations to run
WARM_SKIP = 300         # ignore the warmup/transient phase in the statistics
K_MAX = 40              # largest lag to probe
LR, MIN_LR, WARMUP = 1e-3, 1e-4, 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'data', 'shakespeare_char')

torch.manual_seed(1337)

train_data = np.memmap(os.path.join(DATA, 'train.bin'), dtype=np.uint16, mode='r')
with open(os.path.join(DATA, 'meta.pkl'), 'rb') as f:
    vocab_size = pickle.load(f)['vocab_size']


def get_batch():
    ix = torch.randint(len(train_data) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(train_data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train_data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


cfg = GPTConfig(
    block_size=BLOCK, vocab_size=vocab_size, n_layer=6, n_head=6, n_embd=384,
    dropout=0.0, attn_dropout=0.0, mlp_dropout=0.0, resid_dropout=0.0,
    bias=False, normalization_type='full_bw', batch_center_mode='running_mean',
    bw_group_size=GROUP_SIZE,
)
model = GPT(cfg).to(DEVICE)
opt = model.configure_optimizers(0.1, LR, (0.9, 0.99), DEVICE.split(':')[0])
BW = list(model.bw_layers)
print(f"device={DEVICE}  bw_layers={len(BW)}  groups={BW[0].num_groups}"
      f"  group_size={BW[0].group_size}  momentum={BW[0].momentum}", flush=True)


# ------------------------------------------------- fresh-statistics probe
def loo_stats(X, ng, gs):
    """Leave-one-out mean and covariance of a BW layer's input, exactly as
    model.batch_orthonorm computes them, reduced over the batch."""
    B, T, _ = X.shape
    Xg = X.view(B, T, ng, gs).float()
    tot = Xg.sum(dim=(0, 1), keepdim=True)
    per = Xg.sum(dim=1, keepdim=True)
    m_other = (tot - per) / ((B - 1) * T)                      # (B,1,ng,gs)
    d = Xg.unsqueeze(0) - m_other.unsqueeze(1)
    d = d.masked_fill(torch.eye(B, dtype=torch.bool, device=X.device)
                      .view(B, B, 1, 1, 1), 0.0)
    cov = torch.einsum('bstgc,bstgd->bgcd', d, d) / ((B - 1) * T)
    return m_other.mean(dim=(0, 1)), cov.mean(0)                # (ng,gs),(ng,gs,gs)


_capture = []


def _hook(mod, inp, out):
    m, c = loo_stats(inp[0].detach(), mod.num_groups, mod.group_size)
    _capture.append((m, c))


def fresh_stats():
    """Fresh F at the CURRENT weights, from one independent batch. Buffer
    updates are off; grad stays enabled so the training-mode whitening branch
    is taken (the eval branch uses a different transform)."""
    global _capture
    _capture = []
    model.set_bw_update_running_stats(False)
    model.set_bw_update_running_cov_stats(False)
    hs = [l.register_forward_hook(_hook) for l in BW]
    try:
        X, Y = get_batch()
        model(X, targets=Y)
    finally:
        for h in hs:
            h.remove()
        model.set_bw_update_running_stats(True)
        model.set_bw_update_running_cov_stats(True)
    opt.zero_grad(set_to_none=True)
    return [(m.clone(), c.clone()) for m, c in _capture]


def buffers():
    return [(l.running_mean.detach().squeeze(0).clone().float(),
             l.running_cov.detach().squeeze(0).clone().float()) for l in BW]


def rel(a, b):
    """Relative Frobenius distance, averaged over BW layers."""
    out = []
    for (am, ac), (bm, bc) in zip(a, b):
        out.append(((am - bm).norm() / bm.norm().clamp_min(1e-12)).item())
        out.append(((ac - bc).norm() / bc.norm().clamp_min(1e-12)).item())
    return np.array(out[0::2]).mean(), np.array(out[1::2]).mean()


def lr_at(it):
    if it < WARMUP:
        return LR * it / max(WARMUP, 1)
    r = (it - WARMUP) / (N_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)


# ------------------------------------------------------------- main loop
ring = {}                                   # iter -> F(t)
acc = {'noise': [], 'buf': []}
Dk = {k: {'m': [], 'c': []} for k in range(1, K_MAX + 1)}
BufLag = {k: {'m': [], 'c': []} for k in range(0, K_MAX + 1)}

model.train()
for it in range(N_ITERS):
    # train.py clears the covariance warmup at iter_num == eval_interval
    if it == 250:
        model.set_bw_cov_warmup(False)
    for g in opt.param_groups:
        g['lr'] = lr_at(it)

    if it >= WARM_SKIP:
        B_t = buffers()
        F_t = fresh_stats()
        F_t2 = fresh_stats()                # independent batch, same weights

        acc['noise'].append(rel(F_t, F_t2))
        acc['buf'].append(rel(B_t, F_t))
        for k in range(1, K_MAX + 1):
            if (it - k) in ring:
                dm, dc = rel(F_t, ring[it - k])
                Dk[k]['m'].append(dm)
                Dk[k]['c'].append(dc)
        for k in range(0, K_MAX + 1):
            tgt = F_t if k == 0 else ring.get(it - k)
            if tgt is not None:
                bm, bc = rel(B_t, tgt)
                BufLag[k]['m'].append(bm)
                BufLag[k]['c'].append(bc)
        ring[it] = F_t
        ring.pop(it - K_MAX - 1, None)

    X, Y = get_batch()
    _, loss = model(X, targets=Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if it % 100 == 0:
        print(f"iter {it}  loss {loss.item():.4f}", flush=True)

# ---------------------------------------------------------------- report
nm = np.array([a[0] for a in acc['noise']]).mean()
nc = np.array([a[1] for a in acc['noise']]).mean()
bm = np.array([a[0] for a in acc['buf']]).mean()
bc = np.array([a[1] for a in acc['buf']]).mean()

print("\n================ RESULTS ================")
print(f"samples: {len(acc['noise'])} probe iterations (iters {WARM_SKIP}..{N_ITERS})")
print(f"\nnoise floor  (two independent batches, SAME weights)")
print(f"    mean {nm:.5f}   cov {nc:.5f}")
print(f"buffer error (EMA buffer vs fresh stats at current weights)")
print(f"    mean {bm:.5f}   cov {bc:.5f}")
print(f"    buffer/noise ratio:  mean {bm / nm:.2f}x   cov {bc / nc:.2f}x")

print(f"\nD(k) = ||F(t) - F(t-k)||, and drift after removing the noise floor")
print(f"{'k':>4} {'D_mean':>9} {'drift_mean':>11} {'D_cov':>9} {'drift_cov':>11}")
for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]:
    if not Dk[k]['m']:
        continue
    dm, dc = np.mean(Dk[k]['m']), np.mean(Dk[k]['c'])
    gm = math.sqrt(max(dm ** 2 - nm ** 2, 0.0))
    gc = math.sqrt(max(dc ** 2 - nc ** 2, 0.0))
    print(f"{k:>4} {dm:9.5f} {gm:11.5f} {dc:9.5f} {gc:11.5f}")

print(f"\nbestLag: which past network does the buffer resemble most?")
print(f"{'k':>4} {'||B(t)-F(t-k)||_mean':>22} {'||B(t)-F(t-k)||_cov':>21}")
best_m = min((np.mean(BufLag[k]['m']), k) for k in BufLag if BufLag[k]['m'])
best_c = min((np.mean(BufLag[k]['c']), k) for k in BufLag if BufLag[k]['c'])
for k in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40]:
    if not BufLag[k]['m']:
        continue
    print(f"{k:>4} {np.mean(BufLag[k]['m']):22.5f} {np.mean(BufLag[k]['c']):21.5f}")
print(f"  -> bestLag mean = {best_m[1]} iters, cov = {best_c[1]} iters")
print(f"     (0 => buffer tracks the current network; >0 => it lags)")

print(f"\nbias/variance tradeoff: pooling n batches gives noise/sqrt(n) + drift(n)")
print(f"{'n':>4} {'noise/sqrt(n)':>14} {'drift(n)':>10} {'total_cov':>10}")
best = None
for n in [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]:
    if not Dk.get(n, {}).get('c'):
        continue
    dc = np.mean(Dk[n]['c'])
    gc = math.sqrt(max(dc ** 2 - nc ** 2, 0.0))
    tot = math.sqrt((nc / math.sqrt(n)) ** 2 + gc ** 2)
    print(f"{n:>4} {nc / math.sqrt(n):14.5f} {gc:10.5f} {tot:10.5f}")
    if best is None or tot < best[0]:
        best = (tot, n)
print(f"  -> optimal pooling window ~{best[1]} batches; "
      f"momentum={MOMENTUM} implies {1 / MOMENTUM:.0f}")
