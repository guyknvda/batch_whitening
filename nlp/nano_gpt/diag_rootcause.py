"""Why is running_mean bad, when drift alone is far too small to explain it?

Hypothesis under test (the centering asymmetry):

  The data is centered with  m_b = (1-mom)*running_mean + mom*mean_other, which
  is 90% a stale EMA, while the covariance is built about the *fresh* LOO mean,
  so it estimates Sigma and carries no information about the centering error.
  The whitened output therefore keeps a residual mean

      L^-1 * d,      d = mu_t - m_b  ~  0.9 * (mu_t - running_mean)

  which the covariance never removes. A BW layer that emits a large, systematic,
  time-varying mean is not normalising -- it is injecting a bias. And because
  that bias enters the residual stream and is re-read by the next layer, it can
  compound: offset -> activation mean grows -> EMA lags a growing target ->
  larger offset.

leave_one_out has d ~ 0 by construction (m_b IS mean_other), and calibrated
recomputes its mean at the current weights every iteration, so neither should
show the effect. That makes this differential, not just descriptive.

Measured per BW layer, per mode:
  offset   ||mean(Y)|| / rms(sd(Y))   residual mean of the whitened output, in sd
  scale    rms(sd(Y))                 is the output actually unit-scale?
  bufErr   ||running_mean - fresh||/||fresh||
  actNorm  rms of the layer INPUT      does the activation scale run away?

Run (from the repo root): sbatch nlp/sbatch_diag_rootcause.sh
"""
import math
import os
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig

BATCH, BLOCK = 64, 256
GROUP_SIZE = 16
N_ITERS = 700
PROBE_AT = [100, 200, 251, 300, 400, 500, 600, 699]
CALIB_BATCHES = 10
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


def fresh_loo_mean(X, ng, gs):
    B, T, _ = X.shape
    Xg = X.view(B, T, ng, gs).float()
    tot = Xg.sum(dim=(0, 1), keepdim=True)
    per = Xg.sum(dim=1, keepdim=True)
    return ((tot - per) / ((B - 1) * T)).mean(dim=(0, 1))       # (ng,gs)


def run_mode(mode):
    torch.manual_seed(1337)
    cfg = GPTConfig(
        block_size=BLOCK, vocab_size=VOCAB, n_layer=6, n_head=6, n_embd=384,
        dropout=0.0, attn_dropout=0.0, mlp_dropout=0.0, resid_dropout=0.0,
        bias=False, normalization_type='full_bw', batch_center_mode=mode,
        bw_group_size=GROUP_SIZE,
    )
    model = GPT(cfg).to(DEVICE)
    opt = model.configure_optimizers(0.1, LR, (0.9, 0.99), DEVICE.split(':')[0])
    BW = list(model.bw_layers)
    calib = (mode == 'calibrated')
    if calib:                      # train.py disables warmup for calibrated
        model.set_bw_cov_warmup(False)

    def do_calib():
        xs = [get_batch()[0] for _ in range(CALIB_BATCHES)]
        model.bw_calibration_reset()
        model.bw_calibration_enable(True)
        with torch.no_grad():
            for x in xs:
                model(x)
        model.bw_calibration_enable(False)
        model.bw_calibration_commit()
        model.set_bw_update_running_stats(False)
        model.set_bw_update_running_cov_stats(False)

    if calib:
        do_calib()

    cap = []

    def hook(mod, inp, out):
        Xi = inp[0].detach().float()
        Y = out.detach().float()
        B, T, C = Y.shape
        ng, gs = mod.num_groups, mod.group_size
        Yg = Y.view(B, T, ng, gs)
        ym = Yg.mean(dim=(0, 1))                                # (ng,gs)
        ysd = Yg.std(dim=(0, 1))
        rms_sd = ysd.pow(2).mean().sqrt()
        fm = fresh_loo_mean(Xi, ng, gs)
        rb = mod.running_mean.detach().squeeze(0).float()
        cap.append((
            (ym.norm() / rms_sd.clamp_min(1e-12)).item(),        # offset in sd
            rms_sd.item(),                                       # output scale
            ((rb - fm).norm() / fm.norm().clamp_min(1e-12)).item(),
            Xi.pow(2).mean().sqrt().item(),                      # input rms
        ))

    rows = {}
    model.train()
    for it in range(N_ITERS):
        if it == 250 and not calib:
            model.set_bw_cov_warmup(False)
        for g in opt.param_groups:
            g['lr'] = lr_at(it)
        if calib:
            do_calib()

        if it in PROBE_AT:
            cap.clear()
            hs = [l.register_forward_hook(hook) for l in BW]
            with torch.no_grad():
                X, Y = get_batch()
                _, lp = model(X, targets=Y)
            for h in hs:
                h.remove()
            a = np.array(cap)                                    # (13,4)
            rows[it] = (lp.item(), a)

        X, Y = get_batch()
        _, loss = model(X, targets=Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it % 100 == 0:
            print(f"  [{mode}] iter {it} loss {loss.item():.4f}", flush=True)
    return rows


results = {}
for mode in ('leave_one_out', 'running_mean', 'calibrated'):
    print(f"\n### {mode}", flush=True)
    results[mode] = run_mode(mode)

print("\n" + "=" * 78)
print("RESIDUAL MEAN OF THE WHITENED OUTPUT, in units of its own sd")
print("(a correctly normalising layer emits ~0; values >1 mean the layer's")
print(" output is dominated by a systematic bias rather than by signal)")
print("=" * 78)
print(f"{'iter':>5} {'loss':>7} | " + " | ".join(f"{m[:12]:>12}" for m in results))
for it in PROBE_AT:
    cells = []
    for m in results:
        a = results[m][it][1]
        cells.append(f"{a[:, 0].mean():12.3f}")
    ls = " ".join(f"{results[m][it][0]:7.3f}" for m in results)
    print(f"{it:>5} {ls} | " + " | ".join(cells))

for label, idx, note in [
    ("OUTPUT SCALE  rms(sd(Y))", 1, "should sit near 1"),
    ("RUNNING-MEAN ERROR  ||buf-fresh||/||fresh||", 2, "0 = buffer is current"),
    ("INPUT ACTIVATION RMS", 3, "growth => runaway feedback"),
]:
    print(f"\n{label}   ({note})")
    print(f"{'iter':>5} | " + " | ".join(f"{m[:12]:>12}" for m in results))
    for it in PROBE_AT:
        cells = [f"{results[m][it][1][:, idx].mean():12.4f}" for m in results]
        print(f"{it:>5} | " + " | ".join(cells))

print("\nPER-LAYER residual mean at the last probe (depth dependence)")
print(f"{'layer':>5} | " + " | ".join(f"{m[:12]:>12}" for m in results))
last = PROBE_AT[-1]
for i in range(len(results['running_mean'][last][1])):
    cells = [f"{results[m][last][1][i, 0]:12.3f}" for m in results]
    print(f"{i:>5} | " + " | ".join(cells))
