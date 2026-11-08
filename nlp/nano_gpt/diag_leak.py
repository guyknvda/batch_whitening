"""Does a trained BW model read its own future through the batch statistics?

Usage: python diag_leak.py <ckpt.pt> [num_batches] [seqs_per_batch]

For sequence j of a batch, the second half of j alone is replaced with fresh text
and the loss on j's first half is measured in a training-mode forward (batch
statistics). Every other sequence is unchanged, so j's first-half prediction can
only change through the multi-hop leave-one-out path: j's future -> other
sequences' activations (via their statistics) -> j's statistics in a later layer.
A causal model (LayerNorm, or BW in inference mode) gives exactly zero change; a
rise means the model uses its own future tokens.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

ckpt_path = sys.argv[1]
num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seqs_per_batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

ckpt = torch.load(ckpt_path, map_location=device)
args = ckpt["model_args"]
model = GPT(GPTConfig(**args))
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
model.to(device)
model.set_bw_cov_warmup(False)
model.train()
print(f"checkpoint iter {ckpt['iter_num']}, norm {args.get('normalization_type')}, "
      f"mode {args.get('batch_center_mode')}, layers {args['n_layer']}")

data_path = os.environ.get("LEAK_DATA", os.path.join("data", "shakespeare_char", "train.bin"))
data = np.memmap(data_path, dtype=np.uint16, mode="r")
B, T = 64, args["block_size"]
H = T // 2


def batch():
    ix = torch.randint(len(data) - T, (B,))
    return torch.stack([torch.from_numpy(data[i: i + T].astype(np.int64)) for i in ix]).to(device)


def first_half_losses(x, grad):
    """Per-sequence mean loss over the first half (targets inside the first half)."""
    with torch.set_grad_enabled(grad):
        logits, _ = model(x, x)  # targets only to get full-length logits
    ce = F.cross_entropy(logits[:, : H - 1].reshape(-1, logits.size(-1)), x[:, 1:H].reshape(-1),
                         reduction="none")
    return ce.view(B, H - 1).mean(dim=1).detach()


true_l, corrupt_l, eval_l = [], [], []
for _ in range(num_batches):
    x, fresh = batch(), batch()
    lt = first_half_losses(x, True)
    eval_l.append(first_half_losses(x, False)[:seqs_per_batch])
    for j in range(seqs_per_batch):
        xs = x.clone()
        xs[j, H:] = fresh[j, H:]
        true_l.append(lt[j].item())
        corrupt_l.append(first_half_losses(xs, True)[j].item())

t, c = np.array(true_l), np.array(corrupt_l)
d = c - t
print(f"first-half loss (training mode): own future intact {t.mean():.4f}, own future replaced {c.mean():.4f}")
print(f"first-half loss (inference mode): {torch.cat(eval_l).mean().item():.4f}")
print(f"replaced - intact: {d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (SE, n={len(d)}) "
      f"= {100 * d.mean() / t.mean():+.1f}%; mean |change| {np.abs(d).mean():.2e}")
