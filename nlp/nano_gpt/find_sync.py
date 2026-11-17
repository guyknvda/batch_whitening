"""Find host-device synchronizations in a BW training step: each piece is issued
while the GPU is kept busy by a sleep kernel, so a piece whose CPU time is near
the sleep length waits for the GPU.

Run: python find_sync.py [config overrides as key=value ...]   (needs a GPU)
"""
import sys
import time

import torch

from model import GPT, GPTConfig, BatchWhiteningBlock


def parse(v):
    if v in ("True", "False"):
        return v == "True"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def issue_ms(fn):
    torch.cuda.synchronize()
    torch.cuda._sleep(300_000_000)
    t = time.time()
    out = fn()
    ms = (time.time() - t) * 1e3
    torch.cuda.synchronize()
    return ms, out


torch.backends.cuda.matmul.allow_tf32 = True
kw = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, dropout=0.0, bias=False)
for a in sys.argv[1:]:
    k, v = a.split("=")
    kw[k] = parse(v)
model = GPT(GPTConfig(**kw)).cuda()
for l in getattr(model, "bw_layers", []):
    l.cov_warmup = False
opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cuda")
x = torch.randint(0, 65, (64, 256), device="cuda")


def step():
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(5):
    step()

A = torch.randn(384, 384, device="cuda")
A = A @ A.T + 384 * torch.eye(384, device="cuda")
L = torch.linalg.cholesky(A)
I = torch.eye(384, device="cuda")
bw = next(m for m in model.modules() if isinstance(m, BatchWhiteningBlock))
X = torch.randn(64, 256, 384, device="cuda", requires_grad=True)
print(f"sleep alone: {issue_ms(lambda: torch.cuda._sleep(300_000_000))[0]:.1f} ms (reference for a full wait)")
pieces = {
    "cholesky_ex": lambda: torch.linalg.cholesky_ex(A),
    "solve_triangular": lambda: torch.linalg.solve_triangular(L, I, upper=False),
    "bw layer forward": lambda: bw(X),
}
for name, fn in pieces.items():
    print(f"{name}: {issue_ms(fn)[0]:.1f} ms")
_, Y = issue_ms(lambda: bw(X))
print(f"bw layer backward: {issue_ms(lambda: Y.sum().backward())[0]:.1f} ms")
ms, (_, loss) = issue_ms(lambda: model(x, x))
print(f"model forward: {ms:.1f} ms")
print(f"model backward: {issue_ms(lambda: loss.backward())[0]:.1f} ms")
print(f"optimizer step: {issue_ms(lambda: (opt.step(), opt.zero_grad(set_to_none=True)))[0]:.1f} ms")
