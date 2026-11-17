"""GPU time vs wall time of full nanoGPT training steps (forward, backward, AdamW).

Run: python profile_step.py [config overrides as key=value ...]   (needs a GPU)
"""
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

import model as model_module
from model import GPT, GPTConfig

model_module.COMPILE_GLUE = os.environ.get("COMPILE_GLUE", "1") == "1"

torch.backends.cuda.matmul.allow_tf32 = True
kw = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65, dropout=0.0, bias=False)
def parse(v):
    if v in ("True", "False"):
        return v == "True"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


for a in sys.argv[1:]:
    k, v = a.split("=")
    kw[k] = parse(v)
model = GPT(GPTConfig(**kw)).cuda()
for l in getattr(model, "bw_layers", []):
    l.cov_warmup = False
opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cuda")
x = torch.randint(0, 65, (int(os.environ.get("BATCH", 64)), 256), device="cuda")


def step():
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


for _ in range(20):
    step()
walls = []
for _ in range(3):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(100):
        step()
    torch.cuda.synchronize()
    walls.append((time.time() - t) / 100 * 1e3)
wall = min(walls)
# CPU time to issue a step while the GPU is kept busy (so launches never wait for it)
torch.cuda.synchronize()
torch.cuda._sleep(2_000_000_000)
t = time.time()
for _ in range(3):
    step()
issue = (time.time() - t) / 3 * 1e3
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
ev = prof.key_averages()
gpu = sum(e.self_cuda_time_total for e in ev) / 10 / 1e3
print(f"batch {x.size(0)} compile_glue {model_module.COMPILE_GLUE} {' '.join(sys.argv[1:]) or 'default'}: wall {wall:.2f} ms/step (100-step blocks: {', '.join(f'{w:.2f}' for w in walls)}), CPU issue {issue:.2f} ms/step, summed kernel time {gpu:.2f} ms/step")
print(ev.table(sort_by="self_cuda_time_total", row_limit=25))
prof.export_chrome_trace("/tmp/profile_step_trace.json")
kernels = [e for e in json.load(open("/tmp/profile_step_trace.json"))["traceEvents"]
           if e.get("cat") == "kernel"]
span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 10 / 1e3
busy, end = 0.0, -1.0
for s, d in sorted((e["ts"], e["dur"]) for e in kernels):
    busy += max(0.0, s + d - max(s, end))
    end = max(end, s + d)
by_stream = {}
for e in kernels:
    by_stream[e["args"]["stream"]] = by_stream.get(e["args"]["stream"], 0.0) + e["dur"]
print(f"GPU span {span:.2f} ms/step, busy (any stream) {busy / 10 / 1e3:.2f} ms/step, per stream: "
      + ", ".join(f"{k}: {v / 10 / 1e3:.2f}" for k, v in by_stream.items()))
# Idle gaps on the busiest stream: one that ends within 5 us of a kernel ending on
# another stream is a wait on that stream; the rest wait for the CPU to launch.
main = max(by_stream, key=by_stream.get)
mk = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in kernels if e["args"]["stream"] == main)
other_ends = sorted(e["ts"] + e["dur"] for e in kernels if e["args"]["stream"] != main)
cross, cpu, big = 0.0, 0.0, []
for (s0, e0, n0), (s1, e1, n1) in zip(mk, mk[1:]):
    gap = s1 - e0
    if gap <= 2:
        continue
    if any(s1 - 5 <= t <= s1 for t in other_ends):
        cross += gap
    else:
        cpu += gap
    big.append((gap, n0[:50], n1[:50]))
print(f"main stream idle: waiting on side stream {cross / 10 / 1e3:.2f} ms/step, "
      f"other (CPU launch) {cpu / 10 / 1e3:.2f} ms/step")
for gap, n0, n1 in sorted(big, reverse=True)[:12]:
    print(f"  gap {gap:.0f} us after {n0} before {n1}")

by_name = {}
for e in kernels:
    k = (e["args"]["stream"] == main, e["name"][:70])
    by_name[k] = by_name.get(k, 0.0) + e["dur"]
for (on_main, n), d in sorted(by_name.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  kernel {'main' if on_main else 'side'} {d / 10:7.1f} us/step {n}")
