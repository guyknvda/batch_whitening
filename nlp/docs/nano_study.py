"""Summarize nanoGPT study logs (nlp/study/logs/*.out) against a LayerNorm run.

Usage: python nano_study.py <log_dir> <baseline_name> [name_regex]
Prints one row per run: mean eval train loss over steps 250-5000 (area under
the curve) and its change vs. the baseline, final train loss, steps to reach
train loss 1.0 / 0.5 / 0.2 / 0.1, best val loss, the geometric-mean condition
number of the MLP input and of the residual stream at steps 1000 and 5000, the
median ms per iteration, and whether the run aborted.
"""
import glob
import math
import os
import re
import statistics
import sys

STEP = re.compile(r"^step (\d+): train loss ([\d.na]+), val loss ([\d.na]+)", re.M)
COND = re.compile(r"^step (\d+): cond (\w+) ([\d.e+]+) \[", re.M)
ITER = re.compile(r"^iter (\d+): loss [\d.na]+, time ([\d.]+)ms", re.M)


def parse(path):
    s = open(path).read()
    ev = {int(a): (float(b), float(c)) for a, b, c in STEP.findall(s)}
    cond = {}
    for st, k, v in COND.findall(s):
        cond.setdefault(k, {})[int(st)] = float(v)
    ms = [float(t) for i, t in ITER.findall(s) if int(i) % 250 and int(i) > 20]
    abort = re.search(r"^ABORT at step (\d+)", s, re.M)
    return dict(ev=ev, cond=cond, ms=statistics.median(ms) if ms else math.nan,
                abort=int(abort.group(1)) if abort else None,
                error="Traceback" in s, done="Job completed" in s or 5000 in ev)


def steps_to(ev, target):
    its = sorted(ev)
    for a, b in zip(its, its[1:]):
        la, lb = ev[a][0], ev[b][0]
        if lb <= target < la:
            return a + (b - a) * (la - target) / (la - lb)
    return math.nan


def mean_train(ev, upto=5000):
    v = [ev[i][0] for i in sorted(ev) if 0 < i <= upto]
    return sum(v) / len(v) if v else math.nan


def summarize(log_dir, base, pattern="."):
    runs = {os.path.basename(p)[:-4]: parse(p) for p in sorted(glob.glob(os.path.join(log_dir, "*.out")))}
    b = runs[base]
    last = max(b["ev"])
    rows = []
    for name, r in runs.items():
        if not re.search(pattern, name) or not r["ev"]:
            continue
        upto = max(r["ev"])
        m, mb = mean_train(r["ev"], upto), mean_train(b["ev"], upto)
        c = r["cond"]
        rows.append((name, m, 100 * (m / mb - 1), r["ev"].get(last, (math.nan,))[0],
                     *[steps_to(r["ev"], t) for t in (1.0, 0.5, 0.2, 0.1)],
                     min(v for _, v in r["ev"].values()),
                     c.get("mlp_in", {}).get(1000, math.nan), c.get("mlp_in", {}).get(last, math.nan),
                     c.get("resid", {}).get(last, math.nan), r["ms"],
                     f"abort@{r['abort']}" if r["abort"] else ("error" if r["error"] else
                                                              ("" if upto >= last else f"at {upto}"))))
    return rows


if __name__ == "__main__":
    rows = summarize(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    hdr = ("run", "mean", "vs base", "final", "to1.0", "to0.5", "to0.2", "to0.1", "bestval",
           "condMLP@1k", "condMLP@end", "condRes@end", "ms/it", "status")
    print("%-22s %6s %8s %7s %6s %6s %6s %6s %7s %10s %11s %11s %6s %s" % hdr)
    for r in sorted(rows, key=lambda r: r[1]):
        print("%-22s %6.3f %+7.1f%% %7.4f %6.0f %6.0f %6.0f %6.0f %7.3f %10.3g %11.3g %11.3g %6.1f %s" % r)
