"""Compare wandb train-loss curves against the no-dropout LayerNorm baseline.

Usage: python compare_runs.py [--baseline=ID] [--thresholds=1.5,1.2,...] <name-prefix> [extra-run-id ...]
Prints, per run: steps to reach each train-loss threshold (linear
interpolation between evals), mean train loss over all evals (area under the
curve), the final train/val loss, and the best val loss with its step.
Writes the raw curves to /tmp/compare_<prefix>.json.
"""
import json
import sys

import numpy as np
import wandb

PROJECT = "nv-welcome/shakespeare-char"
BASELINE = "l965qcs2"
THRESHOLDS = [1.0, 0.5, 0.2, 0.1]


def curve(run):
    h = run.history(keys=["iter", "train/loss", "val/loss"], pandas=False, samples=1000)
    h = sorted(h, key=lambda x: x["iter"])
    return ([x["iter"] for x in h], [x["train/loss"] for x in h], [x["val/loss"] for x in h])


def steps_to(its, losses, target):
    for i in range(1, len(its)):
        if losses[i] <= target:
            a, b = losses[i - 1], losses[i]
            return its[i - 1] + (its[i] - its[i - 1]) * (a - target) / max(a - b, 1e-12)
    return float("nan")


def main():
    global BASELINE, THRESHOLDS
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--baseline="):
            BASELINE = a.split("=", 1)[1]
        elif a.startswith("--thresholds="):
            THRESHOLDS = [float(t) for t in a.split("=", 1)[1].split(",")]
        else:
            args.append(a)
    prefix = args[0]
    api = wandb.Api(timeout=60)
    runs = [api.run(f"{PROJECT}/{BASELINE}")]
    runs += [r for r in api.runs(PROJECT, filters={"display_name": {"$regex": f"^{prefix}"}})]
    runs += [api.run(f"{PROJECT}/{i}") for i in args[1:]]
    rows, raw = [], {}
    for r in runs:
        its, tr, va = curve(r)
        if not its or its[-1] < 5000:
            print(f"skip {r.name} ({r.state}, last iter {its[-1] if its else None})")
            continue
        raw[r.name] = dict(id=r.id, iter=its, train=tr, val=va)
        bi = int(np.argmin(va))
        rows.append((r.name, r.id, [steps_to(its, tr, t) for t in THRESHOLDS],
                     float(np.mean(tr[1:])), tr[-1], va[-1], va[bi], its[bi]))
    base = rows[0]
    hdr = "".join(f"{'to ' + str(t):>9}" for t in THRESHOLDS)
    print(f"{'run':38s}{hdr}{'meanL':>8}{'final':>8}{'val':>7}{'bestval':>9}{'@':>6}")
    for name, rid, st, auc, fin, va, bv, bit in sorted(rows, key=lambda x: x[3]):
        cells = "".join(f"{s:9.0f}" for s in st)
        print(f"{name[:37]:38s}{cells}{auc:8.4f}{fin:8.4f}{va:7.3f}{bv:9.4f}{bit:6d}"
              f"{'  <- baseline' if rid == BASELINE else ''}")
    print(f"\nbaseline mean loss {base[3]:.4f}, final {base[4]:.4f}")
    with open(f"/tmp/compare_{prefix}.json", "w") as f:
        json.dump(raw, f)


if __name__ == "__main__":
    main()
