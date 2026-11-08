"""Build bw_causal_report.html: BW after the causality-leak fix.

Usage: python make_causal_report.py <repo_root> <nanogpt.json> [more.json ...]
The JSON files are compare_runs.py outputs (rn1-, rn2-, exp2- prefixes); the
GPT-2 numbers are parsed from the Slurm logs in <repo_root>.
"""
import json
import math
import os
import re
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bw_causal_report.html")
STYLE = re.search(r"<style>.*?</style>", open(os.path.join(HERE, "bw_paper_vs_code.html")).read(), re.S).group(0)

ROOT = sys.argv[1]
D = {}
for f in sys.argv[2:]:
    D.update(json.load(open(f)))

# ---------------------------------------------------------------- nanoGPT
N_BASE = ["mini-gpt-no-dropout-baseline", "exp2-base-s1", "exp2-base-s2"]
seeds = lambda p: [p, p + "-s1", p + "-s2"]
N_ROWS = [
    ("LayerNorm baseline", N_BASE, "14"),
    ("leave_one_out, per-sequence stats", seeds("rn1-loo-gs384-ff0.8"), "55"),
    ("leave_one_out, shared stats", seeds("rn2-loo-gs384-ff0.8"), "27.5"),
    ("running_mean, per-sequence stats", seeds("rn1-rm-gs384-mom0.7"), "57"),
    ("<b>running_mean, shared stats</b>", seeds("rn2-rm-gs384-mom0.7"), "27.8"),
]


def steps_to(its, losses, target):
    for i in range(1, len(its)):
        if losses[i] <= target:
            a, b = losses[i - 1], losses[i]
            return its[i - 1] + (its[i] - its[i - 1]) * (a - target) / max(a - b, 1e-12)
    return math.nan


def nano_metrics(names):
    a = np.array([[np.mean(D[n]["train"][1:]), steps_to(D[n]["iter"], D[n]["train"], 1.0),
                   D[n]["train"][-1], min(D[n]["val"])] for n in names])
    return a.mean(0), a.min(0), a.max(0)


def nano_table():
    base = nano_metrics(N_BASE)[0]
    rows = []
    for label, names, ms in N_ROWS:
        m, lo, hi = nano_metrics(names)
        is_base = names is N_BASE
        cell = lambda j, f: (f'<td class="num{"" if is_base or m[j] >= base[j] else " better"}">{m[j]:{f}} '
                             f'<span class="rng">({lo[j]:{f}}&ndash;{hi[j]:{f}})</span></td>')
        rel = "&mdash;" if is_base else f"{100 * (m[0] / base[0] - 1):+.1f}%".replace("-", "&minus;")
        rows.append(f'<tr{" class=baseline" if is_base else ""}><td>{label}</td><td class="num">{ms}</td>'
                    f'{cell(0, ".3f")}<td class="num">{rel}</td><td class="num">{m[1]:.0f}</td>'
                    f'{cell(2, ".4f")}{cell(3, ".3f")}</tr>')
    return "\n".join(rows)


# ---------------------------------------------------------------- GPT-2
LN_LOG = "g2-ln_21146146.out"
G_RUNS = [
    # label, log, settings, colour
    ("momentum 0.7 (first try)", "g2-rn-rm-gs768-mom0.7_21175878.out",
     "per-sequence stats, warmup on, <code>fix_factor</code> 0.9", "#afb8c1"),
    ("momentum 0.1", "g2-rnb-rm-gs768-mom0.1_21226208.out",
     "shared stats, warmup on, <code>fix_factor</code> 0.9", "#9a6700"),
    ("momentum 0.1, boost 3", "g2-rnb-rm-gs768-mom0.1-gb3_21226209.out",
     "as above, <code>bw_grad_scale=3</code>", "#cf222e"),
    ("momentum 0.1, no warmup, ff 0.9", "g2-rnb-mom0.1-nowarm-ff0.9_21227260.out",
     "shared stats, warmup off, <code>fix_factor</code> 0.9", "#0969da"),
    ("momentum 0.1, no warmup, ff 0.4", "g2-rnb-mom0.1-nowarm-ff0.4_21227261.out",
     "shared stats, warmup off, <code>fix_factor</code> 0.4", "#8250df"),
    ("momentum 0.1, no warmup, ff 0.1", "g2-rnb-mom0.1-nowarm-ff0.1_21227262.out",
     "shared stats, warmup off, <code>fix_factor</code> 0.1", "#1a7f37"),
    ("momentum 0.3, no warmup, ff 0.4", "g2-rnb-mom0.3-nowarm-ff0.4_21228219.out",
     "as ff 0.4, momentum 0.3", "#bf8700"),
    ("momentum 0.1, no warmup, ff 0.4, BW in layers 0&ndash;5 only", "g2-rnb-mom0.1-nowarm-ff0.4-l6_21228220.out",
     "as ff 0.4, <code>bw_layers=6</code>", "#57606a"),
    ("momentum 0.3, no warmup, ff 0.4, BW in layers 0&ndash;3 only", "g2-rnb-mom0.3-nowarm-ff0.4-l4_21229480.out",
     "as ff 0.4, momentum 0.3, <code>bw_layers=4</code>", "#e16f24"),
    ("<b>momentum 0.3, no warmup, ff 0.4, BW in layers 0&ndash;5 only</b>", "g2-rnb-mom0.3-nowarm-ff0.4-l6_21229479.out",
     "as ff 0.4, momentum 0.3, <code>bw_layers=6</code>", "#000080"),
]
WINDOWS = [(200, 800), (800, 1400), (1400, 2000), (2000, 4000)]
EVALS = [2000, 4000]


def log(path):
    s = open(os.path.join(ROOT, path)).read()
    it = {int(a): float(b) for a, b in re.findall(r"^iter (\d+): loss ([\d.]+)", s, re.M)}
    tm = [float(b) for a, b in re.findall(r"^iter (\d+): loss [\d.]+, time ([\d.]+)ms", s, re.M) if int(a) % 1000 and int(a) > 50]
    ev = {int(a): (float(b), float(c)) for a, b, c in re.findall(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", s, re.M)}
    return it, statistics.median(tm) / 1000 if tm else math.nan, ev


LN_IT, LN_T, LN_EV = log(LN_LOG)


def rel_window(it, a, b):
    w = [i for i in range(a, b) if i in it and i in LN_IT]
    if len(w) < 0.8 * (b - a) / 10:
        return None
    return 100 * (sum(it[i] for i in w) / sum(LN_IT[i] for i in w) - 1)


def pct(v):
    if v is None:
        return '<td class="num">&mdash;</td>'
    txt = f"{v:+.1f}%".replace("-", "&minus;")
    return f'<td class="num{" better" if v < 0 else ""}">{txt}</td>'


def g2_table():
    rows = [f'<tr class="baseline"><td>LayerNorm baseline</td><td>&mdash;</td><td class="num">{LN_T:.2f}</td>'
            + "".join('<td class="num">0</td>' for _ in WINDOWS) + "<td class=num>&mdash;</td>" * len(EVALS) + "</tr>"]
    for label, path, settings, _ in G_RUNS:
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        it, t, ev = log(path)
        evs = "".join(
            f"<td class=num>{100 * (ev[s][0] / LN_EV[s][0] - 1):+.1f}% / {100 * (ev[s][1] / LN_EV[s][1] - 1):+.1f}%</td>"
            .replace("-", "&minus;") if s in ev else
            f"<td class=num>{'stopped at ' + str(max(it)) if max(it) < s - 10 and s == 2000 else '&mdash;'}</td>"
            for s in EVALS)
        rows.append(f"<tr><td>{label}</td><td>{settings}</td><td class=num>{t:.2f}</td>"
                    + "".join(pct(rel_window(it, a, b)) for a, b in WINDOWS) + evs + "</tr>")
    return "\n".join(rows)


def g2_chart():
    W, H, L, R, T, B = 760, 300, 56, 16, 16, 40
    ylim, xmax = 10, 4000
    xs = lambda i: L + (W - L - R) * i / xmax
    ys = lambda v: T + (H - T - B) * (ylim - max(-ylim, min(ylim, v))) / (2 * ylim)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="GPT-2 train loss relative to LayerNorm">']
    for v in range(-ylim, ylim + 1, 5):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#d1d9e0"/>')
        g.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#59636e">{v:+d}%</text>')
    g.append(f'<line x1="{xs(0)}" x2="{xs(xmax)}" y1="{ys(0):.1f}" y2="{ys(0):.1f}" stroke="#000" stroke-width="2"/>')
    for i in range(0, xmax + 1, 500):
        g.append(f'<text x="{xs(i):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="#59636e">{i}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" font-size="12">Iteration</text>')
    legend = ['<span><i style="background:#000"></i>LayerNorm (0%)</span>']
    for label, path, _, color in G_RUNS:
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        it, _, _ = log(path)
        pts = []
        for c in range(100, min(xmax, max(it)) - 99, 20):
            v = rel_window(it, c - 100, c + 100)
            if v is not None:
                pts.append(f"{xs(c):.1f},{ys(v):.1f}")
        g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')
    g.append("</svg>")
    return '<div class="legend-line">' + "".join(legend) + "</div>\n" + "\n".join(g)


TEXT = dict(
    tldr="""<b>Summary.</b>
<ul>
  <li>Training-mode BW leaked each sequence's future tokens once two BW layers were stacked. The fix (<code>bw_renorm</code>) whitens the forward pass with the stored statistics and keeps the gradient through the batch statistics. It is causal by construction, and the leak is now checked by the unit test and at every training start.</li>
  <li><b>nanoGPT:</b> the fixed BW still beats LayerNorm (about 8% lower mean train loss), half the old, leak-inflated margin. Shared batch statistics (<code>bw_renorm_stats=batch</code>) halve its cost at no loss.</li>
  <li><b>GPT-2:</b> no setting beats LayerNorm over a longer run yet. The best (momentum 0.3, <code>fix_factor</code> 0.4, BW in layers 0&ndash;5 only) is 2&ndash;3% ahead early, 0.5% at step 2000 and level by step 4000 (train +0.2%, val tie), at 1.25&times; LayerNorm's time per iteration.</li>
</ul>""",
    fix="""<p><b>Leak.</b> Leave-one-out statistics keep one BW layer causal, but with two layers there is a path through another sequence: layer <i>k</i> writes sequence <i>b</i>'s future into sequence <i>s</i>'s activations, and layer <i>k</i>+1 reads them back into <i>b</i>'s statistics. On GPT-2 (B&nbsp;=&nbsp;12) the model exploited it within 2,000 steps: training-forward loss 1.60 against 9.0 in eval mode. All nanoGPT numbers in the main report are eval-mode losses and are not inflated, but its training may have used the leak.</p>
<p><b>Fix: Batch Renormalization applied to whitening</b> (<code>bw_renorm=True</code>).</p>
<ul class="summary">
  <li><b>Forward:</b> whiten with the buffers from earlier steps, <i>L<sub>R</sub></i><sup>&minus;1</sup>(<i>x</i>&minus;<i>&mu;<sub>R</sub></i>), exactly what inference computes. No current-batch value enters the forward pass, so training and evaluation compute the same function.</li>
  <li><b>Gradient:</b> through the current-batch statistics, written as <i>R</i>&nbsp;<i>L<sub>B</sub></i><sup>&minus;1</sup>(<i>x</i>&minus;<i>&mu;<sub>B</sub></i>)&nbsp;+&nbsp;<i>d</i> with <i>R</i>&nbsp;=&nbsp;<i>L<sub>R</sub></i><sup>&minus;1</sup><i>L<sub>B</sub></i> and <i>d</i>&nbsp;=&nbsp;<i>L<sub>R</sub></i><sup>&minus;1</sup>(<i>&mu;<sub>B</sub></i>&minus;<i>&mu;<sub>R</sub></i>) held constant. The value is bit-for-bit the stored-statistics whitening.</li>
  <li><b>Shared statistics</b> (<code>bw_renorm_stats=batch</code>): since the batch statistics only shape the gradient, they no longer need to be per-sequence leave-one-out. One mean and covariance per batch, applied as a matrix product: 2&times; faster on nanoGPT and 2.4&times; on GPT-2, same results.</li>
  <li><b>Guards:</b> <code>test_batch_orthonorm.py</code> stacks 1&ndash;3 layers and fails if a causal mode leaks or the known leak is not detected. <code>train.py</code> measures the leak on the real model at startup (<code>GPT.causal_leak</code>, wandb <code>causal_leak</code>) and refuses to train if it is nonzero, unless <code>--allow_causal_leak=True</code>.</li>
</ul>
<p>Two other fixes were considered and not built: causal per-position statistics (statistics at position <i>t</i> from positions &le;&nbsp;<i>t</i> only), and a reference sub-batch (a quarter of the sequences whitened with stored statistics, the rest with statistics from that quarter). Both keep current-batch statistics in the forward pass.</p>""",
    retested="""<table>
<tr><th>Feature</th><th>After the fix</th></tr>
<tr><td><code>running_mean</code>, <code>leave_one_out</code> with renorm</td><td>Tested: nanoGPT (3 seeds each) and GPT-2.</td></tr>
<tr><td>Group size, <code>fix_factor</code>, momentum</td><td>nanoGPT: only the two best pre-fix settings. GPT-2: group 768, <code>fix_factor</code> 0.1/0.4/0.9, momentum 0.1/0.3/0.7.</td></tr>
<tr><td>Gradient boost (<code>bw_grad_scale</code>)</td><td>GPT-2 only, boost 3: worse.</td></tr>
<tr><td>BW warmup (<code>bw_warmup</code>)</td><td>GPT-2: <code>diag</code> and <code>none</code>.</td></tr>
<tr><td>BW in the first layers only (<code>bw_layers</code>, new)</td><td>GPT-2: first 6 of 12.</td></tr>
<tr><td>Dropout 0.2 configurations, <code>calibrated</code> mode, the nanoGPT boost sweep</td><td><b>Not re-tested.</b> <code>calibrated</code> still has a current-batch term in its forward pass and is rejected by the startup check.</td></tr>
</table>""",
    nano="""<ul class="summary">
  <li><b>Causal BW still wins, by about half the leaky margin.</b> Mean train loss 7&ndash;8% below the baseline on every seed, against 13% before the fix. The gain is early: loss 1.0 is reached 10&ndash;13% sooner; final loss ties.</li>
  <li><b>Shared statistics cost nothing and halve the BW overhead</b> (55 &rarr; 27.5 ms per iteration; LayerNorm 14 ms).</li>
</ul>""",
    diag="""<p><code>diag_whitening_quality.py</code> runs the LayerNorm checkpoints (no training, about 4 GPU-minutes). For the input of each BW layer it compares the condition number of the covariance (largest over smallest eigenvalue; lower is better) after per-channel standardization only, after whitening with statistics estimated the way BW does in training, and after whitening with the exact covariance.</p>
<table>
<tr><th>Model</th><th class="num">Standardization only</th><th>BW setting</th><th class="num">Estimated, momentum 0.7</th><th class="num">0.3</th><th class="num">0.1</th><th class="num">Exact covariance</th></tr>
<tr><td>nanoGPT (B&nbsp;=&nbsp;64)</td><td class="num">2,939</td><td>group 384, <code>fix_factor</code> 0.8</td><td class="num">87</td><td class="num">77</td><td class="num">77</td><td class="num">73</td></tr>
<tr><td>GPT-2 (B&nbsp;=&nbsp;12)</td><td class="num">4,524</td><td>group 768, <code>fix_factor</code> 0.9</td><td class="num">137</td><td class="num">87</td><td class="num">59</td><td class="num">42</td></tr>
</table>
<p class="cap">Geometric mean over layers. GPT-2 layers 1&ndash;4 are the worst conditioned (7,000&ndash;130,000); later layers 340&ndash;2,400.</p>
<ul class="summary">
  <li><b>The problem BW solves exists in GPT-2</b>, as much as in nanoGPT.</li>
  <li><b>GPT-2's statistics are much noisier.</b> BW updates its buffers on every micro-batch, and one GPT-2 optimizer step is 10 micro-batches of 12 sequences. Momentum 0.7 therefore averages about 18 sequences; momentum 0.1 averages about 120 and is still only one optimizer step old, the same staleness nanoGPT has at 0.7.</li>
</ul>""",
    gpt2="""<ul class="summary">
  <li><b>Momentum matched to gradient accumulation gives an early gain.</b> Momentum 0.7 (about 18 sequences in the buffers) was 1&ndash;2% behind throughout. Momentum 0.1&ndash;0.3 is about 2.5&ndash;3% ahead over iterations 200&ndash;800 in every configuration.</li>
  <li><b>Full decorrelation hurts on GPT-2; mild is best.</b> <code>fix_factor</code> 0.9 falls behind after iteration 800 with or without warmup (+4% and +6% at step 2000). With 0.1&ndash;0.4 the early gain mostly survives.</li>
  <li><b>The warmup handoff is not the main problem.</b> With <code>bw_warmup=diag</code> the switch to full covariance at step 1000 causes a jump, but turning warmup off does not remove the fade after iteration 800 at <code>fix_factor</code> 0.9. It coincides with the end of the learning-rate ramp (2,000 steps), when stale statistics matter most.</li>
  <li><b>Gradient boost 3 diverges</b> (+17% over iterations 800&ndash;1400; stopped).</li>
  <li><b>BW only in the first layers is best, and cheaper.</b> The diagnostic shows layers 1&ndash;4 are by far the worst conditioned. With BW in layers 0&ndash;5 and LayerNorm in 6&ndash;11, momentum 0.1 is the first run that is still ahead at step 2000 (&minus;0.4%), at 1.06 s per iteration (LayerNorm 0.85 s). BW in layers 0&ndash;3 only (0.99 s) and momentum 0.3 give the same &minus;0.4% to &minus;0.5%.</li>
  <li><b>The gain does not last.</b> Run to step 4000, the best setting goes from &minus;2.4% at step 1000 to &minus;0.5% at 2000, &minus;0.1% at 3000 and +0.2% at 4000. BW speeds up early training; LayerNorm catches up once the learning rate is at its peak. The BW input scale keeps growing (<code>input_rms</code> 1.0 at step 2000, 2.2 at 4000).</li>
</ul>""",
    conclusions="""<ul class="summary">
  <li><b>Causality:</b> fixed with renorm, guarded by the unit test and a startup check. Pre-fix GPT-2 results and the pre-fix nanoGPT margins are leak-affected; the tables here are all post-fix.</li>
  <li><b>nanoGPT:</b> causal BW is about 8% better than LayerNorm in mean train loss and twice as cheap with shared statistics.</li>
  <li><b>GPT-2:</b> causal BW helps early (2&ndash;3%) but is level with LayerNorm by step 4000. What mattered: momentum matched to gradient accumulation (0.1&ndash;0.3), mild decorrelation (<code>fix_factor</code> 0.1&ndash;0.4), no warmup, and BW only in the first 4&ndash;6 layers. What did not help: full decorrelation, gradient boost, BW warmup.</li>
  <li><b>Next, if continuing:</b> (1) update the BW buffers once per optimizer step from all micro-batches instead of on every micro-batch, which gives exact 120-sequence statistics with no staleness; (2) BW in the first layers early and a switch to LayerNorm later, since the gain is early convergence; (3) the 4-GPU setup, with a 4&times; larger batch per step. Each is one short run (about 1 GPU-hour).</li>
</ul>""",
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Causal Batch Whitening: after the leak fix</title>
{STYLE}
</head>
<body>
<h1>Causal Batch Whitening: after the leak fix</h1>
<p class="sub">What changed once the training-mode leak was found and fixed, and what we tried since. The full feature list and the pre-fix experiments are in <a href="bw_paper_vs_code.html">bw_paper_vs_code.html</a>.</p>

<div class="todo">{TEXT["tldr"]}</div>

<h2>The leak and the fix</h2>
{TEXT["fix"]}

<h2>What was re-tested after the fix</h2>
{TEXT["retested"]}

<h2>nanoGPT (shakespeare_char, dropout 0, 3 seeds)</h2>
<table>
<tr><th>Configuration (group 384)</th><th class="num">ms / iter</th><th class="num">Mean train loss</th><th class="num">vs. baseline</th><th class="num">Steps to loss 1.0</th><th class="num">Final train loss</th><th class="num">Best val</th></tr>
{nano_table()}
</table>
<p class="cap">All renorm. <code>leave_one_out</code> with <code>fix_factor</code> 0.8, <code>running_mean</code> with momentum 0.7. Mean over seeds (min&ndash;max). Eval-mode losses. Bold green = better than the baseline.</p>
{TEXT["nano"]}

<h2>Why GPT-2 is harder: a no-training diagnostic</h2>
{TEXT["diag"]}

<h2>GPT-2 124M, 1 GPU (B&nbsp;=&nbsp;12 &times; 10 micro-batches, T&nbsp;=&nbsp;1024)</h2>
<table>
<tr><th>Run (renorm <code>running_mean</code>, group 768)</th><th>Settings</th><th class="num">s / iter</th>
{"".join(f'<th class="num">iters {a}&ndash;{b}</th>' for a, b in WINDOWS)}{"".join(f'<th class="num">Step {s} train / val</th>' for s in EVALS)}</tr>
{g2_table()}
</table>
<p class="cap">Per-iteration training loss relative to the LayerNorm baseline at the same iterations, averaged over each window. Negative = better (bold green). Same learning-rate schedule in every run (warmup 2,000 steps); BW runs stop at step 2000 except the first and the last (4000).</p>
{g2_chart()}
<p class="cap">Training loss relative to LayerNorm, 200-iteration moving average, clipped to &plusmn;10%.</p>
{TEXT["gpt2"]}

<h2>Conclusions and next steps</h2>
{TEXT["conclusions"]}
</body>
</html>
"""
open(OUT, "w").write(html)
print("wrote", OUT)
