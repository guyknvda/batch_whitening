"""Fill the results section of bw_paper_vs_code.html from compare_runs.py output.

Usage: python make_bw_report.py /tmp/compare_exp.json
       python make_bw_report.py --dropout /tmp/compare_drop.json
       python make_bw_report.py --boost /tmp/compare_exp.json /tmp/compare_drop.json /tmp/compare_gb.json
Replaces everything between <!--RESULTS--> and <!--/RESULTS--> (or the
<!--DROPOUT--> markers) in the report.
"""
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "bw_paper_vs_code.html")
THRESHOLDS = [1.0, 0.5, 0.2, 0.1]

BASE = ["mini-gpt-no-dropout-baseline", "exp2-base-s1", "exp2-base-s2"]
MAIN = [
    ("Baseline: LayerNorm", BASE, True),
    ("running_mean, group 16 (current default)", ["exp1-rm-gs16-ref"], False),
    ("running_mean, group 48 (paper's rule)", ["exp1-rm-gs48"], False),
    ("running_mean, group 384, momentum 0.1", ["exp2-rm-gs384"], False),
    ("running_mean, group 384, momentum 0.7", ["exp3-rm-gs384-mom0.7"], False),
    ("leave_one_out, group 16", ["mini-gpt-no-dropout-loo-groupsize-16"], False),
    ("leave_one_out, group 384", ["exp1-loo-gs384", "exp2-loo-gs384-s1", "exp2-loo-gs384-s2"], False),
    ("leave_one_out, group 384, fix_factor 0.8",
     ["exp2-loo-gs384-ff0.8", "exp3-loo-gs384-ff0.8-s1", "exp3-loo-gs384-ff0.8-s2"], False),
]
CHART = [
    ("running_mean, group 16 (current default)", ["exp1-rm-gs16-ref"], "#9a6700"),
    ("running_mean, group 384, momentum 0.7", ["exp3-rm-gs384-mom0.7"], "#8250df"),
    ("leave_one_out, group 384, fix_factor 0.8",
     ["exp2-loo-gs384-ff0.8", "exp3-loo-gs384-ff0.8-s1", "exp3-loo-gs384-ff0.8-s2"], "#1a7f37"),
]


def steps_to(its, losses, target):
    for i in range(1, len(its)):
        if losses[i] <= target:
            a, b = losses[i - 1], losses[i]
            return its[i - 1] + (its[i] - its[i - 1]) * (a - target) / max(a - b, 1e-12)
    return math.nan


def metrics(d, names):
    names = [n for n in names if n in d]
    if not names:
        return None
    per = []
    for n in names:
        its, tr, va = d[n]["iter"], d[n]["train"], d[n]["val"]
        per.append([steps_to(its, tr, t) for t in THRESHOLDS] + [np.mean(tr[1:]), tr[-1], va[-1]])
    return np.nanmean(np.array(per, dtype=float), axis=0), len(names)


def mean_curve(d, names):
    names = [n for n in names if n in d]
    if not names:
        return None, None
    its = d[names[0]]["iter"]
    return its, np.mean([d[n]["train"] for n in names], axis=0)


def main_table(d):
    base, _ = metrics(d, BASE)
    out = ['<table>',
           '<tr><th rowspan="2">Configuration</th><th rowspan="2" class="num">Seeds</th>'
           f'<th colspan="{len(THRESHOLDS)}" style="text-align:center">Steps to reach train loss</th>'
           '<th rowspan="2" class="num">Mean loss</th><th rowspan="2" class="num">Final loss</th></tr>',
           '<tr>' + "".join(f'<th class="num">{t}</th>' for t in THRESHOLDS) + '</tr>']
    for label, names, is_base in MAIN:
        m = metrics(d, names)
        if m is None:
            continue
        vals, n = m
        cells = []
        for i, v in enumerate(vals[:6]):
            better = (not is_base) and not math.isnan(v) and v < base[i]
            txt = "&mdash;" if math.isnan(v) else (f"{v:.0f}" if i < 4 else f"{v:.4f}" if i == 5 else f"{v:.3f}")
            cells.append(f'<td class="num{" better" if better else ""}">{txt}</td>')
        cls = ' class="baseline"' if is_base else ""
        out.append(f'<tr{cls}><td>{label}</td><td class="num">{n}</td>{"".join(cells)}</tr>')
    out.append("</table>")
    return "\n".join(out)


def chart(d, base_names=BASE, series=CHART, ylim=40, step=20):
    bits, base = mean_curve(d, base_names)
    W, H, L, R, T, B = 760, 300, 56, 16, 16, 40
    ymin, ymax = -ylim, ylim
    xs = lambda it: L + (W - L - R) * it / 5000
    ys = lambda v: T + (H - T - B) * (ymax - v) / (ymax - ymin)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Train loss relative to baseline">']
    for v in range(ymin, ymax + 1, step):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#d1d9e0" stroke-width="1"/>')
        g.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#59636e">{v:+d}%</text>')
    g.append(f'<line x1="{xs(bits[1]):.1f}" x2="{xs(bits[-1]):.1f}" y1="{ys(0):.1f}" y2="{ys(0):.1f}" stroke="#000" stroke-width="2"/>')
    for it in range(0, 5001, 1000):
        g.append(f'<text x="{xs(it):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="#59636e">{it}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" font-size="12" fill="#1f2328">Training step</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" font-size="12" fill="#1f2328" '
             f'transform="rotate(-90 14 {(T + H - B) / 2})">Train loss vs. baseline</text>')
    legend = []
    for label, names, color in series:
        its, cur = mean_curve(d, names)
        if its is None:
            continue
        pts = [(xs(i), ys(max(ymin, min(ymax, 100 * (c / b - 1)))))
               for i, c, b in zip(its, cur, base) if i > 0]
        g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')
    g.append("</svg>")
    return ('<div class="legend-line"><span><i style="background:#000"></i>Baseline (0%)</span>'
            + "".join(legend) + "</div>\n" + "\n".join(g))


def speedups(d, base_names, series):
    """Per series: (label, color, step at which its seed-mean curve first reaches the baseline's final loss)."""
    its, base = mean_curve(d, base_names)
    out = []
    for label, names, color in series:
        _, cur = mean_curve(d, names)
        if cur is not None:
            s = steps_to(list(its), list(cur), base[-1])
            if not math.isnan(s):
                out.append((label, color, s))
    return base[-1], its[-1], out


def learning_curves(d, base_names, series, ticks, accel=False):
    """Absolute eval-mode train loss per step on a log axis, one line per configuration (seed mean).
    With accel, marks where each series reaches the baseline's final loss."""
    W, H, L, R, T, B = 760, 300, 56, 16, 16, 40
    lo, hi = math.log(ticks[0]), math.log(ticks[-1])
    xs = lambda it: L + (W - L - R) * it / 5000
    ys = lambda v: T + (H - T - B) * (hi - math.log(min(max(v, ticks[0]), ticks[-1]))) / (hi - lo)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Train loss learning curves">']
    for v in ticks:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#d1d9e0" stroke-width="1"/>')
        g.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#59636e">{v:g}</text>')
    for it in range(0, 5001, 1000):
        g.append(f'<text x="{xs(it):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="#59636e">{it}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" font-size="12" fill="#1f2328">Training step</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" font-size="12" fill="#1f2328" '
             f'transform="rotate(-90 14 {(T + H - B) / 2})">Train loss (log scale)</text>')
    legend = []
    for label, names, color in [("Baseline", base_names, "#000")] + list(series):
        its, cur = mean_curve(d, names)
        if its is None:
            continue
        g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
                 + " ".join(f"{xs(i):.1f},{ys(c):.1f}" for i, c in zip(its, cur) if i > 0) + '"/>')
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')
    if accel:
        target, end, hits = speedups(d, base_names, series)
        y = ys(target)
        for k, (_, color, s) in enumerate(sorted(hits, key=lambda h: h[2])):
            g.append(f'<line x1="{xs(s):.1f}" x2="{xs(end):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" '
                     f'stroke-width="2" stroke-dasharray="5 3"/>')
            g.append(f'<circle cx="{xs(s):.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            g.append(f'<text x="{xs(s) - 6:.1f}" y="{y - 8 - 14 * k:.1f}" text-anchor="end" font-size="11" '
                     f'font-weight="600" fill="{color}">{100 * (1 - s / end):.1f}% fewer steps</text>')
        g.append(f'<circle cx="{xs(end):.1f}" cy="{y:.1f}" r="3.5" fill="#000"/>')
    g.append("</svg>")
    return '<div class="legend-line">' + "".join(legend) + "</div>\n" + "\n".join(g)


D_BASE = ["mini-gpt-baseline", "drop1-base", "drop1-base-s1", "drop1-base-s2"]
D_FF = {
    "0.2": ["drop2-loo-gs384-ff0.2", "drop3-loo-gs384-ff0.2-s1", "drop3-loo-gs384-ff0.2-s2"],
    "0.4": ["drop2-loo-gs384-ff0.4", "drop3-loo-gs384-ff0.4-s1", "drop3-loo-gs384-ff0.4-s2"],
    "0.6": ["drop1-loo-gs384-ff0.6", "drop2-loo-gs384-ff0.6-s1", "drop2-loo-gs384-ff0.6-s2"],
    "0.8": ["drop1-loo-gs384-ff0.8", "drop1-loo-gs384-ff0.8-s1", "drop1-loo-gs384-ff0.8-s2"],
}
D_CHART = [
    ("leave_one_out, group 384, fix_factor 0.8 (best at dropout 0)", D_FF["0.8"], "#cf222e"),
    ("leave_one_out, group 384, fix_factor 0.6", D_FF["0.6"], "#9a6700"),
    ("leave_one_out, group 384, fix_factor 0.4", D_FF["0.4"], "#1a7f37"),
]


def dropout_section(d):
    def stats(names):
        names = [n for n in names if n in d]
        a = np.array([[np.mean(d[n]["train"][1:]), steps_to(d[n]["iter"], d[n]["train"], 1.5),
                       d[n]["train"][-1], min(d[n]["val"])] for n in names])
        return a

    def cell(a, j, f):
        s = f"{a[:, j].mean():{f}}"
        return s + (f' <span class="rng">({a[:, j].min():{f}}&ndash;{a[:, j].max():{f}})</span>' if len(a) > 1 else "")

    def row(label, names):
        a = stats(names)
        return (f'<tr><td>{label}</td><td class="num">{len(a)}</td>'
                + "".join(f'<td class="num">{cell(a, j, f)}</td>' for j, f in enumerate([".3f", ".0f", ".4f", ".4f"]))
                + "</tr>")

    final = "\n".join(row(l, n) for l, n in [
        ("Baseline: LayerNorm (<code>a9554bt7</code> + 3 reruns)", D_BASE),
        ("leave_one_out, group 384, fix_factor 0.8 (best at dropout 0)", D_FF["0.8"]),
        ("leave_one_out, group 384, fix_factor 0.6", D_FF["0.6"]),
        ("<b>leave_one_out, group 384, fix_factor 0.4 (best here)</b>", D_FF["0.4"]),
        ("leave_one_out, group 384, fix_factor 0.2", D_FF["0.2"]),
    ])
    m = lambda names: stats(names)[:, 0].mean()
    chips = {"yes": '<span class="chip ok">Helped</span>', "no": '<span class="chip diff">No gain</span>',
             "worse": '<span class="chip gap">Worse</span>'}
    tried = [
        ("Larger groups", "leave_one_out", "16 &rarr; 128 &rarr; 384",
         " &rarr; ".join(f"{m([n]):.3f}" for n in ["drop1-loo-gs16", "drop1-loo-gs128", "drop1-loo-gs384"]),
         "worse", "Reversed: at dropout 0 this was the biggest gain."),
        ("Larger groups", "running_mean", "16 &rarr; 48",
         f"{m(['drop1-rm-gs16']):.3f} &rarr; {m(['drop1-rm-gs48']):.3f}", "yes", "Still behind the baseline."),
        ("Higher momentum", "running_mean, group 384", "0.5 / 0.7",
         f"{m(['drop1-rm-gs384-mom0.5']):.3f} / {m(['drop1-rm-gs384-mom0.7']):.3f}", "worse", f"Best running_mean at dropout 0; here behind group 48 ({m(['drop1-rm-gs48']):.3f})."),
        ("More shrinkage", "leave_one_out, group 384", "0.9 / 0.8 / 0.6 / 0.4 / 0.2",
         " / ".join(f"{m(n):.3f}" for n in [["drop1-loo-gs384"], D_FF["0.8"], D_FF["0.6"], D_FF["0.4"], D_FF["0.2"]]),
         "yes", "Reaches the baseline at 0.2&ndash;0.4 (3 seeds each), does not pass it."),
        ("No decorrelation", "leave_one_out, group 384", "fix_factor 0.0",
         f"{m(['drop2-loo-gs384-ff0.0']):.3f}", "worse", "Per-channel scaling alone is worse; a little decorrelation is needed."),
        ("Shrinkage + smaller group", "leave_one_out, group 128 / 16", "fix_factor 0.6 / 0.4",
         f"{m(['drop2-loo-gs128-ff0.6']):.3f} / {m(['drop3-loo-gs128-ff0.4']):.3f} (128), {m(['drop2-loo-gs16-ff0.6']):.3f} (16)",
         "no", "Same as group 384 at matched shrinkage."),
        ("Shrinkage", "running_mean, group 48", "0.9 &rarr; 0.6",
         f"{m(['drop1-rm-gs48']):.3f} &rarr; {m(['drop2-rm-gs48-ff0.6']):.3f}", "no", ""),
    ]
    rows = "\n".join(
        f'<tr><td>{b}</td><td>{c}</td><td>{v}</td><td class="num">{r}</td>'
        f'<td class="s">{chips[h]}</td><td class="n">{n}</td></tr>'
        for b, c, v, r, h, n in tried)
    base, best = stats(D_BASE), stats(D_FF["0.4"])
    return f"""<!--DROPOUT-->
<h2 id="dropout">Experiments with dropout 0.2</h2>
<p class="sub">Question: does what helped at dropout 0 also help with nanoGPT's original dropout (0.2 on embeddings, attention, MLP and residual), against the LayerNorm baseline <code>a9554bt7</code>?
29 full 5000-step runs (wandb names <code>drop1-*</code>, <code>drop2-*</code>, <code>drop3-*</code>).</p>

<h3>Conclusions</h3>
<ul class="summary">
  <li><b>The dropout-0 gain does not carry over.</b> The best dropout-0 setting (<code>leave_one_out</code>, group 384, <code>fix_factor</code> 0.8) is ~7% worse in mean loss here, on all 3 seeds.</li>
  <li><b>BW at best ties the baseline.</b> With strong shrinkage (<code>fix_factor</code> 0.2&ndash;0.4), mean loss is {best[:, 0].mean():.3f} vs {base[:, 0].mean():.3f}, and best validation loss {best[:, 3].mean():.4f} vs {base[:, 3].mean():.4f}: within seed noise.</li>
  <li><b>Only an early lead remains.</b> These runs reach loss 1.5 ~5% sooner ({best[:, 1].mean():.0f} vs {base[:, 1].mean():.0f} steps), are even by step ~2000, and end ~2% higher in train loss.</li>
  <li><b>The levers flip.</b> Larger groups, higher momentum and weak shrinkage all hurt with dropout. Decorrelation must be kept small, but not zero (<code>fix_factor</code> 0 is worse).</li>
  <li><b>Not an eval-statistics problem.</b> Train-mode loss (dropout on, batch statistics) shows the same ranking, so optimization itself is slower, not just the stored statistics used at eval.</li>
  <li><b>Likely cause: noisy covariance.</b> Dropout noise makes the batch covariance estimate unreliable, so full decorrelation adds noise instead of conditioning the input well. Two follow-ups could test this: tie shrinkage to the noise level (strong early, weaker later, or scaled with the dropout rate), or remove dropout only on the MLP input, where BW sits.</li>
</ul>

<h3>Final results</h3>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">Mean loss</th><th class="num">Steps to loss 1.5</th><th class="num">Final loss</th><th class="num">Best val loss</th></tr>
{final}
</table>
<p class="cap">Mean over seeds, (min&ndash;max). Train losses measured in eval mode every 250 steps. Seed-to-seed spread of mean loss is about &plusmn;0.01 here, twice the dropout-0 spread.</p>

{chart(d, D_BASE, D_CHART, ylim=20, step=5)}
<p class="cap">Train loss relative to the baseline (mean of 4 runs); each line is a mean of 3 seeds. Below 0% = better than the baseline.</p>

{learning_curves(d, D_BASE, D_CHART, [0.5, 0.7, 1, 1.5, 2, 3])}
<p class="cap">Learning curves: train loss (eval mode, every 250 steps, mean over seeds) on a log axis.</p>

<h3>What we tried</h3>
<table>
<tr><th>Change</th><th>Mode</th><th>Values</th><th class="num">Mean loss</th><th>Result</th><th>Note</th></tr>
{rows}
</table>
<p class="cap">Result is relative to the previous setting. Baseline mean loss is {base[:, 0].mean():.3f} ({base[:, 0].min():.3f}&ndash;{base[:, 0].max():.3f}).</p>

<h3>Recommendations</h3>
<ul class="summary">
  <li><b>With dropout:</b> no BW setting is worth using over LayerNorm; if BW is used, take <code>leave_one_out</code>, <code>bw_fix_factor=0.4</code>.</li>
  <li><b>Next:</b> make shrinkage depend on noise (e.g. strong early, weaker late, or tied to dropout rate); try BW without dropout on the MLP input, where BW sits.</li>
</ul>
<!--/DROPOUT-->"""


GB_COLS = [0.1, 0.25, 0.5, 0.75, 1, 1.43, 2, 5, 10, 15]
GB_FF04_05 = ["gb2-drop-loo-gs384-ff0.4-gb0.5", "gb3-drop-loo-gs384-ff0.4-gb0.5-s1", "gb3-drop-loo-gs384-ff0.4-gb0.5-s2"]
GB_ROWS = [
    # (section, label, {boost: run names}, diverged-above threshold or None)
    ("Dropout 0", "running_mean, group 16", {1: ["exp1-rm-gs16-ref"], 2: ["gb1-rm-gs16-gb2"],
                                             5: ["gb1-rm-gs16-gb5"], 10: ["gb1-rm-gs16-gb10"]}, 1.0),
    ("Dropout 0", "running_mean, group 384, momentum 0.7",
     {1: ["exp3-rm-gs384-mom0.7"], 1.43: ["gb1-rm-gs384-mom0.7-gb1.43"], 2: ["gb1-rm-gs384-mom0.7-gb2"],
      5: ["gb1-rm-gs384-mom0.7-gb5"]}, 1.0),
    ("Dropout 0", "leave_one_out, group 384, fix_factor 0.8 (best)",
     {0.5: ["gb1-loo-gs384-ff0.8-gb0.5"], 1: MAIN[7][1], 2: ["gb1-loo-gs384-ff0.8-gb2"]}, 1.0),
    ("Dropout 0", "calibrated, group 16, blended centering (old default)",
     {1: ["gb1-cal-gs16-gb1"], 5: ["gb1-cal-gs16-gb5"], 15: ["gb1-cal-gs16-gb15"]}, 1.0),
    ("Dropout 0", "calibrated, group 16, pre-<code>cdcba46</code> covariance centering, no warmup",
     {10: ["gb2-cal-gs16-oldcov-gb10-nowarm"]}, 1.0),
    ("Dropout 0", "calibrated, group 16, leave-one-out centering (new default)",
     {1: ["gb2-cal-gs16-loo-gb1"], 10: ["gb2-cal-gs16-loo-gb10"]}, 1.0),
    ("Dropout 0", "calibrated, group 384, blended centering (old default)", {1: ["gb1-cal-gs384-gb1"]}, 1.0),
    ("Dropout 0", "calibrated, group 384, leave-one-out centering, fix_factor 0.8",
     {10: ["gb4-cal-gs384-loo-gb10-ff0.8"]}, 1.0),
    ("Dropout 0.2", "leave_one_out, group 384, fix_factor 0.4 (best)",
     {0.1: ["gb3-drop-loo-gs384-ff0.4-gb0.1"], 0.25: ["gb3-drop-loo-gs384-ff0.4-gb0.25"], 0.5: GB_FF04_05,
      0.75: ["gb2-drop-loo-gs384-ff0.4-gb0.75"], 1: D_FF["0.4"]}, None),
    ("Dropout 0.2", "leave_one_out, group 384, fix_factor 0.6",
     {0.5: ["gb3-drop-loo-gs384-ff0.6-gb0.5"], 1: D_FF["0.6"]}, None),
]


def boost_section(d):
    def ml(names):
        names = [n for n in names if n in d]
        return (float(np.mean([np.mean(d[n]["train"][1:]) for n in names])), len(names)) if names else (None, 0)

    def stats(names):
        names = [n for n in names if n in d]
        return np.array([[np.mean(d[n]["train"][1:]), steps_to(d[n]["iter"], d[n]["train"], 0.8),
                          d[n]["train"][-1], min(d[n]["val"])] for n in names])

    head = "".join(f'<th class="num">{c:g}</th>' for c in GB_COLS)
    rows, section = [], None
    for sec, label, cells, div in GB_ROWS:
        if sec != section:
            rows.append(f'<tr class="baseline"><td colspan="{len(GB_COLS) + 1}"><b>{sec}</b></td></tr>')
            section = sec
        vals = {b: ml(n) for b, n in cells.items()}
        ok = [v for v, _ in vals.values() if v is not None and (div is None or v < div)]
        best = min(ok) if len(ok) > 1 else None
        tds = []
        for c in GB_COLS:
            v, n = vals.get(c, (None, 0))
            if v is None:
                tds.append('<td class="num">' + ("running" if c in vals else "") + "</td>")
                continue
            cls = " better" if v == best else (" bad" if div is not None and v >= div else "")
            seeds = f' <span class="rng">&times;{n}</span>' if n > 1 else ""
            tds.append(f'<td class="num{cls}">{v:.3f}{seeds}</td>')
        rows.append(f"<tr><td>{label}</td>{''.join(tds)}</tr>")

    base, b1, b05 = stats(D_BASE), stats(D_FF["0.4"]), stats(GB_FF04_05)

    def frow(label, a):
        rng = lambda j, f: f"{a[:, j].mean():{f}}" + (f' <span class="rng">({a[:, j].min():{f}}&ndash;{a[:, j].max():{f}})</span>' if len(a) > 1 else "")
        return (f'<tr><td>{label}</td><td class="num">{len(a)}</td><td class="num">{rng(0, ".3f")}</td>'
                f'<td class="num">{rng(1, ".0f")}</td><td class="num">{rng(2, ".4f")}</td><td class="num">{rng(3, ".4f")}</td></tr>')

    final = "\n".join([frow("Baseline: LayerNorm (<code>a9554bt7</code> + 3 reruns)", base),
                       frow("leave_one_out, group 384, fix_factor 0.4, boost 1", b1),
                       frow("<b>leave_one_out, group 384, fix_factor 0.4, boost 0.5</b>", b05)])
    series = [("leave_one_out, group 384, fix_factor 0.4, boost 1", D_FF["0.4"], "#9a6700"),
              ("leave_one_out, group 384, fix_factor 0.4, boost 0.5", GB_FF04_05, "#1a7f37")]
    m = lambda names: ml(names)[0]
    cal384 = m(["gb4-cal-gs384-loo-gb10-ff0.8"])
    cal384_txt = ("still running" if cal384 is None else
                  f"mean loss {cal384:.3f}, worse than the baseline ({m(BASE):.3f}) and plain "
                  f"<code>leave_one_out</code> ({m(MAIN[7][1]):.3f}); pooling 10 calibration batches does not "
                  "substitute for the fresh full-gradient covariance")
    n_runs = sum(1 for n in d if n.startswith("gb"))
    return f"""<!--BOOST-->
<h2 id="boost">Experiments with the gradient boost</h2>
<p class="sub">Question: does scaling the gradient of the current-batch BW statistics (forward pass unchanged) improve on the best configurations above?
New knob <code>bw_grad_scale</code> (0 = per-mode default: 1/momentum for <code>calibrated</code>, 1 otherwise). It scales the covariance gradient in every mode, and also the centering-mean gradient in <code>calibrated</code>.
{n_runs} full 5000-step runs (wandb names <code>gb1-*</code> &hellip; <code>gb4-*</code>).</p>

<h3>Conclusions</h3>
<ul class="summary">
  <li><b>Without dropout, no boost for <code>running_mean</code> or <code>leave_one_out</code>.</b> Boost 1 (the true gradient) is best in every configuration. Any boost above 1 hurts, and 2 already makes the best group-384 configurations diverge. Boost 0.5 (tested on <code>leave_one_out</code>) also hurts.</li>
  <li><b>With dropout, a damped boost helps a little.</b> Boost 0.5 on <code>leave_one_out</code>, group 384, <code>fix_factor</code> 0.4 gives mean loss {b05[:, 0].mean():.3f} (3 seeds) vs {base[:, 0].mean():.3f} for the baseline and {b1[:, 0].mean():.3f} with boost 1, and reaches loss 0.8 ~{100 * (1 - b05[:, 1].mean() / base[:, 1].mean()):.0f}% sooner. This is the first BW setting that beats the dropout baseline on every seed. Best validation loss is unchanged ({b05[:, 3].mean():.4f} vs {base[:, 3].mean():.4f}), and in train mode (dropout on) the gain shrinks to a tie: boost 0.5 mainly stops the late-training drift that boost 1 shows.</li>
  <li><b>Blended-mean centering breaks <code>calibrated</code>; the boost is not the cause.</b> With data and covariance centered about the blended mean (<code>cdcba46</code>, now <code>bw_calib_center=blend</code>), activations run away at every boost (input RMS 150&ndash;21,000 by step 5000 at group 16, mean loss 1.7&ndash;2.4). Reverting the covariance centering restores the September behaviour (0.696, stable). <b>Fixed:</b> the default is now <code>bw_calib_center=loo</code>.</li>
  <li><b>With leave-one-out centering, the boost does what the theory says for <code>calibrated</code>.</b> Boost 10 (= 1/momentum) beats boost 1: {m(['gb2-cal-gs16-loo-gb10']):.3f} vs {m(['gb2-cal-gs16-loo-gb1']):.3f} at group 16, the best <code>calibrated</code> result so far. It only ties plain <code>leave_one_out</code> at group 16 ({m(['mini-gpt-no-dropout-loo-groupsize-16']):.3f}) at ~7&times; the cost. Group 384: {cal384_txt}.</li>
</ul>

<h3>Mean loss by boost</h3>
<table>
<tr><th>Configuration</th>{head}</tr>
{chr(10).join(rows)}
</table>
<p class="cap">Mean train loss over the run (eval mode; lower is better). Columns are the boost level; 1 = no boost. Single seed unless marked &times;3 (mean of 3 seeds); seed spread is about &plusmn;0.005 without dropout and &plusmn;0.01 with it.
Green = best in the row; red = diverged or stalled (mean loss &ge; 1.0 without dropout). Dropout-0 baseline: {m(BASE):.3f}; dropout-0.2 baseline: {base[:, 0].mean():.3f}.</p>

<h3>Final results with dropout 0.2</h3>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">Mean loss</th><th class="num">Steps to loss 0.8</th><th class="num">Final loss</th><th class="num">Best val loss</th></tr>
{final}
</table>

{chart(d, D_BASE, series, ylim=10, step=5)}
<p class="cap">Train loss relative to the dropout-0.2 baseline (mean of 4 runs); each line is a mean of 3 seeds. Below 0% = better than the baseline.</p>

{learning_curves(d, D_BASE, series, [0.5, 0.7, 1, 1.5, 2, 3])}
<p class="cap">Learning curves: train loss (eval mode, every 250 steps, mean over seeds) on a log axis.</p>

<h3>Recommendations</h3>
<ul class="summary">
  <li><b>Keep boost 1</b> for <code>running_mean</code> and <code>leave_one_out</code> without dropout (the current default).</li>
  <li><b>With dropout:</b> <code>leave_one_out</code>, group 384, <code>bw_fix_factor=0.4</code>, <code>bw_grad_scale=0.5</code>.</li>
  <li><b><code>calibrated</code>:</b> use the new default, leave-one-out centering (<code>bw_calib_center=loo</code>) with boost 1/momentum. Do not use <code>blend</code>.</li>
</ul>
<!--/BOOST-->"""


def main():
    if sys.argv[1] == "--boost":
        d = {}
        for f in sys.argv[2:]:
            d.update(json.load(open(f)))
        html = boost_section(d)
        s = open(REPORT).read()
        if "<!--BOOST-->" in s:
            s = re.sub(r"<!--BOOST-->.*?<!--/BOOST-->", lambda _: html, s, count=1, flags=re.S)
        else:
            s = s.replace("<!--/DROPOUT-->", "<!--/DROPOUT-->\n" + html, 1)
        open(REPORT, "w").write(s)
        print("wrote", REPORT)
        return
    if sys.argv[1] == "--dropout":
        html = dropout_section(json.load(open(sys.argv[2])))
        s = open(REPORT).read()
        if "<!--DROPOUT-->" in s:
            s = re.sub(r"<!--DROPOUT-->.*?<!--/DROPOUT-->", lambda _: html, s, count=1, flags=re.S)
        else:
            s = s.replace("<!--/RESULTS-->", "<!--/RESULTS-->\n" + html, 1)
        open(REPORT, "w").write(s)
        print("wrote", REPORT)
        return
    d = json.load(open(sys.argv[1]))
    m = lambda names: metrics(d, names)[0][4]
    chips = {"yes": '<span class="chip ok">Helped</span>', "no": '<span class="chip diff">No gain</span>',
             "worse": '<span class="chip gap">Worse</span>'}
    tried = [
        ("Adaptive block size", "Larger groups", "leave_one_out", "16 &rarr; 384",
         f"{m(['mini-gpt-no-dropout-loo-groupsize-16']):.3f} &rarr; {m(MAIN[6][1]):.3f}", "yes", "Biggest single gain."),
        ("Adaptive block size", "Larger groups", "running_mean", "16 &rarr; 48 &rarr; 384",
         f"{m(['exp1-rm-gs16-ref']):.3f} &rarr; {m(['exp1-rm-gs48']):.3f} &rarr; {m(['exp2-rm-gs384']):.3f}", "yes", "48 is the paper's rule; 384 needs higher momentum."),
        ("Adaptive block size", "Larger groups", "calibrated", "16 &rarr; 384",
         "0.660 &rarr; diverged", "worse", "Activations grew ~16,000&times;; stopped at step 3500."),
        ("Adaptive momentum", "Higher momentum", "running_mean, group 384", "0.1 / 0.3 / 0.5 / 0.7",
         " / ".join(f"{m([n]):.3f}" for n in ["exp2-rm-gs384", "exp3-rm-gs384-mom0.3", "exp2-rm-gs384-mom0.5", "exp3-rm-gs384-mom0.7"]), "yes", "Steady gain toward leave_one_out (1.0)."),
        ("Adaptive momentum", "Other momentum", "running_mean, group 16", "0.03 / 0.1 / 0.3 / 0.5",
         " / ".join(f"{m([n]):.3f}" for n in ["exp2-rm-gs16-mom0.03", "exp1-rm-gs16-ref", "exp1-rm-mom0.3", "exp1-rm-mom0.5"]), "no", "0.1 (current) is best at group 16."),
        ("Correlated channels (&epsilon;)", "Less shrinkage", "running_mean, group 16", "0.9 / 0.99 / 1.0",
         " / ".join(f"{m([n]):.3f}" for n in ["exp1-rm-gs16-ref", "exp1-rm-ff0.99", "exp1-rm-ff1.0"]), "no", "Paper's &epsilon; &rarr; 0 does not help."),
        ("Correlated channels (&epsilon;)", "Less shrinkage", "leave_one_out, group 384", "0.9 &rarr; 0.95",
         f"{m(MAIN[6][1]):.3f} &rarr; {m(['exp2-loo-gs384-ff0.95']):.3f}", "worse", "Unstable (final loss 0.27)."),
        ("Correlated channels (&epsilon;)", "More shrinkage", "leave_one_out, group 384", "0.9 &rarr; 0.8 / 0.7 / 0.6",
         f"{m(MAIN[6][1]):.3f} &rarr; " + " / ".join(f"{m(n):.3f}" for n in [MAIN[7][1], ["exp3-loo-gs384-ff0.7"], ["exp3-loo-gs384-ff0.6"]]), "yes", "Small gain; 0.8 confirmed on 3 seeds."),
        ("Warmup handoff", "EMA buffer during warmup / no warmup", "running_mean, group 16", "diag / diag+EMA / none",
         " / ".join(f"{m([n]):.3f}" for n in ["exp1-rm-gs16-ref", "exp1-rm-warm-ema", "exp1-rm-warm-none"]), "no", "The handoff is not what limits it."),
        ("Inference statistics", "EMA instead of last batch", "leave_one_out, group 128", "off &rarr; on",
         f"0.476 &rarr; {m(['exp1-loo-gs128-infema']):.3f}", "worse", "EMA statistics lag the model."),
    ]
    rows = "\n".join(
        f'<tr><td class="f">{a}</td><td>{b}</td><td>{c}</td><td>{v}</td><td class="num">{r}</td>'
        f'<td class="s">{chips[h]}</td><td class="n">{n}</td></tr>'
        for a, b, c, v, r, h, n in tried)

    def final_row(label, names):
        per = []
        for n in names:
            its, tr = d[n]["iter"], d[n]["train"]
            per.append([np.mean(tr[1:]), tr[-1], steps_to(its, tr, 1.0), steps_to(its, tr, 0.1)])
        a = np.array(per)
        rng = lambda j, f: f"{a[:, j].mean():{f}}" + (f' <span class="rng">({a[:, j].min():{f}}&ndash;{a[:, j].max():{f}})</span>' if len(a) > 1 else "")
        return (f"<tr><td>{label}</td><td class=\"num\">{len(a)}</td><td class=\"num\">{rng(0, '.3f')}</td>"
                f"<td class=\"num\">{rng(1, '.4f')}</td><td class=\"num\">{rng(2, '.0f')}</td><td class=\"num\">{rng(3, '.0f')}</td></tr>")

    final = "\n".join(final_row(l, n) for l, n in [
        ("Baseline: LayerNorm", BASE),
        ("running_mean, group 16 (current default)", ["exp1-rm-gs16-ref"]),
        ("running_mean, group 384, momentum 0.7 (best running_mean)", ["exp3-rm-gs384-mom0.7"]),
        ("leave_one_out, group 384", MAIN[6][1]),
        ("<b>leave_one_out, group 384, fix_factor 0.8 (best)</b>", MAIN[7][1]),
    ])
    base = metrics(d, BASE)[0]
    best = metrics(d, MAIN[7][1])[0]
    html = f"""<!--RESULTS-->
<h2 id="experiments">Experiments and conclusions</h2>
<p class="sub">Question: does fixing the paper/code differences make BW beat the no-dropout LayerNorm baseline (<code>l965qcs2</code>) in training convergence?
35 full 5000-step runs on 1 GPU each (wandb names <code>exp1-*</code>, <code>exp2-*</code>, <code>exp3-*</code>).</p>

<h3>Conclusions</h3>
<ul class="summary">
  <li><b>BW beats the baseline.</b> <code>leave_one_out</code> + full-channel whitening (group 384) + <code>fix_factor</code> 0.8: 15&ndash;22% lower train loss through step 2000, about even at steps 2500&ndash;3250, 3&ndash;5% lower at the end. Holds on all 3 seeds.</li>
  <li><b>Group size is what matters.</b> The paper's error budget (e<sub>max</sub>&nbsp;=&nbsp;0.1) is far too conservative: group 384 has e&nbsp;&asymp;&nbsp;2.1 per batch and trains best.</li>
  <li><b>Little history is best.</b> At large groups, <code>running_mean</code> improves as momentum rises toward 1 (= <code>leave_one_out</code>).</li>
  <li><b>Keep shrinkage.</b> &epsilon;&nbsp;=&nbsp;0.1&ndash;0.2 works; the paper's &epsilon;&nbsp;&rarr;&nbsp;0 is neutral at group 16 and unstable at group 384.</li>
  <li><b><code>calibrated</code> does not scale.</b> It diverged at group 384 with the old blended-mean centering; with the new leave-one-out centering it trains but stays behind (see the gradient-boost section). It also costs ~7&times; run time.</li>
</ul>

<h3>Final results</h3>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">Mean loss</th><th class="num">Final loss</th><th class="num">Steps to loss 1.0</th><th class="num">Steps to loss 0.1</th></tr>
{final}
</table>
<p class="cap">Mean over seeds, (min&ndash;max) when there are several. Mean loss = average train loss over all evals (area under the curve). Lower is better everywhere.</p>

{chart(d)}
<p class="cap">Train loss relative to the baseline (mean of 3 seeds), measured in eval mode every 250 steps. Below 0% = better than the baseline. Clipped to &plusmn;40%.</p>

{learning_curves(d, BASE, CHART, [0.05, 0.1, 0.2, 0.5, 1, 2, 3], accel=True)}
<p class="cap">Learning curves: train loss (eval mode, every 250 steps, mean over seeds) on a log axis.
Dashed lines: the step at which each BW configuration first reaches the baseline's final loss ({speedups(d, BASE, CHART)[0]:.4f} at step 5000; black dot).
Configurations that never reach it have no line.</p>

<h3>What we tried</h3>
<table>
<tr><th>Paper difference</th><th>Change</th><th>Mode</th><th>Values</th><th class="num">Mean loss</th><th>Result</th><th>Note</th></tr>
{rows}
</table>
<p class="cap">Single seed unless stated. Seed-to-seed spread of mean loss is about &plusmn;0.005, so smaller differences are noise.</p>

<h3>Recommendations</h3>
<ul class="summary">
  <li><b>New default:</b> <code>leave_one_out</code>, <code>bw_group_size=384</code>, <code>bw_fix_factor=0.8</code>.</li>
  <li><b>If history is wanted:</b> <code>running_mean</code>, group 384, momentum 0.7.</li>
  <li><b>Next:</b> give <code>calibrated</code> leave-one-out centering; confirm at GPT-2 scale; update the paper's block-size and momentum guidance.</li>
</ul>

<details>
<summary>All configurations (convergence detail)</summary>
{main_table(d)}
<p class="cap">Steps to reach each train loss (linear interpolation between evals). Bold green = better than the baseline. Validation loss is not compared: with no dropout every configuration overfits (val 4.2&ndash;4.7).</p>
</details>
<!--/RESULTS-->"""
    s = open(REPORT).read()
    s = re.sub(r"<!--RESULTS-->.*?(<!--/RESULTS-->|$)", lambda _: html, s, count=1, flags=re.S) \
        if "<!--/RESULTS-->" in s else s.replace("<!--RESULTS-->", html)
    open(REPORT, "w").write(s)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
