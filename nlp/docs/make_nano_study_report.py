"""Build bw_nanogpt_study.html: the post-fix nanoGPT study (causal BW, renorm).

Usage: python make_nano_study_report.py <repo_root>
Reads <repo_root>/nlp/study/logs/*.out (written by nlp/sbatch_nanogpt_pack.sh).
"""
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nano_study import mean_train, parse, steps_to  # noqa: E402

OUT = os.path.join(HERE, "bw_nanogpt_study.html")
STYLE = re.search(r"<style>.*?</style>", open(os.path.join(HERE, "bw_paper_vs_code.html")).read(), re.S).group(0)
LOGS = os.path.join(sys.argv[1], "nlp", "study", "logs")
R = {f[:-4]: parse(os.path.join(LOGS, f)) for f in os.listdir(LOGS) if f.endswith(".out")}

BASE = "s1-ln"
LN_SEEDS = ["s1-ln", "s3-ln-s1", "s3-ln-s2"]
BEST_SEEDS = ["s3-best-lr1.5e-3", "s4-best-lr1.5e-3-s1", "s4-best-lr1.5e-3-s2"]
LR2_SEEDS = ["s2-rm-nw-lr2e-3-ff0.8", "s3-best-s1", "s3-best-s2"]
D_LN_SEEDS = ["s3-d-ln", "s5-d-ln-s1", "s5-d-ln-s2"]
D_BEST_SEEDS = ["s4-d-ff0.4-m0.3", "s5-d-m0.3-s1", "s5-d-m0.3-s2"]
D10_LN = ["s6-d10k-ln-s1", "s6-d10k-ln-s2"]
D10_BW = ["s6-d10k-bw-s1", "s6-d10k-bw-s2"]


def rel(name, base=BASE):
    r = R.get(name)
    if r is None or not r["ev"]:
        return None
    upto = max(r["ev"])
    return 100 * (mean_train(r["ev"], upto) / mean_train(R[base]["ev"], upto) - 1)


def cell(v, status=""):
    if v is None:
        return '<td class="num">&mdash;</td>'
    txt = f"{v:+.1f}%".replace("-", "&minus;")
    cls = " better" if v < 0 and not status else ""
    note = f' <span class="rng">({status})</span>' if status else ""
    return f'<td class="num{cls}">{txt}{note}</td>'


def status(name):
    r = R[name]
    if r["abort"]:
        return f"diverged, stopped at {r['abort']}"
    if max(r["ev"]) < 5000:
        return f"stopped at {max(r['ev'])}"
    return ""


# question, rows (label, run name, marker of the chosen value), conclusion
STUDY = [
    ("Mode (statistics in the stored buffers)", [
        ("<code>running_mean</code>, momentum 0.7", "s1-rm-m0.7", True),
        ("<code>leave_one_out</code> (= momentum 1)", "s1-loo", False),
        ("<code>calibrated</code>, 2 calibration batches, gradient weight 0.7", "s2-cal-m0.7-nb2", False),
        ("<code>calibrated</code>, default (10 batches, boost 10)", "s1-cal-nb10", False),
     ], "<code>running_mean</code>. With renorm the modes differ only in what the buffers hold; "
        "<code>leave_one_out</code> is <code>running_mean</code> with momentum 1. "
        "<code>calibrated</code> at best ties the baseline at 2.5&ndash;7&times; the cost: drop it."),
    ("Momentum (no warmup)", [
        ("0.5", "s2-rm-nw-m0.5", False), ("0.7", "s1-rm-nowarm", True),
        ("0.85", "s2-rm-nw-m0.85", False), ("1.0 (<code>leave_one_out</code>)", "s1-loo", False),
     ], "0.7: the buffer is 70% the last batch. Less history is worse, but some history helps."),
    ("Warmup (diagonal whitening for the first 250 steps)", [
        ("on (<code>diag</code>)", "s1-rm-m0.7", False), ("off", "s1-rm-nowarm", True),
        ("on, best recipe", "s3-best-warm", False), ("off, best recipe", "s2-rm-nw-lr2e-3-ff0.8", True),
     ], "Off. With renorm the forward pass already uses stored statistics, so the diagonal start only delays the whitening."),
    ("Gradient boost (<code>bw_grad_scale</code>)", [
        ("<code>running_mean</code>: 0.5", "s1-rm-gb0.5", False), ("1", "s1-rm-m0.7", True),
        ("1.43 (= 1/momentum)", "s1-rm-gb1.43", False), ("3", "s1-rm-gb3", False),
        ("<code>leave_one_out</code>: 0.5", "s1-loo-gb0.5", False), ("1", "s1-loo", True),
        ("2", "s1-loo-gb2", False),
        ("<code>calibrated</code>: 1 (weight 0.1)", "s1-cal-nb10-gb1", False),
        ("10 (default)", "s1-cal-nb10", False),
     ], "No boost. The current batch's natural gradient weight (momentum for <code>running_mean</code>, 1 for "
        "<code>leave_one_out</code>) is best; scaling it up diverges in every mode, scaling it down is worse."),
    ("Depth (which blocks get BW)", [
        ("all 6", "s1-rm-m0.7", True), ("first 1", "s1-rm-L1", False), ("first 2", "s1-rm-L2", False),
        ("first 3", "s1-rm-L3", False), ("last 3", "s1-rm-last3", False),
        ("first 2, best recipe", "s2-rm-nw-lr2e-3-L2", False),
     ], "All blocks. Unlike GPT-2, nanoGPT gains from BW at every depth."),
    ("Site", [
        ("MLP input, BW &rarr; LayerNorm (current)", "s1-rm-m0.7", True),
        ("MLP and attention inputs", "s1-rm-both", False),
        ("attention input only", "s1-rm-attn", False),
        ("MLP input, no LayerNorm after BW", "s1-rm-noln", False),
     ], "MLP input with the LayerNorm kept. BW on the attention input hurts."),
    ("Group size and <code>fix_factor</code>", [
        ("group 384, ff 0.9", "s1-rm-m0.7", False), ("group 384, ff 0.8", "s1-rm-ff0.8", False),
        ("group 384, ff 0.6", "s1-rm-ff0.6", False), ("group 128, ff 0.9", "s1-rm-gs128", False),
        ("best recipe, ff 0.8", "s2-rm-nw-lr2e-3-ff0.8", True), ("best recipe, ff 0.7", "s3-best-ff0.7", False),
        ("best recipe, group 192", "s2-rm-nw-lr2e-3-gs192", False),
     ], "Full width (384), ff 0.8."),
    ("Learning rate", [
        ("LayerNorm 5e-4", "s2-ln-lr5e-4", False), ("LayerNorm 1e-3 (baseline)", "s1-ln", True),
        ("LayerNorm 1.5e-3", "s2-ln-lr1.5e-3", False), ("LayerNorm 2e-3", "s1-ln-lr2e-3", False),
        ("BW 1e-3 (ff 0.9)", "s1-rm-nowarm", False), ("BW 1.5e-3 (ff 0.8)", "s3-best-lr1.5e-3", True),
        ("BW 2e-3 (ff 0.8)", "s2-rm-nw-lr2e-3-ff0.8", False), ("BW 2.5e-3 (ff 0.8)", "s3-best-lr2.5e-3", False),
        ("BW 3e-3", "s2-rm-nw-lr3e-3", False),
     ], "BW trains well at 1.5&ndash;2&times; the baseline's learning rate, where LayerNorm gets worse; "
        "the baseline is at its own best (1e-3). 2.5e-3 is already past BW's edge, so 1.5e-3 is the safe choice."),
]


def study_table():
    out = []
    for q, rows, concl in STUDY:
        rows = [r for r in rows if r[1] in R]
        for i, (label, name, chosen) in enumerate(rows):
            lab = f"<b>{label}</b>" if chosen else label
            first = (f'<td rowspan="{len(rows)}">{q}</td>' if i == 0 else "")
            last = (f'<td rowspan="{len(rows)}">{concl}</td>' if i == 0 else "")
            out.append(f"<tr>{first}<td>{lab}</td>{cell(rel(name), status(name))}{last}</tr>")
    return "\n".join(out)


def seed_metrics(names, low=0.1, last=5000, high=1.0):
    names = [n for n in names if n in R and last in R[n]["ev"]]
    a = np.array([[mean_train(R[n]["ev"], last), R[n]["ev"][last][0], steps_to(R[n]["ev"], high),
                   steps_to(R[n]["ev"], low), min(v for _, v in R[n]["ev"].values())] for n in names])
    return names, a


def seeds_table(groups, base_names, low=0.1, last=5000, high=1.0):
    rows = []
    _, base = seed_metrics(base_names, low, last, high)
    for label, names, ms in groups:
        n, a = seed_metrics(names, low, last, high)
        if not n:
            continue
        m, lo, hi = a.mean(0), a.min(0), a.max(0)
        cells = []
        for j, f in enumerate((".3f", ".4f", ".0f", ".0f", ".3f")):
            d = 100 * (m[j] / base.mean(0)[j] - 1)
            is_base = names is base_names
            dtxt = "" if is_base else f" ({d:+.0f}%)".replace("-", "&minus;")
            better = not is_base and d < 0 and j < 4
            cells.append(f'<td class="num{" better" if better else ""}">{m[j]:{f}}{dtxt} '
                         f'<span class="rng">{lo[j]:{f}}&ndash;{hi[j]:{f}}</span></td>')
        rows.append(f"<tr><td>{label}</td><td class=num>{len(n)}</td><td class=num>{ms}</td>{''.join(cells)}</tr>")
    return "\n".join(rows)


def final_table():
    return seeds_table([
        ("LayerNorm, lr 1e-3 (baseline)", LN_SEEDS, "14.0"),
        ("<b>BW, lr 1.5e-3 (recommended)</b>", BEST_SEEDS, "27.7"),
        ("BW, lr 2e-3", LR2_SEEDS, "27.7"),
    ], LN_SEEDS)


def dropout_table():
    return seeds_table([
        ("LayerNorm, lr 1e-3 (vanilla nanoGPT)", D_LN_SEEDS, "14.0"),
        ("<b>BW, momentum 0.3, ff 0.4, lr 2e-3</b>", D_BEST_SEEDS, "27.7"),
    ], D_LN_SEEDS, low=0.7)


def dropout_10k_table():
    return seeds_table([
        ("LayerNorm, lr 1e-3", D10_LN, "14.0"),
        ("LayerNorm, lr 2e-3", ["s6-d10k-ln-lr2e-3"], "14.0"),
        ("<b>BW, momentum 0.3, ff 0.4, lr 2e-3</b>", D10_BW, "27.7"),
    ], D10_LN, low=0.3, last=10000, high=0.5)


def catch_up(ln, bw, last):
    """Step at which BW's seed-mean train loss first reaches LayerNorm's seed-mean final train loss."""
    target = np.mean([R[n]["ev"][last][0] for n in ln])
    steps = sorted(set.intersection(*(set(R[n]["ev"]) for n in bw)))
    st = steps_to({s: (np.mean([R[n]["ev"][s][0] for n in bw]), 0) for s in steps}, target)
    return (f"BW (best) reaches LayerNorm's final train loss ({target:.3f}) at step {st:.0f}, "
            f"{100 * (1 - st / last):.0f}% fewer steps ({last / st:.2f}&times; faster per step; "
            f"at 2&times; the cost per step, about {2 * st / last:.2f}&times; LayerNorm's wall-clock time).")


def gap_table(ln, bw, steps):
    out = []
    for st in steps:
        a = np.mean([R[n]["ev"][st][0] for n in ln])
        b = np.mean([R[n]["ev"][st][0] for n in bw])
        out.append(f"<tr><td class=num>{st}</td><td class=num>{a:.3f}</td><td class=num>{b:.3f}</td>"
                   f"{cell(100 * (b / a - 1))}</tr>")
    return "\n".join(out)


def dropout_variants():
    rows = []
    for label, name in (("LayerNorm, lr 2e-3", "s3-d-ln-lr2e-3"),
                        ("BW dropout-0 recipe (momentum 0.7, ff 0.8, lr 2e-3)", "s3-d-best"),
                        ("momentum 0.7, ff 0.4", "s3-d-best-ff0.4"),
                        ("momentum 0.7, ff 0.2", "s4-d-ff0.2"),
                        ("momentum 0.3, ff 0.8", "s4-d-ff0.8-m0.3"),
                        ("<b>momentum 0.3, ff 0.4</b>", "s4-d-ff0.4-m0.3"),
                        ("momentum 0.1, ff 0.4", "s5-d-m0.1"),
                        ("momentum 0.3, ff 0.4, lr 1.5e-3", "s5-d-m0.3-lr1.5e-3"),
                        ("momentum 0.7, ff 0.4, lr 1e-3", "s4-d-ff0.4-lr1e-3")):
        if name in R and R[name]["ev"]:
            rows.append(f"<tr><td>{label}</td>{cell(rel(name, 's3-d-ln'), status(name))}</tr>")
    return "\n".join(rows)


def cond_table():
    rows = []
    for label, name in (("LayerNorm", "s1-ln"), ("BW, warmup on", "s1-rm-m0.7"),
                        ("BW, warmup off", "s1-rm-nowarm"), ("BW <code>leave_one_out</code>", "s1-loo"),
                        ("BW group 128 (worse)", "s1-rm-gs128"), ("BW best recipe", "s3-best-lr1.5e-3")):
        s = open(os.path.join(LOGS, name + ".out")).read()
        cells = []
        for st in (250, 1000, 5000):
            m = re.search(rf"^step {st}: cond mlp_in [\d.e+]+ \[(.*)\]", s, re.M)
            v = [float(x) for x in m.group(1).split(",")]
            cells.append(f"<td class=num>{v[0]:.0e}</td><td class=num>{math.exp(np.mean(np.log(v[1:]))):.0f}</td>")
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}{cell(rel(name))}</tr>")
    return "\n".join(rows)


def chart():
    W, H, L, Rm, T, B = 760, 280, 56, 16, 16, 40
    ylim = 30
    xs = lambda i: L + (W - L - Rm) * i / 5000
    ys = lambda v: T + (H - T - B) * (ylim - max(-ylim, min(ylim, v))) / (2 * ylim)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Train loss relative to LayerNorm">']
    for v in range(-ylim, ylim + 1, 10):
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#d1d9e0"/>')
        g.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#59636e">{v:+d}%</text>')
    g.append(f'<line x1="{xs(0)}" x2="{xs(5000)}" y1="{ys(0):.1f}" y2="{ys(0):.1f}" stroke="#000" stroke-width="2.5"/>')
    for i in range(0, 5001, 1000):
        g.append(f'<text x="{xs(i):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="#59636e">{i}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2}" y="{H - 6}" text-anchor="middle" font-size="12">Iteration</text>')
    ln = {st: np.mean([R[n]["ev"][st][0] for n in LN_SEEDS if n in R and st in R[n]["ev"]]) for st in range(250, 5001, 250)}
    legend = ['<span><i style="background:#000"></i>LayerNorm (0%)</span>']
    for label, names, color in (("BW, renorm default (warmup on, lr 1e-3)", ["s1-rm-m0.7"], "#afb8c1"),
                                ("BW, warmup off", ["s1-rm-nowarm"], "#0969da"),
                                ("BW best recipe (3 seeds)", BEST_SEEDS, "#1a7f37")):
        names = [n for n in names if n in R]
        pts = []
        for st in range(250, 5001, 250):
            v = [R[n]["ev"][st][0] for n in names if st in R[n]["ev"]]
            if v:
                pts.append(f"{xs(st):.1f},{ys(100 * (np.mean(v) / ln[st] - 1)):.1f}")
        g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{label}</span>')
    g.append("</svg>")
    return '<div class="legend-line">' + "".join(legend) + "</div>\n" + "\n".join(g)


def curves(lines, ymin, ymax, ticks, label, xmax=5000):
    """Learning curves on a log y axis: train loss solid, val loss dashed, mean over each run group."""
    W, H, L, Rm, T, B = 760, 320, 56, 16, 16, 40
    xs = lambda i: L + (W - L - Rm) * i / xmax
    ys = lambda v: T + (H - T - B) * (math.log(ymax) - math.log(min(ymax, max(ymin, v)))) / (math.log(ymax) - math.log(ymin))
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{label}">']
    for v in ticks:
        g.append(f'<line x1="{L}" x2="{W - Rm}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#d1d9e0"/>')
        g.append(f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#59636e">{v:g}</text>')
    for i in range(0, xmax + 1, xmax // 5):
        g.append(f'<text x="{xs(i):.1f}" y="{H - B + 16}" text-anchor="middle" font-size="11" fill="#59636e">{i}</text>')
    g.append(f'<text x="{(L + W - Rm) / 2}" y="{H - 6}" text-anchor="middle" font-size="12">Iteration</text>')
    g.append(f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" font-size="12" '
             f'transform="rotate(-90 14 {(T + H - B) / 2})">Loss (log scale)</text>')
    legend = []
    for name, names, color in lines:
        names = [n for n in names if n in R]
        steps = sorted(set.intersection(*(set(R[n]["ev"]) for n in names)))
        for k, dash in ((0, ""), (1, ' stroke-dasharray="6 4"')):
            pts = " ".join(f"{xs(st):.1f},{ys(np.mean([R[n]['ev'][st][k] for n in names])):.1f}" for st in steps)
            g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{dash} points="{pts}"/>')
        legend.append(f'<span><i style="background:{color}"></i>{name}</span>')
    legend.append('<span>solid = train, dashed = val</span>')
    g.append("</svg>")
    return '<div class="legend-line">' + "".join(legend) + "</div>\n" + "\n".join(g)


def curves_nodrop():
    return curves([("LayerNorm, lr 1e-3 (3 seeds)", LN_SEEDS, "#000"),
                   ("BW, same lr 1e-3, ff 0.9 (1 seed)", ["s1-rm-nowarm"], "#0969da"),
                   ("BW best recipe, lr 1.5e-3 (3 seeds)", BEST_SEEDS, "#1a7f37")],
                  0.05, 5, (0.05, 0.1, 0.2, 0.5, 1, 2, 4), "Learning curves, dropout 0")


def curves_drop10k():
    return curves([("LayerNorm, lr 1e-3 (2 seeds)", D10_LN, "#000"),
                   ("LayerNorm, lr 2e-3 (1 seed)", ["s6-d10k-ln-lr2e-3"], "#afb8c1"),
                   ("BW, momentum 0.3, ff 0.4, lr 2e-3 (2 seeds)", D10_BW, "#1a7f37")],
                  0.15, 5, (0.2, 0.3, 0.5, 1, 1.5, 2, 4), "Learning curves, dropout 0.2, 10k steps", xmax=10000)


def cond_corr(step):
    """Spearman rank correlation, over completed dropout-0 runs, between the
    MLP-input condition number (geometric mean over layers) and mean train loss."""
    names = [n for n, r in R.items() if "-d-" not in n and not n.startswith(("timing", "mps"))
             and 5000 in r["ev"] and step in r["cond"].get("mlp_in", {})]
    c = np.log([R[n]["cond"]["mlp_in"][step] for n in names])
    m = [mean_train(R[n]["ev"]) for n in names]
    rk = lambda a: np.argsort(np.argsort(a))
    return np.corrcoef(rk(c), rk(m))[0, 1], len(names)


def pct(names, base_names, j=0, low=0.1):
    return 100 * (seed_metrics(names, low)[1].mean(0)[j] / seed_metrics(base_names, low)[1].mean(0)[j] - 1)


TLDR = f"""<b>Summary.</b> Causal BW gives a clear, seed-robust gain over LayerNorm on nanoGPT, larger than before the leak fix.
<ul>
  <li><b>No dropout:</b> {-pct(BEST_SEEDS, LN_SEEDS):.0f}% lower mean train loss and {-pct(BEST_SEEDS, LN_SEEDS, 1):.0f}% lower final train loss
  on every seed; loss 0.1 is reached {-pct(BEST_SEEDS, LN_SEEDS, 3):.0f}% sooner. Validation is unchanged. Before this study the causal version gave 8%; the leaky one 13%.</li>
  <li><b>Vanilla nanoGPT (dropout 0.2):</b> {-pct(D_BEST_SEEDS, D_LN_SEEDS, low=0.7):.0f}% lower mean and {-pct(D_BEST_SEEDS, D_LN_SEEDS, 1, 0.7):.0f}% lower final train loss on every seed, validation tied. Over 10k steps the final gain grows to &minus;18% (2 seeds). Before the fix the best BW setting was 2% ahead, with the leak.</li>
  <li><b>Recipe:</b> renorm with shared statistics, <code>running_mean</code>, full width (group 384), no warmup, no gradient boost, BW on every MLP input followed by the LayerNorm, and a higher learning rate (BW 1.5e-3; LayerNorm's best is 1e-3).
  No dropout: momentum 0.7, <code>fix_factor</code> 0.8. Dropout 0.2: more history and shrinkage, momentum 0.3, <code>fix_factor</code> 0.4, lr 2e-3.</li>
  <li><b>Cost:</b> 2&times; per iteration (27.7 against 14.0 ms), so per step BW wins but in wall-clock time it does not yet. BW in only the first 3 blocks costs 1.4&times; but keeps only a third of the gain.</li>
  <li><b>Dropped:</b> <code>calibrated</code> (ties LayerNorm at 2.5&ndash;7&times; the cost), warmup, gradient boost (diverges above 1), BW on the attention input, BW without the following LayerNorm.</li>
</ul>
<p>Commands: <code>--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --batch_center_mode=running_mean --bw_group_size=384 --bw_warmup=none --bw_momentum=0.7 --bw_fix_factor=0.8 --learning_rate=1.5e-3</code>
(dropout 0.2: <code>--bw_momentum=0.3 --bw_fix_factor=0.4 --learning_rate=2e-3</code>). About 2.9 GPU-hours for 61 runs, using CUDA MPS to run 4&ndash;6 per GPU.</p>"""

r250, n250 = cond_corr(250)
r5k, _ = cond_corr(5000)
COND_TEXT = f"""<ul class="summary">
  <li><b>BW conditions the MLP input much better early on:</b> 10&ndash;30&times; better than LayerNorm at step 250 and 4&ndash;14&times; at step 1000 (layers 1&ndash;5). By step 5000 LayerNorm has caught up, and the best recipe, with its higher learning rate, ends <i>worse</i> conditioned than LayerNorm while training best.</li>
  <li><b>Layer 0 stays very ill-conditioned</b> (10<sup>3</sup>&ndash;10<sup>6</sup>) in every model: its input is the token plus position embedding of a 65-symbol vocabulary, and the whitening shrinkage (<code>fix_factor</code>) caps how much BW can lift its smallest directions.</li>
  <li><b>As a predictor it works early and within one knob.</b> Across {n250} runs, the rank correlation between MLP-input condition number and mean train loss is {r250:+.2f} at step 250 and {r5k:+.2f} at step 5000, matching a gain that comes from early training. Within group size and <code>fix_factor</code>, worse conditioning means worse loss (group 128: about 10<sup>5</sup> at step 250, the worst completed BW run). Across modes and learning rates it misleads: <code>leave_one_out</code> conditions best and trains worse than <code>running_mean</code>.</li>
</ul>"""

DROPOUT_TEXT = """<ul class="summary">
  <li><b>Dropout needs more history and more shrinkage.</b> The dropout-0 recipe is 6.5% worse here. Dropout makes each batch's covariance noisy, so the buffer has to average more batches (momentum 0.3 instead of 0.7) and the off-diagonals need stronger shrinkage (<code>fix_factor</code> 0.4 instead of 0.8). Either change alone is not enough.</li>
  <li><b>The higher learning rate still matters</b> (lr 1e-3: +2.7%), and LayerNorm does not gain from it (&minus;1.2%, within noise).</li>
</ul>"""

DROPOUT10K_TEXT = """<ul class="summary">
  <li><b>The longer run doubles the final gain:</b> &minus;18% final train loss on both seeds (0.175 against 0.213), against &minus;10% at 5k. Loss 0.3 is reached about 600 steps sooner.</li>
  <li><b>Most of the gain comes while the learning rate decays.</b> The gap is &minus;1 to &minus;2% for the first 4000 steps, then grows to &minus;17% by step 8000 and holds. The absolute gap peaks around step 7000&ndash;8000 (0.047) and shrinks slightly by 10k (0.038), so an even longer run is unlikely to widen the relative gain much.</li>
  <li><b>It is not the learning rate:</b> LayerNorm at 2e-3 ends level with LayerNorm at 1e-3 (0.213).</li>
  <li><b>Validation:</b> the best validation loss is tied (about 1.46 for all). After that, every model overfits; BW overfits more (final val 2.38 against 2.27).</li>
</ul>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Causal BW on nanoGPT: what works</title>
{STYLE}
<style>.figcap {{ color: var(--muted); font-style: italic; font-size: 12.5px; margin: 6px 0 18px 28px; }}</style>
</head>
<body>
<h1>Causal BW on nanoGPT: what works</h1>
<p class="sub">shakespeare_char, B&nbsp;=&nbsp;64, T&nbsp;=&nbsp;256, 6 layers, C&nbsp;=&nbsp;384, dropout 0, 5000 steps, 1 GPU.
All BW runs use the causal fix (<code>bw_renorm=True</code>, shared statistics). Feature list and pre-fix work:
<a href="bw_paper_vs_code.html">bw_paper_vs_code.html</a>; the fix and GPT-2: <a href="bw_causal_report.html">bw_causal_report.html</a>.</p>

<div class="todo">{TLDR}</div>

<h2>Result: best recipe against LayerNorm (3 seeds each)</h2>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">ms / iter</th><th class="num">Mean train loss</th><th class="num">Final train loss</th>
<th class="num">Steps to loss 1.0</th><th class="num">Steps to loss 0.1</th><th class="num">Best val</th></tr>
{final_table()}
</table>
<p class="cap">Mean over seeds (range in grey). Mean train loss = average eval-mode train loss over steps 250&ndash;5000 (area under the curve). Bold green = better than LayerNorm.</p>
{chart()}
<p class="figcap"><b>Figure 1:</b> Eval-mode train loss relative to LayerNorm (mean of 3 LayerNorm seeds), every 250 steps. Below 0 = better.
{catch_up(LN_SEEDS, BEST_SEEDS, 5000)}</p>
{curves_nodrop()}
<p class="figcap"><b>Figure 2:</b> Learning curves, dropout 0. Eval-mode train loss (solid) and validation loss (dashed), every 250 steps, averaged over the seeds of each configuration.
The blue run keeps LayerNorm's learning rate and differs from the BW defaults only by warmup off.
Without dropout every model overfits after about step 1000 (validation rises); BW fits faster, so its validation loss rises sooner and ends higher (4.46 against 4.26 at step 5000), while the best validation loss is tied.
{catch_up(LN_SEEDS, BEST_SEEDS, 5000)}</p>

<h2>What each knob does</h2>
<table>
<tr><th>Question</th><th>Setting</th><th class="num">Mean train loss vs. LayerNorm</th><th>Conclusion</th></tr>
{study_table()}
</table>
<p class="cap">One seed (1337) per row; seed-to-seed spread is about &plusmn;1%. Each row changes one knob from the row in bold, starting from renorm <code>running_mean</code>, group 384, momentum 0.7, ff 0.9, warmup on, lr 1e-3. Runs stop automatically once they are 50% worse than LayerNorm at the same step.</p>

<h2>Conditioning: does BW help the MLP input, and does it predict the gain?</h2>
<table>
<tr><th rowspan="2">Model</th><th colspan="2" class="num">Step 250</th><th colspan="2" class="num">Step 1000</th><th colspan="2" class="num">Step 5000</th><th rowspan="2" class="num">Mean train loss vs. LayerNorm</th></tr>
<tr><th class="num">layer 0</th><th class="num">layers 1&ndash;5</th><th class="num">layer 0</th><th class="num">layers 1&ndash;5</th><th class="num">layer 0</th><th class="num">layers 1&ndash;5</th></tr>
{cond_table()}
</table>
<p class="cap">Condition number (largest / second-smallest eigenvalue; LayerNorm output always has one null direction) of the MLP input covariance, one training batch, eval mode. Layers 1&ndash;5: geometric mean.</p>
{{COND_TEXT}}

<h2>Vanilla nanoGPT (dropout 0.2 everywhere)</h2>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">ms / iter</th><th class="num">Mean train loss</th><th class="num">Final train loss</th>
<th class="num">Steps to loss 1.0</th><th class="num">Steps to loss 0.7</th><th class="num">Best val</th></tr>
{dropout_table()}
</table>
<table>
<tr><th>Variant (seed 1337, renorm, warmup off, lr 2e-3 unless stated)</th><th class="num">Mean train loss vs. LayerNorm</th></tr>
{dropout_variants()}
</table>
{{DROPOUT_TEXT}}

<h3>Dropout 0.2, 10k steps (cosine schedule stretched to 10k for every run)</h3>
<table>
<tr><th>Configuration</th><th class="num">Seeds</th><th class="num">ms / iter</th><th class="num">Mean train loss</th><th class="num">Final train loss</th>
<th class="num">Steps to loss 0.5</th><th class="num">Steps to loss 0.3</th><th class="num">Best val</th></tr>
{dropout_10k_table()}
</table>
<table>
<tr><th class="num">Step</th><th class="num">LayerNorm train loss</th><th class="num">BW train loss</th><th class="num">BW vs. LayerNorm</th></tr>
{gap_table(D10_LN, D10_BW, range(1000, 10001, 1000))}
</table>
{curves_drop10k()}
<p class="figcap"><b>Figure 3:</b> Learning curves, dropout 0.2, 10k steps. Eval-mode train loss (solid; dropout off during eval) and validation loss (dashed), mean over seeds.
The gap opens in the second half, as the learning rate decays, and settles at about &minus;18%; LayerNorm at lr 2e-3 tracks LayerNorm at 1e-3.
{catch_up(D10_LN, D10_BW, 10000)}</p>
{{DROPOUT10K_TEXT}}
</body>
</html>
"""
html = (html.replace("{COND_TEXT}", COND_TEXT).replace("{DROPOUT_TEXT}", DROPOUT_TEXT)
        .replace("{DROPOUT10K_TEXT}", DROPOUT10K_TEXT))
open(OUT, "w").write(html)
print("wrote", OUT)
