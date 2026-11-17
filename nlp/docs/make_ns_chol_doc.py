"""Build bw_ns_chol.html: the ns_chol whitening (Newton-Schulz forward, Cholesky backward).

Usage: python make_ns_chol_doc.py <repo_root>
Reads <repo_root>/nlp/study/logs/*.out (written by nlp/sbatch_nanogpt_pack.sh).
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nano_study import mean_train, parse, steps_to  # noqa: E402

OUT = os.path.join(HERE, "bw_ns_chol.html")
STYLE = re.search(r"<style>.*?</style>", open(os.path.join(HERE, "bw_paper_vs_code.html")).read(), re.S).group(0)
LOGS = os.path.join(sys.argv[1], "nlp", "study", "logs")

LN = "s2-ln-lr1.5e-3"
# label, runs (seeds 1337, 1, 2), ms / step (nlp/study/logs/timing-*.out)
METHODS = (("Cholesky (previous default)", ["f-ln0-fused", "f-ln0-fused-s1", "f-ln0-fused-s2"], 15.72),
           ("Newton&ndash;Schulz, 10 iterations", ["e-ns10", "e-ns10-s1", "e-ns10-s2"], 21.55),
           ("<b>ns_chol</b>, 10 iterations", ["i-nschol10", "i-nschol10-s1", "i-nschol10-s2"], 15.77))
LN_MS = 13.94

R = {n: parse(os.path.join(LOGS, n + ".out")) for _, ns, _ in METHODS for n in ns + [LN]
     if os.path.exists(os.path.join(LOGS, n + ".out"))}
LN_EV = R[LN]["ev"]
TARGET = LN_EV[5000][0]
LN_TIME = 5000 * LN_MS / 1e3


def pct(v):
    return f"{v:+.1f}%".replace("-", "&minus;")


def stats(names, ms):
    names = [n for n in names if n in R and 5000 in R[n]["ev"]]
    mean = [100 * (mean_train(R[n]["ev"]) / mean_train(LN_EV) - 1) for n in names]
    steps = [steps_to(R[n]["ev"], TARGET) for n in names]
    wall = [100 * (s * ms / 1e3 / LN_TIME - 1) for s in steps]
    return names, mean, steps, wall


def seeds(vals, fmt):
    return '<span class="rng">(' + ", ".join(fmt(v) for v in vals) + ")</span>"


rows, res = [], {}
for label, names, ms in METHODS:
    names, mean, steps, wall = stats(names, ms)
    res[label] = np.mean(steps), np.mean(wall)
    best = label.startswith("<b>")
    td = '<td class="num better">' if best else "<td class=num>"
    rows.append(f"<tr><td>{label}</td><td class=num>{ms:.2f}</td>"
                f"<td class=num>{pct(np.mean(mean))} {seeds(mean, pct)}</td>"
                f"<td class=num>{np.mean(steps):.0f} {seeds(steps, lambda v: f'{v:.0f}')}</td>"
                f"{td}{np.mean(steps) * ms / 1e3:.1f} s ({pct(np.mean(wall))})</td>"
                f"<td class=num>{len(names)}</td></tr>")
chol_steps, chol_wall = res[METHODS[0][0]]
ns_steps, _ = res[METHODS[1][0]]
nc_steps, nc_wall = res[METHODS[2][0]]

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ns_chol: Newton&ndash;Schulz forward, Cholesky backward</title>
{STYLE}
</head>
<body>
<h1>ns_chol: Newton&ndash;Schulz forward, Cholesky backward</h1>
<p class="sub">Causal BW (renorm, shared statistics) on nanoGPT shakespeare_char, the best recipe of
<a href="bw_nanogpt_study.html">bw_nanogpt_study.html</a>: B&nbsp;=&nbsp;64, T&nbsp;=&nbsp;256, 6 layers, C&nbsp;=&nbsp;384,
dropout 0, lr 1.5e-3, 5000 steps, one H100.</p>

<div class="todo"><b>Summary.</b>
<ul>
  <li>BW can whiten with <b>Cholesky</b> (cheap) or with <b>Newton&ndash;Schulz</b> (symmetric, ZCA whitening: lower loss, but 21.6 instead of 15.7 ms / step).
  <b>ns_chol uses each where it is cheap</b>, and gets Newton&ndash;Schulz's loss at Cholesky's step time.</li>
  <li><b>The gain is in steps, not step time:</b> an ns_chol step costs the same as a Cholesky step (15.77 against 15.72 ms; both slower than LayerNorm's {LN_MS:.2f}),
  but reaches a given loss in fewer steps ({nc_steps:.0f} against {chol_steps:.0f} to LayerNorm's final loss).</li>
  <li>With the same hyperparameters for both, BW with ns_chol reaches LayerNorm's final train loss in {nc_steps:.0f} instead of 5000 steps:
  <b>{pct(nc_wall)} wall-clock time</b> against LayerNorm, compared with {pct(chol_wall)} for Cholesky (mean of 3 seeds).</li>
  <li>ns_chol is now the default (<code>bw_whiten="ns_chol"</code>, <code>bw_ns_iters=10</code> in <code>train.py</code>);
  <code>--bw_whiten=cholesky</code> or <code>--bw_whiten=ns</code> switch back. It needs renorm with batch statistics; without them BW uses Cholesky.</li>
</ul></div>

<h2>How it works</h2>
<p><b>The layer.</b> Each MLP input is normalized by one BW layer that now also does the needed part of LayerNorm. Before, it was BW followed by
a separate LayerNorm with its own gain and bias. Now BW is followed by a LayerNorm without gain or bias, applied inside the same fused BW
function (native LayerNorm kernels, forward and backward), with no separate LayerNorm module. The per-token normalization is what is needed after BW;
its gain and bias are redundant with BW's own &gamma; and &beta;, and dropping them does not change the loss
(<a href="bw_nanogpt_study.html">bw_nanogpt_study.html</a>, &ldquo;Do we need the LayerNorm after BW?&rdquo;). All BW runs below use this layer.</p>
<ul class="summary">
  <li><b>Forward: Newton&ndash;Schulz.</b> The whitening matrix applied to the activations is the symmetric one,
  A<sup>&minus;1/2</sup> of the running covariance, from 10 Newton&ndash;Schulz iterations. With renorm the forward only needs the matrix
  from the previous step, so Newton&ndash;Schulz is never computed inside the forward pass: it computes next step's matrix,
  in parallel with the backward pass (below), and costs almost no time.</li>
  <li><b>Backward: Cholesky.</b> The gradient is the closed-form Cholesky backward, with this batch's Cholesky factor L.
  This avoids backpropagating through the 10 Newton&ndash;Schulz iterations, which is what makes plain Newton&ndash;Schulz slow.</li>
  <li><b>Why the two fit together:</b> a symmetric whitening matrix and the Cholesky one differ only by a rotation,
  A<sup>&minus;1/2</sup> = Q L<sup>&minus;1</sup> with Q orthogonal. With Q held constant, the gradient through the symmetric matrix
  is exactly the Cholesky gradient. The only approximation is that the gradient ignores how Q changes with the covariance.</li>
</ul>

<h3>What runs when (one BW layer, training step t)</h3>
<ol>
  <li><b>Forward, main stream:</b> Y = &gamma; W<sub>R</sub>(x &minus; &mu;<sub>R</sub>) + &beta;, one matrix product. W<sub>R</sub> was computed during step t&minus;1;
  no covariance, Cholesky or Newton&ndash;Schulz here, so the next layer starts right away.</li>
  <li><b>Side CUDA stream, overlapping the later layers' forward:</b> this batch's statistics (X<sup>T</sup>X and the mean),
  the running-covariance update and its Cholesky factor. Only the backward and the next step need them.</li>
  <li><b>Backward:</b> the main stream computes the gradient with the Cholesky formula. At the same time the side stream runs the
  10 Newton&ndash;Schulz iterations on the updated covariance, producing next step's W<sub>R</sub>, overlapped with the backward of this layer and the layers below.</li>
  <li><b>Step t+1:</b> the forward uses that W<sub>R</sub>, long finished by then.</li>
</ol>
<p class="cap">"In parallel" means concurrently on the GPU, on a second CUDA stream, not across layers: the forward is still layer after layer,
but each layer's extra work is not needed by anything later in the same forward. The placement matters: running Newton&ndash;Schulz on
the side stream during the forward made the side stream fall behind, so the backward waited for the last layers' Cholesky factors (16.98 ms / step instead of 15.77).</p>

<h2>Result: against LayerNorm, same hyperparameters</h2>
<table>
<tr><th>Whitening</th><th class="num">ms / step</th><th class="num">Mean train loss vs LayerNorm</th>
<th class="num">Steps to LayerNorm's final loss ({TARGET:.4f})</th><th class="num">Time to that loss</th><th class="num">Seeds</th></tr>
<tr class="baseline"><td>LayerNorm</td><td class=num>{LN_MS:.2f}</td><td class=num>0%</td><td class=num>5000</td><td class=num>{LN_TIME:.1f} s</td><td class=num>1</td></tr>
{chr(10).join(rows)}
</table>
<p class="cap">Mean over seeds, per-seed values in parentheses. Time = steps &times; ms / step, training steps only;
step times are synchronized 100-step blocks (<code>nlp/nano_gpt/profile_step.py</code>). Everything except the whitening is identical.</p>
<ul class="summary">
  <li><b>Loss:</b> ns_chol matches Newton&ndash;Schulz ({nc_steps:.0f} against {ns_steps:.0f} steps) and beats Cholesky ({chol_steps:.0f} steps).</li>
  <li><b>Step time:</b> ns_chol costs the same as Cholesky (15.77 against 15.72 ms).</li>
</ul>

<h2>Tried, did not help</h2>
<ul class="summary">
  <li><b>Fewer iterations:</b> 7 Newton&ndash;Schulz iterations lose most of the gain, 3 or 5 diverge. 10 are needed.</li>
  <li><b>BW in layers 0, 2, 4 only</b> (14.8 ms / step): fewer BW layers lose quality, so about the same wall-clock time as all 6 layers.</li>
  <li><b>Newton&ndash;Schulz replayed from a CUDA graph:</b> same values, but much slower (26&ndash;38 ms / step) next to the side stream.</li>
</ul>

<p class="foot">Code: <code>bw_whiten="ns_chol"</code> in <code>renorm_fast_whiten</code>, <code>nlp/nano_gpt/model.py</code>.
Check: <code>nlp/nano_gpt/check_ns_chol.py</code> (given the same previous matrix, output and gradients are identical to Cholesky;
the deferred side-stream path matches the eager one exactly). Runs: <code>nlp/study/pack27.txt</code>&ndash;<code>pack30.txt</code>.
Generated by <code>nlp/docs/make_ns_chol_doc.py</code>.</p>
</body>
</html>
"""

open(OUT, "w").write(HTML)
print("wrote", OUT)
