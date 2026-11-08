# Batch Whitening (BW) in nanoGPT / GPT-2 — session handoff

Knowledge from a long agent session (Jun 29 – Nov 5, 2026) with hsivan. It is meant for a new agent picking up the work.
Read this first, then the two HTML reports:

- `nlp/docs/bw_paper_vs_code.html` (generator `make_bw_report.py`): the full feature list, the paper-vs-code audit for each mode, and all **pre-fix** nanoGPT experiments (dropout 0, dropout 0.2, gradient-boost sweep). It also has a red box at the top explaining the causality leak and the nanoGPT renorm results.
- `nlp/docs/bw_causal_report.html` (generator `make_causal_report.py`): a short report on everything **after the leak fix**, covering the fix, what was re-tested, nanoGPT, the conditioning diagnostic, all GPT-2 trials and the conclusions.
- `nlp/nano_gpt/batch_whitening_notes.md`: the algorithm reference for each mode (centering, covariance, warmup, calibration, boost, dropout interaction, verifier results).

---

## 1. Status in one paragraph

Training-mode BW leaked future tokens across stacked layers. The fix is `bw_renorm=True`: Batch Renormalization applied to whitening. The leak is now guarded by the unit test and by a startup check in `train.py`.

On **nanoGPT** (shakespeare_char, dropout 0), causal BW beats LayerNorm by about 8% in mean train loss. That is about half of the old, leak-inflated margin. With `bw_renorm_stats=batch` the BW overhead is half of what it was.

On **GPT-2 124M** (1 GPU), no setting beats LayerNorm over a longer run. The best one is ahead by 2–3% early, by 0.5% at step 2000, and is level at step 4000 (train +0.2%, val tie). That setting is renorm, shared statistics, `running_mean`, group 768, momentum 0.3, `fix_factor` 0.4, warmup off, BW in layers 0–5 only. The user's target is **train loss** (not val). No jobs are running. A lot of work is **uncommitted** (section 9).

---

## 2. Environment and cluster: practical knowledge

- **Repo:** `/home/hsivan/batch_whitening`. The remote is GitHub `guyknvda/batch_whitening`. Code lives in `nlp/nano_gpt/` (`model.py`, `train.py`); reports are in `nlp/docs/`; the paper is in `nlp/bw_paper/` (LaTeX; compiled artifacts are git-ignored except the PDF).
- **Python:** the system `python3` has no numpy. Use:
  `XDG_CACHE_HOME=~/.local/cache conda run -n env_python39 python script.py`.
  `conda run` with heredoc stdin prints nothing, so write the script to a file (scratch lives in `~/.local/cache/`) and set `PYTHONPATH=$PWD` if it imports `model`.
- **Slurm:**
  - Account `network_research_swarch`.
  - The cluster is CoreWeave DFW with H100 nodes. Jobs are isolated per GPU and nodes are shared, so 1-GPU jobs are fine.
  - Fair-share pending can be long; check it with `sshare -A <account> -u hsivan`.
  - **Fast start trick:** `--partition=batch,batch_short --time=2:00:00` (`batch_short` has a 2 h limit). Pending jobs can be moved with `scontrol update jobid=<id> partition=batch,batch_short,backfill`.
  - The default `sbatch_*` time is 4 h, and jobs are killed at 4 h.
- **Lustre access:**
  - `~/.cache` is `/lustre/fsw/portfolios/network/projects/network_research_swarch/users/hsivan`. It holds OpenWebText (`train.bin` 18 GB, `val.bin`) and GPT-2 checkpoints.
  - It needs Unix group `network-research`. The user lost access once and regained it on Nov 1 (Access-Network-Research DL).
  - **Submit GPT-2 jobs with** `sg network-research -c "sbatch ..."` so the job has the group.
- **Home filesystem:** 10 GB, about 2.3 GB free (78% used). nanoGPT checkpoints go to node-local `/tmp` (the default out_dir in `sbatch_nanogpt_exp.sh`). A full home crashed a checkpoint write once.
- **Policy:** sudo and ssh are blocked. The user said: **avoid delete operations** unless there is a disk issue, and don't kill GPT-2 jobs because of val loss.
- **wandb:**
  - nanoGPT project: `shakespeare-char` (`https://forge.coreweave.com/wandb/nv-welcome/shakespeare-char`).
  - GPT-2 project: `owt`.
  - Baselines: nanoGPT dropout 0 is `l965qcs2` (plus `exp2-base-s1/-s2`); nanoGPT dropout 0.2 is `a9554bt7`; GPT-2 4-GPU LayerNorm is `d729qrk3`.
- **Job logs:** Slurm writes `<jobname>_<jobid>.out` in the repo root (about 244 of them, untracked). The GPT-2 1-GPU LayerNorm reference log is **`g2-ln_21146146.out`** (it runs to about 7,000+ iterations; eval every 1000).

### How to run

```bash
# nanoGPT (1 GPU, ~6 min BW renorm, 5000 iters); args are train.py --key=value overrides
sbatch -J <name> nlp/sbatch_nanogpt_exp.sh --wandb_run_name=<name> --normalization_type=full_bw \
  --batch_center_mode=running_mean --bw_group_size=384 --bw_momentum=0.7 --bw_renorm=True --bw_renorm_stats=batch --seed=...

# GPT-2 124M, 1 GPU, 10 micro-batches x 12 x 1024 per optimizer step (out_dir on lustre)
sg network-research -c "sbatch --partition=batch,batch_short --time=2:00:00 -J <name> nlp/sbatch_gpt2_1gpu.sh \
  --wandb_run_name=<name> --normalization_type=full_bw --batch_center_mode=running_mean --bw_group_size=768 \
  --bw_renorm=True --bw_renorm_stats=batch --bw_warmup=none --bw_fix_factor=0.4 --bw_momentum=0.3 --bw_layers=6 --max_iters=2000"

# LayerNorm reference: --normalization_type=layer_norm
```

- `train.py --config_name=nanogpt|gpt2` selects the config. `nanogpt` is the default: B=64, T=256, 6 layers, C=384, dropout 0 everywhere.
- GPT-2 LR schedule: warmup 2000, decay to 600000. `--max_iters` only stops the run early; the schedule is unchanged, so short runs are comparable to the LayerNorm log at the same iteration.
- All sbatch files live in `nlp/` and are submitted from the repo root (they `cd` to the submit directory): `sbatch_nanogpt.sh` (plain), `sbatch_gpt2.sh` (multi-GPU), `sbatch_nanogpt_pack.sh` (several runs per GPU), `sbatch_diag_*.sh` (diagnostics).

### Comparing a GPT-2 run with LayerNorm (helper `~/.local/cache/g2_cmp2.py`)

```python
import re, sys
def load(f, pat):
    return {int(m[0]): float(m[1]) for m in re.findall(pat, open(f).read(), re.M)}
it, ev = r"^iter (\d+): loss ([\d.]+)", r"^step (\d+): train loss ([\d.]+)"
ln_f = "g2-ln_21146146.out"
ln, lne = load(ln_f, it), load(ln_f, ev)
for rn_f in sys.argv[1:]:
    rn, rne = load(rn_f, it), load(rn_f, ev)
    out = [f"eval {s}: {100*(r/lne[s]-1):+.1f}%" for s, r in sorted(rne.items()) if s and s in lne]
    last = max(rn); w = [i for i in range(max(0, last - 190), last + 1) if i in rn and i in ln]
    l = sum(ln[i] for i in w) / len(w); r = sum(rn[i] for i in w) / len(w)
    print(f"{rn_f}: iter {last}, last-200 window {100*(r/l-1):+.1f}% " + " ".join(out))
```

Run it from the repo root with the log files as arguments.

---

## 3. The model and the code (what BW is here)

- **Placement:** `Block` does `x + mlp(ln_2(bw_2(x)))`. BW sits before `ln_2` on the MLP input only; attention has LayerNorm only. γ and β are applied right after whitening (before LayerNorm).
- **`normalization_type`:** `layer_norm`, `full_bw`, or `center_only`.
- **`batch_center_mode`** (training statistics):
  - `leave_one_out` (LOO): each sequence is whitened with the mean and covariance of the other B−1 sequences in the current batch, with full gradient. Inference uses the last training batch's buffers.
  - `running_mean` (RM): the centering mean is the LOO mean, with full gradient. The covariance is `(1-m)·running_cov (detached) + m·current_cov`, so `bw_momentum` m is the weight of the current batch. The buffers are EMAs. No gradient boost by default.
  - `calibrated`: 10 extra no-grad calibration forwards per step give pooled statistics at the current weights. It now uses LOO centering (`bw_calib_center=loo`, default), and its gradient boost defaults to 1/m. It costs about 7× run time.
- **Knobs** (`GPTConfig` and `train.py` globals):

  | Knob | Meaning |
  |---|---|
  | `bw_group_size` | Block-diagonal whitening, channels per group (0 means automatic, which gives 16) |
  | `bw_fix_factor` | Off-diagonal shrink factor; 1−ε in the paper; 0 means per-channel standardization only |
  | `bw_momentum` | Weight of the current batch in the EMA |
  | `bw_warmup` | `diag`, `diag_ema` or `none`: the first 250 steps use diagonal-dominant whitening |
  | `bw_inference_ema` | LOO only |
  | `bw_grad_scale` | Gradient boost on the BW statistics; 0 means the mode's default |
  | `bw_calib_center` | Calibrated mode's centering |
  | `bw_renorm`, `bw_renorm_stats` (`loo` or `batch`) | The causal fix |
  | `bw_layers` | BW in the first N blocks only; the rest use `nn.Identity` before `ln_2` (LayerNorm only) |
  | `allow_causal_leak` | Skips the startup refusal |

- **Whitening:** Cholesky `L^{-1}(x-μ)` (not Σ^{-1/2}), plus diagonal loading of 1e-8.
- **`boost_grad(x, s)`:** identity in the forward pass, gradient multiplied by s.

---

## 4. Chronology of findings (pre-fix era, nanoGPT)

These are older, but they explain why the code looks the way it does.

1. **Verifier:** `verify_ln2_bw_orthonorm.py` checks that `ln_2(bw_2(x))` is decorrelated. Grouped decorrelation survives the LayerNorm in train mode. LayerNorm alone gives vector normalization, not orthonormal features.
2. **Dropout breaks BW at inference:** with dropout on the residual path, the stored statistics don't match eval activations (`batch_whitening_notes.md`, "Dropout Breaks BW Whitening At Inference"). This is why nanoGPT experiments use dropout 0.
3. **Centering saga:**
   - Several commits changed how the covariance is centered.
   - The mathematically "unbiased" centering of the covariance about `current_mean` (adds a δδᵀ term) **made `running_mean` worse**. The reason: the δδᵀ term accumulates in the EMA buffer, inflating Σ by about 22%, so the stack under-whitens by about 10% per layer, compounding to about 3.6×.
   - It was reverted for RM. The lesson: the EMA buffer must target the true centered Σ.
4. **Root cause of the bad RM runs:**
   - A **detached centering mean breaks backward scale invariance.** Under x→c·x it leaves an offset LayerNorm can't remove, the residual stream grows, and the activations run away (RMS 27–80×).
   - The fix: the RM mean is the LOO current-batch mean with gradient. A detached covariance is harmless because `ln_2` cancels the rescale. This is commit `cdcba46`.
5. **Why the boost helps `calibrated` but hurts RM:**
   - The boosted gradient assumes the whitening statistics move one-for-one with the batch.
   - That is true for `calibrated`, whose history is recomputed at the current weights.
   - It is false for RM, whose EMA history lags 10–30 steps, so the boost overstates the sensitivity by up to 10× and the model overshoots.
   - So no boost for RM.
6. **`calibrated` with the blended centering mean (`blend`) runs away** (input RMS 150–21,000) at every boost. The default is now `loo`. Calibrated with LOO plus boost 10 gives 0.605 at group 16, which only ties plain LOO at about 7× the cost.
7. **Paper-vs-code audit** (main report): code group 16 against the paper's rule of 48; fixed momentum 0.1; ε=0.1; Cholesky; placement differs. "Not in the paper": calibration, boost, warmup.
8. **Experiment results at dropout 0** (pre-fix, eval-mode losses, leak-affected in training):
   - LayerNorm baseline mean train loss 0.492.
   - Best: LOO, group 384, ff 0.8: **0.428 (−13%)**.
   - RM, group 384, momentum 0.7: 0.435.
   - **Group size is the biggest lever** (16→384). At large groups, RM improves as momentum→1. ε 0.1–0.2 is best; ε→0 is unstable at 384.
9. **Dropout 0.2:** the dropout-0 winner is about 7% worse. Only strong shrinkage (ff 0.2–0.4) ties the baseline. With **boost 0.5** (damped), LOO, group 384, ff 0.4 slightly beats the dropout baseline on all seeds (0.981 vs 1.001). This is not yet re-tested after the fix.
10. **Gradient-boost sweep at dropout 0:** boost 1 is best. Above 1 hurts, and group-384 configurations diverge at boost 2.

---

## 5. The causality leak (most important finding)

- **Mechanism (two hops):**
  - Layer k, sequence s: its LOO statistics include sequence b at all positions, so s's activations carry b's future.
  - Layer k+1, sequence b: its statistics include s's layer-k output, so b's position t depends on b's tokens after t.
  - A single-layer test can't see it. It affects every mode with a current-batch statistic in the forward pass (LOO, RM, calibrated's current term), and warmup too. Each sequence weighs 1/(B−1), so it is much worse on GPT-2 (B=12) than on nanoGPT (B=64). Inference (stored statistics) is causal.
- **Evidence:**
  - Stacked layers in float64: 1 layer ~1e-15, 2 layers 1.6e-2, 3 layers 3.5e-2 to 5.8e-2.
  - Trained nanoGPT, `diag_leak.py` (replace a sequence's 2nd half, measure 1st-half loss): LayerNorm 0, best LOO +0.1%, RM g384 m0.7 **+4.5%**.
  - GPT-2 RM at step 2000: training-forward loss **1.60** against eval-mode loss **9.0** on the same data (LayerNorm 4.32 vs 4.10). LOO at step 1000: 3.66 vs 7.0. Both were cancelled.
- **nanoGPT report numbers were eval-mode**, so they are not inflated, but the training may have exploited the leak.
- **Three fixes considered:**
  1. **Batch Renormalization applied to whitening (implemented, `bw_renorm`).** Forward = stored-buffer whitening `Y_R = L_R^{-1}(x-μ_R)`, exactly inference. Gradient through `Z = R·L_B^{-1}(x-μ_B) + d`, with `R = L_R^{-1}L_B` and `d = L_R^{-1}(μ_B-μ_R)` constant. Implemented as `Y = Y_R + (Z - Z.detach())`, so the forward value is bit-for-bit `Y_R`.
  2. **Causal per-position statistics:** positions ≤ t only. Early positions are starved (B samples at t=0) and need a whitening matrix per position; a chunked version is cheaper.
  3. **Reference sub-batch:** a quarter of the batch is whitened with stored statistics, and the rest with statistics from that quarter, with gradient. It is one-way, so causal.

  Options 2 and 3 are not built. They would show how much of the lost margin is the leak and how much is the cost of one-step-stale statistics.
- **`bw_renorm_stats=batch`:** since the batch statistics only shape the gradient, they don't need to be per-sequence LOO. One shared mean and covariance per batch, applied by matrix products (`_renorm_batch_whiten` in `model.py`). It is 2× faster on nanoGPT (55 → 27.5 ms/iter; LayerNorm 14 ms) and 2.4× faster on GPT-2 (3.1 → 1.3 s/iter) with the same results.
- **Guards (keep them!):**
  - `nlp/nano_gpt/test_batch_orthonorm.py` runs a stacked 1–3 layer forward-perturbation test in float64 with random mixers. It fails if a causal mode leaks **or if the known leak stops being detected**. Currently everything passes.
  - `train.py` calls `GPT.causal_leak(batch)` at startup (it perturbs `idx[0,H:]` and measures the change in sequence 0's first-half logits in training mode, restoring the buffers). It prints the result, logs `causal_leak` to the wandb summary, and **raises if > 1e-6** unless `--allow_causal_leak=True`. Non-renorm BW modes therefore refuse to train by default.

---

## 6. Post-fix nanoGPT results (dropout 0, 3 seeds, renorm)

| Configuration (group 384) | ms/iter | Mean train loss | Final | Best val |
|---|---|---|---|---|
| LayerNorm baseline | 14 | 0.492 | 0.0832 | 1.545 |
| LOO ff0.8, per-sequence stats (`rn1-loo-gs384-ff0.8`) | 55 | 0.454 | 0.0871 | 1.551 |
| LOO ff0.8, shared stats (`rn2-loo-…`) | 27.5 | 0.456 | 0.0866 | 1.557 |
| RM m0.7, per-sequence (`rn1-rm-gs384-mom0.7`) | 57 | 0.457 | 0.0849 | 1.536 |
| **RM m0.7, shared (`rn2-rm-…`)** | 27.8 | **0.451** | 0.0835 | **1.533** |
| (leaky LOO, pre-fix) | | 0.428 | 0.0804 | 1.583 |

The gain is early: loss 1.0 is reached 10–13% sooner, and the final loss ties or is slightly worse. Validation is not hurt. Compare JSONs (from `compare_runs.py`) are in `~/.local/cache/compare/compare_{rn1-,rn2-,exp2-}.json` and are needed by both report generators.

---

## 7. GPT-2 124M experiments after the fix (1 GPU, B=12 × 10 micro-batches, T=1024)

### Key insight: BW buffers update on every micro-batch

One optimizer step is 10 micro-batches of 12 sequences, so the effective history is:
- **momentum 0.7:** about 18 sequences, which is very noisy.
- **momentum 0.1:** about 120 sequences, about one optimizer step old. This matches nanoGPT's staleness at 0.7.

### No-training diagnostic (`diag_whitening_quality.py`, `sbatch_diag_quality.sh`, output `diag-quality_21219285.out`)

It measures the condition number of the BW input covariance on the LayerNorm checkpoints (geometric mean over layers):

| Model | Standardization only | Estimated, m0.7 | m0.3 | m0.1 | Exact covariance |
|---|---|---|---|---|---|
| nanoGPT g384 ff0.8 | 2,939 | 87 | 77 | 77 | 73 |
| GPT-2 g768 ff0.9 | 4,524 | 137 | 87 | 59 | 42 |

GPT-2 layers 1–4 are the worst conditioned (7k–130k); later layers are 340–2,400. The diagnostic showed that low momentum and early layers matter, and that turned out right.

### Runs

All are renorm, `running_mean`, group 768. Train loss is relative to LayerNorm (`g2-ln_21146146.out`); negative is better.

| Run (log) | s/iter | 200–800 | 800–1400 | 1400–2000 | 2000–4000 | Step 2000 train/val | Step 4000 |
|---|---|---|---|---|---|---|---|
| LayerNorm | 0.85 | 0 | 0 | 0 | 0 | — | — |
| m0.7, per-sequence, warmup, ff0.9 (`g2-rn-rm-gs768-mom0.7_21175878`) | 3.03 | +0.8 | +2.3 | +1.7 | +2.3 | +1.6/+1.6 | +2.8/+2.7 |
| m0.1, shared, warmup, ff0.9 (`g2-rnb-rm-gs768-mom0.1_21226208`) | 1.31 | −2.9 | +3.9 | +3.0 | | +4.2/+4.2 | |
| m0.1 + boost 3 (`…-gb3_21226209`) | 1.30 | −3.0 | +17.5 | | | stopped at 1410 | |
| m0.1, no warmup, ff0.9 (`g2-rnb-mom0.1-nowarm-ff0.9_21227260`) | 1.30 | +1.5 | +5.1 | +6.9 | | +6.1/+6.1 | |
| m0.1, no warmup, ff0.4 (`…-ff0.4_21227261`) | 1.29 | −2.5 | −0.4 | +1.7 | | +1.1/+1.1 | |
| m0.1, no warmup, ff0.1 (`…-ff0.1_21227262`) | 1.29 | −2.9 | +1.3 | +0.4 | | +0.7/+0.6 | |
| m0.3, no warmup, ff0.4 (`g2-rnb-mom0.3-nowarm-ff0.4_21228219`) | 1.29 | −2.7 | −2.0 | −0.5 | | +0.3/+0.3 | |
| m0.1, ff0.4, BW layers 0–5 (`…-ff0.4-l6_21228220`) | 1.06 | −2.1 | −1.7 | −0.7 | | −0.4/−0.4 | |
| m0.3, ff0.4, BW layers 0–3 (`g2-rnb-mom0.3-nowarm-ff0.4-l4_21229480`) | 0.99 | −1.4 | −0.8 | −0.8 | | −0.4/−0.5 | |
| **m0.3, ff0.4, BW layers 0–5, to 4000** (`…-l6_21229479`) | 1.06 | −2.0 | −1.7 | −1.0 | −0.3 | −0.5/−0.6 | **+0.2/+0.0** |

The best run by eval step, relative to LayerNorm: 1000 −2.4%, 2000 −0.5%, 3000 −0.1%, 4000 +0.2% (train 3.6759 vs 3.6700; val 3.6947 vs 3.6945). Its BW `input_rms` keeps growing: 1.0 at step 2000, 2.2 at step 4000.

### GPT-2 conclusions

- **What mattered:**
  - Momentum matched to gradient accumulation (0.1–0.3).
  - Mild decorrelation (`fix_factor` 0.1–0.4; 0.9 hurts after iteration 800).
  - Warmup off.
  - BW only in the first 4–6 layers, which is also cheaper.
- **What did not help:** full decorrelation, gradient boost (3 diverges), BW warmup (the handoff at step 1000 causes a jump).
- **The fade after about 800–1000 iterations** happens with or without warmup. It coincides with the LR ramp reaching its peak (2,000 steps) and stale statistics, so the warmup handoff is not the main cause.
- **The gain is faster early convergence.** LayerNorm catches up by step about 3,000–4,000.
- **Pre-fix GPT-2 was 3.2× slower** (2.8–3.1 s/iter, per-sequence LOO Cholesky). Shared statistics plus fewer BW layers bring it to 1.25× LayerNorm.

### Suggested next steps (each about 1 GPU-hour), from the report

1. Update the BW buffers once per optimizer step from all micro-batches (exact 120-sequence statistics, no within-step staleness), instead of on every micro-batch. This needs a code change: accumulate the statistics across micro-batches and update when `micro_step == last`.
2. Use BW in the first layers early, then switch to LayerNorm or anneal BW toward identity, since the gain is early convergence.
3. Use the 4-GPU setup (4× larger batch per step; baseline `d729qrk3`).
4. Other ideas not yet tried:
   - fix options 2 and 3, to measure the stale-statistics cost
   - re-test dropout 0.2 plus boost 0.5 with renorm on nanoGPT
   - per-layer `fix_factor`
   - finding out why `input_rms` keeps growing

---

## 8. File inventory (what I created or changed)

**`nlp/nano_gpt/model.py`** (modified):
- `batch_orthonorm(..., renorm, renorm_stats)` and the LOO renorm path after `solve_triangular`.
- `_renorm_batch_whiten` (shared statistics).
- `BatchWhiteningBlock` and `GPTConfig` fields `bw_renorm`, `bw_renorm_stats`, `bw_layers`.
- `Block(config, layer_idx)` for `bw_layers`.
- `GPT.causal_leak(idx)`.
- Earlier: `bw_grad_scale`, `bw_calib_center`, `bw_warmup`, `bw_inference_ema`, input_rms/output_scale/resid_mean_sd/running_mean_err monitoring printed at each eval (`step N: bw input_rms ...`).

**`nlp/nano_gpt/train.py`** (modified): the new globals, the startup causality check, and `wandb.summary["causal_leak"]`.

**`nlp/nano_gpt/test_batch_orthonorm.py`** (modified): `stacked_leak(...)`, the stacked causality tests.

**Diagnostics** (new, untracked):
- `diag_leak.py` (trained-model future-swap test)
- `diag_rootcause.py` and `diag_staleness.py` (RM runaway and staleness analysis)
- `diag_warmup_handoff.py`
- `diag_whitening_quality.py` (condition-number diagnostic: `<ckpt> <data.bin> <batch_size> [num_seqs]`)

**Reports:**
- `nlp/docs/bw_paper_vs_code.html`. Regenerate with `make_bw_report.py`; modes: default, `--dropout`, `--boost`, `--renorm` (for example `python make_bw_report.py --renorm <rn1>.json <exp2>.json <exp3>.json`). Sections are replaced between `<!--RENORM-->` markers and similar.
- `nlp/docs/bw_causal_report.html`. Regenerate with:
  ```bash
  cd /home/hsivan/batch_whitening; C=~/.local/cache/compare
  XDG_CACHE_HOME=~/.local/cache conda run -n env_python39 python nlp/docs/make_causal_report.py $PWD $C/compare_rn1-.json $C/compare_rn2-.json $C/compare_exp2-.json
  ```
  It reads the GPT-2 numbers straight from the Slurm logs listed in `G_RUNS`; add a tuple there for a new run. Texts live in the `TEXT` dict.

**sbatch files** (`nlp/`): `sbatch_nanogpt.sh`, `sbatch_nanogpt_exp.sh`, `sbatch_nanogpt_pack.sh`, `sbatch_gpt2.sh`, `sbatch_gpt2_1gpu.sh`, `sbatch_diag_*.sh`. Old, untracked copies of `sbatch.sh`, `sbatch_nanogpt.sh` and `sbatch_gpt2.sh` (differing only in `--time`) remain in the repo root.

**Scratch** (outside the repo): `~/.local/cache/` holds `g2_cmp.py`, `g2_cmp2.py`, `cmp_rn.py`, `check_batch.py`, `check_layers.py`, `smoke_leak.py`, the `compare/` JSONs, and `g2mon*.log`.

---

## 9. Git state and open housekeeping

- Last commits: `ca714ea` (BW knobs, boost sweep, report, paper draft) and `cdcba46` (RM scale-runaway fix). **The user still needs to `git push origin main`** for these, unless already done.
- Commits stay within `nlp/` (user's rule): anything worth committing that lives outside is moved into `nlp/` first. Commit only when the user asks.
- Untracked data copies (`data/`, `nlp/nano_gpt/data/shakespeare_char/*`) and two `.patch` files in the root belong to the user; leave them.
- Home cleanup: the user approved deleting only old Cursor server versions and the conda package cache, which is done. **Do not delete** wandb dirs, old logs, old checkpoints, or `out-rn1-loo` without asking.

---

## 10b. Post-fix nanoGPT study (Nov 8): report `nlp/docs/bw_nanogpt_study.html`

- **Generator:** `make_nano_study_report.py <repo_root>`. It reads `nlp/study/logs/*.out`; `nano_study.py <log_dir> <baseline> [regex]` prints a text summary.
- **Runs:** packed with `sbatch -J <name> nlp/sbatch_nanogpt_pack.sh nlp/study/packN.txt` from the repo root (job file lines are `name overrides...`; `MAX_PAR` env, default 4). It starts **CUDA MPS** (4 runs: 74 ms/iter each, against 27.5 ms alone, about 1.5× less GPU time; without MPS, packing gives no gain).
- **Totals:** 61 runs, about 2.9 GPU-hours.
- **New knobs:**
  - `bw_where` (block list, e.g. `"0,1"`)
  - `bw_sites` (`mlp`, `attn` or `both`; `bw_1` is now wired before `ln_1`)
  - `bw_ln_after`
  - `calibrated` now works with `bw_renorm_stats=batch`
  - `train.py`: `log_conditioning` (condition numbers of the MLP and attention inputs and the residual stream, every eval), `abort_ref_log`/`abort_ratio` (self-stop when train loss exceeds ratio × a reference log at the same step, or is NaN)
  - string overrides are no longer `literal_eval`'d
- **Best, no dropout:** renorm batch, `running_mean`, momentum 0.7, group 384, ff 0.8, `bw_warmup=none`, no boost, all 6 MLP inputs (BW → LayerNorm), **lr 1.5e-3** (LayerNorm's best is 1e-3).
  - Over 3 seeds: mean train loss −15%, final −7%, val tie.
  - Cost 27.7 vs 14.0 ms/iter.
- **Best, dropout 0.2 (vanilla):** momentum 0.3, ff 0.4, lr 2e-3. Over 3 seeds: mean −4%, final −10%, val tie.
- **Dead ends:**
  - `calibrated` (ties at 2.5–7× cost)
  - boost > 1 (diverges in every mode) and boost < 1 (worse)
  - BW on the attention input
  - dropping the LayerNorm after BW
  - group < 384
  - lr ≥ 2.5e-3
- **Conditioning:** BW is 10–30× better conditioned at step 250 and 4–14× at step 1000; by the end LayerNorm catches up. Rank correlation with mean loss is +0.54 at step 250 and about 0.1 at the end, so it only predicts early.
- **Next ideas:**
  - cut the BW cost: the next step's `L_R` equals this step's `L_B` in `running_mean`, so it could be cached
  - a learning-rate schedule tuned for BW
  - test the recipe on GPT-2 with lr ×1.5

## 10. Working preferences of this user

- Wants **train-loss** wins over the LayerNorm baseline (convergence speed). Val is secondary: "I don't want you to kill the job because of the val loss."
- Budget-conscious with GPU hours: prefer cheap diagnostics and short (2000-iteration) GPT-2 runs before long ones. About 8–10 GPU-hours were spent on GPT-2 after the fix.
- Prefers that the agent keeps working without asking questions or halting when told to "think what to do and do". Wants concise, clean reports (no walls of text). The report baseline line should be black and thick.
- Wants causality guaranteed: "I don't want this leak to happen again w/o noticing." Never bypass the startup check except for deliberate diagnostics.
- Explain claims carefully. The user challenges hand-wavy explanations (for example the boost-asymmetry discussion), so back claims with measurements.
