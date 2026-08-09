# Batch Whitening In nanoGPT: Notes And Reference

This document is a reference for the Batch Whitening (BW) normalization work in the nanoGPT experiments. It covers the normalization layers and their affine parameters, how the code selects a normalization path, the three centering modes and their training/inference behavior, calibration and momentum, what "orthonormalized" means after BW and LayerNorm, the verification methodology, and how dropout interacts with BW at inference.

It begins by comparing the three centering modes used in the BW experiments:

- `leave_one_out`
- `running_mean`
- `calibrated`

The modes all answer the same question: **which mean should be subtracted from the activations during training, and which stored statistics should be used at inference?**

In the code, activations entering the normalization block have shape `(B, T, C)`:

- `B`: batch size / number of sequences
- `T`: sequence length / tokens per sequence
- `C`: channel dimension / embedding size

Channels are split into groups, and each group has its own mean, and for full Batch Whitening, its own covariance.

## Affine Scale And Bias

The normalization layers have two related concepts:

- `learn_affine`: whether the layer learns a multiplicative scale parameter $\gamma$.
- `bias`: whether the layer learns/adds an additive shift parameter $\beta$.

For a standard normalized activation $\hat{x}$, the affine output is:

$$
y = \gamma \hat{x} + \beta
$$

If `bias=False`, the additive term is removed:

$$
y = \gamma \hat{x}
$$

For `LayerNorm`, the normalized activation is:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

For mean-only centering, the normalized activation is:

$$
\hat{x} = x - \mu
$$

For full Batch Whitening, each channel group uses:

$$
\hat{x} = W(x - \mu)
$$

where $W$ is the whitening transform derived from the group covariance.

In the current nanoGPT code:

| Layer / helper call | `learn_affine` actual value | `bias` actual value | Notes |
|---------------------|-----------------------------|---------------------|-------|
| `LayerNorm(config.n_embd, bias=config.bias)` | `True` | `config.bias` (`False` in current `train.py`) | `LayerNorm` always has learned scale. Bias follows `config.bias`. |
| `batch_center_only(...)` from `BatchCenterBlock.forward()` | `True` | `True` | These are the defaults used by the helper call. `BatchCenterBlock` passes learned `gamma` and `beta`. |
| `batch_orthonorm(..., bias=False)` from `BatchWhiteningBlock.forward()` | `True` | `False` | The helper call does not add beta internally. `BatchWhiteningBlock.forward()` adds `self.beta` after the helper returns. |

## How The Code Selects The Normalization Path

There are exactly two user-facing settings:

```python
normalization_type = "full_bw"      # or "center_only" or "layer_norm"
batch_center_mode = "running_mean"  # or "leave_one_out" or "calibrated"
```

`normalization_type` selects the implementation:

| `normalization_type` | Layer selected by `_make_norm_layer()` | Meaning |
|----------------------|----------------------------------------|---------|
| `"layer_norm"` | `LayerNorm` | Original baseline: no BW and no batch centering. |
| `"center_only"` | `BatchCenterBlock` | Mean centering only, no covariance whitening. |
| `"full_bw"` | `BatchWhiteningBlock` | Mean centering plus covariance whitening. |

`batch_center_mode` selects the sub-mode inside the `"center_only"` and `"full_bw"` implementations. It is ignored by the `"layer_norm"` baseline.

| `batch_center_mode` | Meaning |
|---------------------|---------|
| `"leave_one_out"` | Use pure leave-one-out batch statistics during training. |
| `"running_mean"` | Blend leave-one-out batch statistics with running statistics and update the running statistics. |
| `"calibrated"` | Calibrate running statistics before the training step and do not overwrite them during training. |

`bw_iter_calibration` is internal now. The code sets it automatically from:

```python
bw_iter_calibration = (
    normalization_type != "layer_norm" and batch_center_mode == "calibrated"
)
```

`FirstBlock`, `Block`, and `ln_f` all use `_make_norm_layer()`, so this selection rule is shared.

To run the original no-BW baseline:

```python
normalization_type = "layer_norm"
```

## Mean-Only Centering

Mean-only centering is implemented by `BatchCenterBlock` and `batch_center_only()` in `nlp/nano_gpt/model.py`.

This path is a debugging mode. It subtracts a mean, applies the affine scale/bias, and does **not** whiten by covariance.

Enable mean-only centering by setting:

```python
normalization_type = "center_only"
batch_center_mode = "leave_one_out"  # or "running_mean" or "calibrated"
```

When `normalization_type="center_only"`, the model uses `BatchCenterBlock`.

### Mean-Only Settings

These are the top-level settings in `train.py`/`GPTConfig` that select each mean-only mode for layers created through `_make_norm_layer()`.

| Mode | `normalization_type` | `batch_center_mode` | `bw_iter_calibration` | Layer selected | Notes |
|------|---------------------|-------------------|-----------------------|----------------|-------|
| `leave_one_out` | `"center_only"` | `"leave_one_out"` | Auto: `False` | `BatchCenterBlock` | Uses pure leave-one-out centering during training. |
| `running_mean` | `"center_only"` | `"running_mean"` | Auto: `False` | `BatchCenterBlock` | Uses mixed leave-one-out + `running_mean`, and updates `running_mean` during training. |
| `calibrated` | `"center_only"` | `"calibrated"` | Auto: `True` | `BatchCenterBlock` | Calibration commits `running_mean`; the calibrated mode itself does not update `running_mean` during training. |

`bw_iter_calibration` is derived automatically from `batch_center_mode == "calibrated"` after config overrides. You should not need to set it manually.

### Mean-Only Training Behavior

During training with `B > 1`, the code first computes a leave-one-out mean for every sequence:

```text
mean_other[b] = mean of all tokens from all sequences except sequence b
```

Then each mode decides how much to trust that current-batch leave-one-out mean versus the stored `running_mean`.

| Mode | Training mean subtracted | Updates `running_mean` during training? | What gets stored for inference? |
|------|--------------------------|-----------------------------------------|---------------------------------|
| `leave_one_out` | Pure leave-one-out mean: `mean_other` | Yes, but only by copying the current full-batch mean | Last training batch mean |
| `running_mean` | Mixed mean: `(1 - momentum) * running_mean + momentum * mean_other` | Yes, EMA update using current full-batch mean | EMA running mean |
| `calibrated` | Mixed mean: `(1 - momentum) * calibrated_running_mean + momentum * mean_other` | No | Calibrated mean from the calibration pass |

### Mean-Only Inference Behavior

Inference is simpler. All three modes use the stored `running_mean` directly:

| Mode | Inference mean subtracted | Practical implication |
|------|---------------------------|-----------------------|
| `leave_one_out` | Last training batch mean | Inference depends on the final stored batch mean, so it can be noisy. |
| `running_mean` | EMA running mean | Inference uses a smoothed estimate accumulated during training. |
| `calibrated` | Calibrated running mean | Inference uses the mean collected from calibration batches. |

### Mean-Only Summary

| Mode | Main idea | Best used to test |
|------|-----------|-------------------|
| `leave_one_out` | Use only the current batch, excluding each sequence from its own mean. | Whether pure batch-dependent centering is enough. |
| `running_mean` | Blend current leave-one-out mean with an EMA running mean. | Whether running statistics stabilize centering. |
| `calibrated` | Use a separately estimated running mean and do not overwrite it during training. | Whether stale/noisy running means are hurting training. |

## Full Batch Whitening

Full Batch Whitening is implemented by `BatchWhiteningBlock` and `batch_orthonorm()` in `nlp/nano_gpt/model.py`.

This path subtracts a mean **and** whitens by covariance. In other words, it tries to make each channel group zero-mean and approximately identity-covariance.

Enable it with:

```python
normalization_type = "full_bw"
batch_center_mode = "leave_one_out"  # or "running_mean" or "calibrated"
```

Full BW is selected by `normalization_type="full_bw"`. The same `batch_center_mode` string chooses the full-BW sub-mode.

### Full BW Settings

These are the settings and runtime flags needed for the full Batch Whitening variants.

| Full BW behavior | `normalization_type` | `batch_center_mode` | `bw_iter_calibration` | BW `momentum` | `update_running_stats` after calibration | `update_running_cov_stats` after calibration | Notes |
|------|---------------------|-------------------|-----------------------|---------------|------------------------------------------|----------------------------------------------|-------|
| `leave_one_out` | `"full_bw"` | `"leave_one_out"` | Auto: `False` | `0.1` by default | `True` | `True` | Uses pure leave-one-out mean/cov during training and stores current stats for inference. |
| `running_mean` | `"full_bw"` | `"running_mean"` | Auto: `False` | `0.1` by default | `True` | `True` | Standard full BW behavior. Running mean/cov are updated during training. |
| `calibrated` | `"full_bw"` | `"calibrated"` | Auto: `True` | `0.1` by default | `False` | `False` | Calibration commits both `running_mean` and `running_cov`, then freezes both so the training batch does not overwrite them. |

The `update_running_stats` and `update_running_cov_stats` flags are layer attributes on `BatchWhiteningBlock`. They default to `True`; `train.py` sets them to `False` after each calibration commit. `bw_iter_calibration` is derived automatically from `batch_center_mode == "calibrated"` after config overrides.

### Full BW Training Behavior

During training with `B > 1`, full BW computes:

- A leave-one-out mean for centering each sequence.
- A leave-one-out covariance for whitening each sequence.
- Stored running statistics for inference and for mixed training centering.

The centering formula depends on `batch_center_mode`:

```text
leave_one_out:
    current_mean = mean_other

running_mean / calibrated:
    current_mean = (1 - momentum) * running_mean + momentum * mean_other
```

Covariance is also maintained in `running_cov` and used for the Cholesky whitening step.

| Mode | Training mean subtracted | Training covariance used | Updates `running_mean`? | Updates `running_cov`? |
|------|--------------------------|--------------------------|--------------------------|------------------------|
| `leave_one_out` | Pure leave-one-out mean selected by `batch_center_mode="leave_one_out"` | Current leave-one-out covariance | Yes, copies current batch mean | Yes, copies current leave-one-out covariance |
| `running_mean` | Mixed mean: `(1 - momentum) * running_mean + momentum * mean_other` | Running covariance updated from current leave-one-out covariance | Yes, EMA update | Yes |
| `calibrated` | Mixed mean using a calibrated `running_mean` | Calibrated `running_cov` from the same calibration batches | No, after calibration disables mean updates | No, after calibration disables covariance updates |

### Full BW Inference Behavior

In inference mode, full BW does not compute leave-one-out batch statistics. It uses stored buffers:

| Mode | Inference mean | Inference covariance | Practical implication |
|------|----------------|----------------------|-----------------------|
| `leave_one_out` | Last stored batch mean | Last stored covariance | Inference can be noisy because the stored stats come from the last training batch/update. |
| `running_mean` | EMA running mean | EMA running covariance | Standard BatchNorm-like behavior: inference uses smoothed running statistics. |
| `calibrated` | Calibrated running mean | Calibrated running covariance | Mean and covariance come from the same calibration batches. |

### Calibration Flow

Calibrated mode is driven from `train.py` by the mode string:

```python
batch_center_mode = "calibrated"
```

`train.py` derives this internal flag after config overrides:

```python
bw_iter_calibration = (
    normalization_type != "layer_norm" and batch_center_mode == "calibrated"
)
```

Before each training iteration, after covariance warmup is finished for full BW:

1. Sample `bw_iter_calibration_num_batches` random training batches.
2. Run the model in `train()` / no-grad mode on those batches, so dropout matches the training forward while gradients are not recorded.
3. While `collect_calib_stats=True`, BW/center layers force training-style normalization math so calibration matches training-time centering without recording gradients.
4. Each BW layer accumulates pooled input statistics: `sum_x`, `sum_xxT`, and count.
5. `bw_calibration_commit()` copies the calibrated mean and covariance into `running_mean` and `running_cov`.
6. `set_bw_update_running_stats(False)` prevents training from overwriting the calibrated `running_mean`.
7. `set_bw_update_running_cov_stats(False)` prevents training from overwriting the calibrated `running_cov`.

For mean-only centering, the same flow calibrates `running_mean`; there is no covariance buffer to update.

## Training vs Inference Differences

| Path | Training | Inference |
|------|----------|-----------|
| Mean-only `leave_one_out` | Subtracts leave-one-out batch mean. Stores current full-batch mean. | Subtracts last stored mean. |
| Mean-only `running_mean` | Subtracts mix of EMA running mean and leave-one-out mean. Updates EMA. | Subtracts EMA running mean. |
| Mean-only `calibrated` | Subtracts mix of calibrated mean and leave-one-out mean. Does not update mean. | Subtracts calibrated mean. |
| Full BW `leave_one_out` | Uses current leave-one-out mean/cov selected by `batch_center_mode="leave_one_out"`. Stores current stats. | Uses last stored mean/cov. |
| Full BW `running_mean` | Uses mixed mean and running covariance. Updates running mean/cov. | Uses stored running mean/cov. |
| Full BW `calibrated` | Uses calibrated mean mixed with leave-one-out mean and calibrated covariance. Does not update mean/cov after calibration. | Uses calibrated mean and calibrated covariance. |

## Short Interpretation

`leave_one_out` is the most batch-dependent mode. It removes each sequence's own contribution from the mean/cov estimate used to normalize that sequence.

`running_mean` is the standard smoothed-statistics mode. It still uses current batch information during training, but it blends it with stored running estimates and updates those estimates over time.

`calibrated` tests whether online running statistics are the problem. It refreshes `running_mean` and `running_cov` from extra calibration batches before training, then prevents the training batch from overwriting those calibrated statistics.

### Possible reason `leave_one_out` trains better

One plausible reason `leave_one_out` can outperform the other two modes is that the current-batch statistic has a stronger path through the training computation.

For `leave_one_out`, the current leave-one-out mean is used directly:

```text
current_mean = mean_other
```

So the current-batch centering statistic has coefficient `1`.

For `running_mean` and `calibrated`, the current leave-one-out mean is only mixed in with weight `momentum`:

```text
current_mean = (1 - momentum) * running_mean + momentum * mean_other
```

With the default `momentum=0.1`, only `0.1` of the current-batch mean participates in the normalized activation. The other `0.9` comes from the stored running or calibrated buffer. This means the immediate training signal through the current-batch centering path is much weaker than in `leave_one_out`.

This is a hypothesis for why `leave_one_out` may train better: it applies the full current-batch correction each forward pass, while `running_mean` and `calibrated` damp that correction by the momentum factor. In full BW, the same intuition also applies to the covariance side: `leave_one_out` uses the current leave-one-out covariance directly, while the other modes rely more heavily on stored covariance statistics.

## Momentum And Calibration Batch Notes

### What $B$, $T$, and $C$ mean

In this repo, Batch Whitening operates on tensors shaped `(B, T, C)`:

- **$B$**: batch size / number of sequences.
- **$T$**: sequence length / number of tokens per sequence.
- **$C$**: channel dimension / embedding size.

With DDP, each process sees its local micro-batch $B$. Gradient accumulation does not change the forward-pass $B$; it just runs multiple forwards before one optimizer step.

### Group size $g$

Channels are split into groups. In `batch_orthonorm()`, `group_size = C // n_groups`; in `BatchWhiteningBlock`, this is stored as `self.group_size`.

In `get_batch_whitening_config(B, T, C, ...)`, the returned `blk_size` is this group size:

$$
g = \texttt{blk\_size} = \texttt{self.group\_size}
$$

The heuristic computes:

$$
n_{\text{samples}} = \frac{(B-1)\cdot T}{1-\text{momentum}}
$$

and chooses $g$ so that:

$$
n_{\text{samples}} \ge \frac{g^2}{2\cdot \text{threshold}^2}
$$

Solving gives:

$$
g \approx \sqrt{2 \cdot n_{\text{samples}} \cdot \text{threshold}^2}
$$

The code then snaps this raw $g$ to a divisor of $C$ so the channel groups partition cleanly.

Important caveat: `BatchWhiteningBlock.__init__()` currently calls `get_batch_whitening_config(B=32, T=64, ...)` because runtime $B,T$ are not known at init time.

### Why calibrated mode uses extra batches

The concern behind calibrated mode is that `running_mean` and `running_cov` may lag behind a fast-changing model distribution. Instead of relying on online EMA updates, calibrated mode estimates stats from extra random batches before the training step.

For $k$ calibration batches, the approximate sample count is:

$$
n_{\text{samples}} \approx k \cdot (B-1)\cdot T
$$

The number `bw_iter_calibration_num_batches = 10` is not derived directly from momentum, but it matches the intuition of an EMA with update weight $\alpha=0.1$, whose effective window is roughly $1/\alpha \approx 10$ batches.

### Pooled sufficient statistics

Calibration avoids materializing one huge mega-batch by accumulating pooled sufficient statistics:

- $\sum x$
- $\sum xx^\top$
- $N$, the total sample count

Then:

$$
\mu = \frac{\sum x}{N},
\qquad
\Sigma = \frac{\sum xx^\top}{N} - \mu\mu^\top
$$

This gives the mean and covariance of the union of all calibration samples, up to numerical precision, without concatenating all batches into one tensor.

Averaging per-batch covariances is PSD, but it is not generally equal to the covariance of the union if per-batch means differ. The pooled-stat approach is the correct union covariance.

For calibration covariance, add $\varepsilon I$ once when committing the final pooled covariance, not to every intermediate estimate.

## What Orthonormalized Should Mean Here

For the nanoGPT Batch Whitening path, the main orthonormalization check is across **channels**, not across tokens.

The activation tensor has shape:

$$
X \in \mathbb{R}^{B \times T \times C}
$$

Batch Whitening treats the $B \times T$ positions as samples and whitens the $C$ feature channels, split into channel groups. For each group, the desired property is:

$$
\operatorname{Cov}_{B,T}(\hat{X}_{group}) \approx I
$$

This means:

- The group mean over the $B \times T$ samples should be near zero.
- The diagonal covariance entries should be near $1$.
- The off-diagonal covariance entries should be near $0$.

That is the relevant orthonormal property for BW: channel/group covariance close to identity over many token samples.

Token orthogonality is different and is generally not expected. Requiring token vectors to be mutually orthogonal would mean:

$$
\hat{x}_{b,t_1}^\top \hat{x}_{b,t_2} \approx 0
\qquad \text{for } t_1 \ne t_2
$$

Transformer token representations are allowed to be correlated because neighboring tokens, residual streams, and attention outputs carry related information.

### Why LayerNorm after BW does not invalidate the channel check

LayerNorm is applied per token after BW. For one token vector $z = \operatorname{BW}(x)$, ignoring learned affine parameters, LayerNorm computes:

$$
y = \frac{z - \operatorname{mean}(z)}{\operatorname{std}(z)}
$$

Equivalently, for each token $t$:

$$
y_t = a_t z_t + b_t \mathbf{1}
$$

where:

$$
a_t = \frac{1}{\operatorname{std}(z_t)},
\qquad
b_t = -\frac{\operatorname{mean}(z_t)}{\operatorname{std}(z_t)}
$$

LayerNorm subtracts one scalar mean from all channels of a token and multiplies all channels by one scalar standard-deviation factor. It does not apply a dense matrix across channels. It does not rotate the channel basis or mix channel $i$ into channel $j$ differently.

LayerNorm can still change covariance because $a_t$ and $b_t$ vary by token, but it is a scalar per-token centering/scaling operation, not a learned channel-mixing transform. It can also improve apparent channel normalization after BW by forcing each token vector's channel variance close to $1$.

So for `self.ln_2(self.bw_2(x))`, the main useful check is still channel/group covariance identity over $B \times T$ samples after the combined path. We should not require token vectors to be mutually orthogonal.

### Full BW covariance bug note

The full BW covariance must be computed from the same centered quantity that is whitened. For sequence `b`, both the mean and covariance should use all tokens from all sequences except `b`:

$$
\mu^{(-b)} = \operatorname{mean}_{s \ne b,\,t}(X_{s,t})
$$

$$
\Sigma^{(-b)} = \operatorname{Cov}_{s \ne b,\,t}(X_{s,t} - \mu^{(-b)})
$$

This is the correct formulation for channel whitening with sequence-level leave-one-out causality. A per-token leave-one-out mean,

$$
\operatorname{mean}_{s \ne b}(X_{s,t})
$$

is a different statistic and shrinks the covariance too much for channel whitening over the full $B \times T$ sample population.

## Verification: `ln_2(bw_2(x))` Grouped And Full-Matrix Checks

We verified the output of:

```python
self.ln_2(self.bw_2(x))
```

using `nlp/nano_gpt/verify_ln2_bw_orthonorm.py`. The script captures both `bw_2(x)` and `ln_2(bw_2(x))`, treats the $B \times T$ token positions as samples, and computes both grouped BW statistics and full-channel statistics.

For each BW channel group, the grouped whitening target is:

$$
\operatorname{Cov}_{B,T}(Y_{group}) \approx I
$$

This grouped check means:

- off-diagonal covariance entries should be close to $0$;
- diagonal covariance entries should be close to $1$, reported as `diag-1` close to $0$;
- `max|token_mean|` should be near $0$ and `mean token var` should be near $1$ after LayerNorm.

This is not the same as saying the full $C \times C$ covariance or full raw Gram matrix is identity after LayerNorm. The verifier now also reports:

- full covariance over all channels;
- full raw Gram $X^\top X / N$ over all channels, without sample-mean removal;
- the LayerNorm all-ones-direction constraint.

The closest-to-training verification used the Shakespeare data, the train-like GPT config, MPS, full BW, the real `block_size=64`, `batch_size=32`, `n_layer=6`, `n_head=6`, and `n_embd=384` settings:

```bash
cd /Users/hsivan/repos/batch_whitening/nlp/nano_gpt
uv run --with torch --with numpy python verify_ln2_bw_orthonorm.py \
  --device=mps \
  --real_data \
  --train_like \
  --batch_center_mode=<mode> \
  --iters=5 \
  --inference_iters=5 \
  --aggregate
```

For `running_mean`, the verification warmed up the EMA statistics first:

```bash
--batch_center_mode=running_mean --stable_cov --burn_in_iters=80
```

For `calibrated`, the verification first ran a train-mode no-grad calibration pass:

```bash
--batch_center_mode=calibrated --stable_cov --calib_iters=50
```

### Training results over 5 forwards

These are aggregate train-mode measurements over 5 random Shakespeare batches. The result is not a one-pass accident: all three modes pass the grouped BW check in every transformer block after `ln_2(bw_2(x))` when calibrated stats are collected in train mode.

**`leave_one_out`: all 6 blocks pass**

- Worst off-diagonal covariance: `0.059`
- Mean off-diagonal covariance range: `0.006529` to `0.01084`
- Diagonal covariance range: `0.9615` to `1.063`
- Mean diagonal error range: `0.005266` to `0.01243`
- Max token mean after LayerNorm: `4.16e-08`
- Mean token variance after LayerNorm: `1`

**`running_mean`: all 6 blocks pass**

- Worst off-diagonal covariance: `0.09442`
- Mean off-diagonal covariance range: `0.007269` to `0.01743`
- Diagonal covariance range: `0.9295` to `1.114`
- Mean diagonal error range: `0.007305` to `0.02397`
- Max token mean after LayerNorm: `4.036e-08`
- Mean token variance after LayerNorm: `1`

**`calibrated`: all 6 blocks pass**

- Worst off-diagonal covariance: `0.085`
- Mean off-diagonal covariance range: `0.01082` to `0.01621`
- Diagonal covariance range: `0.8326` to `1.096`
- Mean diagonal error range: `0.01241` to `0.02764`
- Max token mean after LayerNorm: `4.222e-08`
- Mean token variance after LayerNorm: `1`

The training results show that all three modes are convincingly close to identity in the **grouped BW sense** after `ln_2(bw_2(x))`: grouped off-diagonal covariance is small, grouped diagonal covariance is close to $1$, and the result holds across 5 measured forwards. This should not be read as a claim that the full $C \times C$ post-LayerNorm covariance or Gram matrix is identity. The calibrated result became comparable after calibration stats were collected in train mode instead of eval mode, which keeps dropout consistent with the training forward.

### Inference results over 5 forwards

The same script also ran 5 eval-mode forwards after the train measurements. These inference results do **not** show channel orthonormality for `ln_2(bw_2(x))` in this untrained verifier setup.

**`leave_one_out`: all 6 inference blocks fail channel orthonormality**

- Worst off-diagonal covariance: `8.632`
- Mean off-diagonal covariance range: `0.3784` to `0.5241`
- Diagonal covariance range: `0.4555` to `28.49`
- Mean diagonal error range: `0.3108` to `0.7235`
- Max token mean after LayerNorm: `7.873e-07`
- Mean token variance after LayerNorm range: `0.9975` to `0.9992`

**`running_mean`: all 6 inference blocks fail channel orthonormality**

- Worst off-diagonal covariance: `8.328`
- Mean off-diagonal covariance range: `0.377` to `0.5243`
- Diagonal covariance range: `0.4529` to `28.27`
- Mean diagonal error range: `0.3096` to `0.7251`
- Max token mean after LayerNorm: `8.04e-07`
- Mean token variance after LayerNorm range: `0.9975` to `0.9992`

**`calibrated`: all 6 inference blocks fail channel orthonormality**

- Worst off-diagonal covariance: `8.627`
- Mean off-diagonal covariance range: `0.3777` to `0.5248`
- Diagonal covariance range: `0.4521` to `28.24`
- Mean diagonal error range: `0.3104` to `0.7249`
- Max token mean after LayerNorm: `7.345e-07`
- Mean token variance after LayerNorm range: `0.9975` to `0.9992`

LayerNorm still does its per-token job in inference: token means are near zero and token variance is near $1$. But the grouped channel covariance after `ln_2(bw_2(x))` is not close to identity in eval mode here. So the verifier supports train-mode channel orthonormality for all three modes, but it does **not** support an inference-mode orthonormality claim for this untrained checkpoint.

## Final Conclusions From The Verifier

For train-mode behavior, the best default is still `leave_one_out`. It uses the current leave-one-out batch mean and covariance directly, so the current-batch whitening correction enters with full strength. `running_mean` is the practical smoothed-statistics alternative, and after warm-up it also passes the train-mode grouped BW check. `calibrated` is now plausible too, but only after fixing calibration to collect stats in `train()` mode with no gradients.

The calibrated issue was not that calibrated statistics are inherently stale. Calibration happens right before the training step, uses the latest network weights, and averages over more batches than the current training batch. The bug/mismatch was that calibration previously ran the whole model in `eval()` mode. BW layers forced training-style BW math while `collect_calib_stats=True`, but the rest of the model still behaved like eval mode, especially dropout. With `dropout=0.2`, calibration collected covariance from dropout-off activations and then applied it to dropout-on training activations. That made the calibrated covariance systematically mismatched and caused poor train-mode whitening in deeper blocks.

The fix is to run calibration forwards as:

```python
model.train()
with torch.no_grad():
    model(Xc)
```

This keeps dropout and other train-mode module behavior consistent with the real training forward, while still preventing calibration from contributing gradients. After this change, calibrated mode passed the 5-forward train-mode grouped verifier across all 6 transformer blocks after `ln_2(bw_2(x))`.

The inference result is a separate issue. In eval mode, BW uses stored `running_mean` and `running_cov`; it does not recompute current-batch leave-one-out covariance. The verifier's stored stats were collected from train-mode activations, with dropout enabled, but inference activations are produced with dropout disabled. The whitening transform is therefore approximately:

$$
\Sigma_{\text{stored train}}^{-1/2}
$$

applied to activations whose covariance is:

$$
\Sigma_{\text{eval}}
$$

The measured output covariance is:

$$
\Sigma_{\text{stored train}}^{-1/2}
\Sigma_{\text{eval}}
\Sigma_{\text{stored train}}^{-1/2}
$$

This is close to identity only if $\Sigma_{\text{eval}} \approx \Sigma_{\text{stored train}}$. In the untrained verifier setup, they do not match. That is why inference `bw_2` variances were far from $1$, and why `ln_2` could restore per-token variance but not grouped channel covariance identity. The verifier therefore supports train-mode orthonormality, but it does not prove inference-mode channel orthonormality for this untrained checkpoint.

## LayerNorm Means Vector-Normalized, Not Full Orthonormal

The more precise post-`ln_2` target is **channel orthogonality plus vector normalization**, not full-channel orthonormality in the original coordinate space.

LayerNorm acts on each token vector $x \in \mathbb{R}^C$ and enforces:

$$
\frac{1}{C}\sum_i y_i \approx 0
$$

and:

$$
\frac{1}{C}\sum_i y_i^2 \approx 1
$$

So each token output lies in the zero-mean channel subspace and has fixed length:

$$
y^\top \mathbf{1} \approx 0,
\qquad
\frac{\lVert y \rVert^2}{C} \approx 1
$$

Because $y^\top \mathbf{1} \approx 0$ for every token, the all-ones direction has near-zero second moment after LayerNorm. Therefore the full raw Gram matrix or full covariance matrix cannot be exactly identity over all $C$ channels: identity would have variance $1$ in the all-ones direction, while LayerNorm removes that direction.

The verifier now reports:

- grouped covariance/Gram: the BW-group `16 x 16` matrices;
- full covariance: centered covariance over all $C$ channels;
- full raw Gram: $X^\top X / N$ over all $C$ channels, without subtracting the sample mean;
- full-token LayerNorm constraint: energy in the all-ones direction and token norm.

A quick verifier run after `ln_2(bw_2(x))` showed:

```text
grouped covariance:
  max|cov-I| = 0.1128
  mean|offdiag| = 0.02033
  diag = [0.9437, 1.113]

grouped raw Gram X^T X/N:
  max|G-I| = 0.113
  mean|offdiag| = 0.02034
  diag = [0.9438, 1.113]

full covariance:
  max|Cov-I| = 0.4597
  mean|offdiag| = 0.08103
  diag = [0.9437, 1.113]

full raw Gram X^T X/N:
  max|G-I| = 0.4604
  mean|offdiag| = 0.08102
  diag = [0.9438, 1.113]

E[(x·1/sqrt(C))^2] = 1.757e-14
max|x·1/sqrt(C)| = 4.321e-07
mean||x||^2/C = 1
||x||^2/C = [1, 1]
trace(full G)/C = 1
```

This confirms the supervisor's point. The grouped `16 x 16` matrices can look close to identity because they only inspect the BW blocks. The full $C \times C$ covariance and Gram are much farther from identity, and the all-ones direction has essentially zero energy. After LayerNorm, the token vector is normalized and constrained to the zero-mean subspace. We can still measure approximate grouped channel decorrelation, but we should not claim exact full-channel orthonormality after `ln_2`.

### How to read covariance vs Gram, diagonal vs off-diagonal

The covariance and raw Gram matrices answer slightly different questions:

$$
\operatorname{Cov}(X) = \mathbb{E}\left[(X-\mu)(X-\mu)^\top\right]
$$

$$
G = \mathbb{E}\left[XX^\top\right]
$$

They are related by:

$$
G = \operatorname{Cov}(X) + \mu\mu^\top
$$

So if covariance and raw Gram look almost identical, it means the sample mean $\mu$ across the measured $B \times T$ token samples is small. In the verifier run above, `max|sample_mean|` after `ln_2` was small, so subtracting the sample mean did not materially change the matrix. The conclusion is not an artifact of covariance mean-removal.

Grouped and full matrices also have the same diagonal entries. This is expected. A diagonal entry is just the per-channel second moment or variance:

$$
G_{ii} = \mathbb{E}[X_i^2]
$$

or:

$$
\operatorname{Cov}_{ii} = \operatorname{Var}(X_i)
$$

Whether channel $i$ is viewed inside a `16 x 16` BW group or inside the full $C \times C$ matrix, its diagonal value is the same. Grouping changes which off-diagonal channel pairs are inspected; it does not change the per-channel diagonal.

This is why the important difference between grouped and full checks appears mostly in the off-diagonal values:

- grouped matrices inspect correlations **within** each BW group;
- full matrices also inspect correlations **across** different BW groups.

BW is implemented groupwise, so it directly tries to decorrelate channels within each BW group. It does not directly whiten cross-group correlations. After LayerNorm, all groups are also coupled by the token-wise mean and variance. Therefore the grouped matrices can look close to identity while the full matrix still has noticeable off-diagonal structure.

LayerNorm explains why the average diagonal is $1$, not why every channel diagonal is exactly $1$. For every token, LayerNorm enforces:

$$
\frac{1}{C}\sum_i y_i^2 \approx 1
$$

Averaging this over tokens gives:

$$
\frac{1}{C}\operatorname{trace}(G) \approx 1
$$

So `trace(full G)/C = 1` is expected from LayerNorm. But LayerNorm does not guarantee $G_{ii} \approx 1$ for every individual channel. When the verifier shows a diagonal range like `[0.9438, 1.113]`, that means individual channel energies are close to $1$ but not forced to be exactly $1$. That closeness is evidence that BW/channel balancing mostly survived LayerNorm, not a guarantee from LayerNorm alone.

## Dropout Breaks BW Whitening At Inference

This section explains a train/inference inconsistency that is specific to Batch Whitening: **BW whitens correctly during training but the whitening degrades at inference, and dropout is the cause.**

### Symptom

During training the grouped covariance after `bw_2` is close to identity (diagonal $\approx 1$). At inference (evaluation), the diagonal collapses below $1$ and the effect worsens with depth. The whitening is *over-scaling* the activations.

### Why dropout causes it

BW estimates and stores whitening statistics (`running_cov`, and the calibrated covariance) **during training, with dropout on**. Dropout inflates the variance of the activations it touches (it zeros a fraction $p$ of units and rescales the survivors by $1/(1-p)$). So the stored covariance reflects a *dropout-inflated* distribution.

At inference, dropout is off, so the activations BW sees have *lower* variance than the stored `running_cov`. Whitening divides by (the Cholesky factor of) that too-large covariance, so the output variance comes out **below 1**:

$$
Y = L^{-1}(X - \mu), \qquad \operatorname{Cov}(Y) = L^{-1}\operatorname{Cov}(X)L^{-\top}
$$

If $L L^\top = \operatorname{Cov}_{\text{train, dropout on}}$ but $\operatorname{Cov}(X) = \operatorname{Cov}_{\text{eval, dropout off}}$ is smaller, then $\operatorname{Cov}(Y) < I$. This is a **scale/variance** mismatch, not a decorrelation failure: the off-diagonals stay as small as in the healthy case, only the diagonal shrinks.

### Which dropout layers matter

BW reads the residual stream `x`. Only dropouts that write to the residual stream **at full magnitude** create the mismatch. In the current block structure:

```python
x = x + self.attn(self.ln_1(x))          # attn ends with resid_dropout(c_proj(y))
x = x + self.mlp(self.ln_2(self.bw_2(x)))  # bw_2 whitens the residual x; mlp ends with mlp_dropout
```

- `resid_dropout` — end of `CausalSelfAttention.forward`: `self.resid_dropout(self.c_proj(y))`. Drops the attention output *before* it is added to the residual `x`, so it perturbs the **same** block's `bw_2(x)` input.
- `mlp_dropout` — end of `MLP.forward`: `self.dropout(...)` after `c_proj`. Drops the MLP output *before* it is added to the residual, so it perturbs the **next** block's `bw_2(x)` input.
- `attn_dropout` — inside `CausalSelfAttention`, on the softmax attention weights (`self.attn_dropout(att)`, or `dropout_p` in the flash path). It only reweights `att @ v`, so it is renormalized/averaged and its variance footprint on the residual stream is negligible.

### Experiments (verifier, `running_mean` mode, real Shakespeare data)

Statistics were accumulated in train mode (burn-in), then the grouped `bw_2` covariance was measured in eval mode. Each cell is `mean|diag-1| / mean token var` at **inference** (ideal is `0 / 1`). Base/embedding dropout was held at `0.0` to isolate the three residual/attention dropouts.

| block | A: all 0.2 | B: all 0.0 | C: resid=0 (attn=mlp=0.2) | D: attn=0.2, mlp=resid=0 |
|-------|-----------|-----------|---------------------------|--------------------------|
| 0     | 0.058 / 0.94 | 0.021 / 0.98 | 0.029 / 0.97 ✅ | 0.033 / 0.97 ✅ |
| 1     | 0.179 / 0.82 | 0.023 / 0.98 | 0.161 / 0.84 ❌ | 0.032 / 0.97 ✅ |
| 2     | 0.199 / 0.80 | 0.025 / 0.98 | 0.181 / 0.82 ❌ | 0.033 / 0.97 ✅ |
| 3     | 0.209 / 0.79 | 0.027 / 0.98 | 0.190 / 0.81 ❌ | 0.032 / 0.97 ✅ |

Reading the columns:

- **A (all dropout on)** reproduces the failure: inference diagonal collapses to `0.79`–`0.94`, worse with depth.
- **B (no dropout)** is the control: inference diagonal $\approx 1$ everywhere. This confirms dropout is the cause.
- **C (`resid_dropout=0` only)** fixes **only block 0**. Block 0's `bw_2` has no upstream `mlp_dropout`, so removing `resid_dropout` cleans its input. Deeper blocks still read a residual stream that earlier blocks' `mlp_dropout` (still `0.2`) perturbed, so they stay broken.
- **D (`mlp_dropout=0` and `resid_dropout=0`, `attn_dropout=0.2`)** fixes **all blocks**, even with `attn_dropout` still on. This proves `mlp_dropout` is the deeper-block culprit and `attn_dropout` is harmless.

**Conclusion:** setting `resid_dropout=0` is necessary but not sufficient. The two dropouts that write to the residual stream at full magnitude (`resid_dropout` and `mlp_dropout`) are what break BW at inference; `attn_dropout` does not.

### Does it get worse with more iterations?

No. It **plateaus** with training iterations and compounds only with **depth**. In `running_mean` mode `running_cov` is an EMA that converges to the dropout-on training covariance, so the train↔eval gap is a fixed multiplicative bias, not a divergent one. Increasing burn-in from 40 to 150 iterations (and inference averaging from 3 to 8 forwards) left the deficit essentially unchanged:

| block | burn-in 40 | burn-in 150 |
|-------|-----------|-------------|
| 0 | 0.058 / 0.939 | 0.058 / 0.939 |
| 1 | 0.179 / 0.819 | 0.179 / 0.819 |
| 2 | 0.199 / 0.799 | 0.202 / 0.797 |
| 3 | 0.209 / 0.790 | 0.211 / 0.788 |

The only monotonic growth is across blocks (0 → 3), because each layer's residual-stream dropout perturbation accumulates for the next layer's BW.

### Does dropout also interfere *during* training?

A natural follow-up: dropout drops different units on every forward, so it changes the activation statistics BW sees at train time too. Does that break the training-time whitening, or only the eval mismatch above?

This was tested directly by feeding **one fixed input batch** through the model `K = 8` times in train mode (`leave_one_out`, so there is no EMA state to confound the result) and capturing `block0.bw_2`, once with dropout `0.2` and once with dropout off (same weights):

| metric | dropout 0.2 | dropout 0.0 |
|--------|-------------|-------------|
| cross-forward output std (same fixed input) | mean `0.39`, max `3.24` | exactly `0.0` |
| within-forward whitening `mean\|diag-1\|` | `0.003` | `0.002` |
| within-forward `diag_mean` | `1.003` | `1.001` |

Two distinct conclusions:

1. **Dropout does inject large randomness into the training statistics.** With the *same* input, the `bw_2` output varies across forwards with `std ≈ 0.39` (up to `3.24` at some positions), versus *exactly* `0.0` without dropout (the forward is deterministic). Since the whitened output has unit variance, a cross-forward std of `~0.39` means roughly 40% of the signal at a given position is dropout-induced noise. So dropout genuinely changes what BW estimates and whitens on every step.

2. **But it does not degrade the training-time whitening quality.** Within each forward the output covariance is still essentially identity (`mean|diag-1| ≈ 0.003`, `diag ≈ 1.00`), the same as with dropout off. The reason is that within a single forward BW estimates the covariance from the *same* dropout-perturbed batch it then whitens, so the covariance and the data are mutually consistent (both "dropout-on"). The randomness is present but self-consistent per forward.

This cleanly separates the two regimes. During **training**, dropout adds stochasticity (the intended regularization) but the whitening stays *correct* because the covariance and the data are both dropout-inflated — which is why the train-mode tables show `diag ≈ 1`. At **inference**, dropout turns off, so the data loses the variance inflation while the stored `running_cov` is still the dropout-inflated one, so the whitening over-scales to `diag ≈ (1-p) ≈ 0.8`. The failure is therefore specifically the train↔eval statistics gap, **not** a training-time whitening error.

### How bad is it, really

Moderate, and partly self-mitigating:

- It is a **scale** error (variance $\approx 0.79$–$0.94$, i.e. a 6–21% shrink at depth), **not** a decorrelation failure — off-diagonals are as small as in the no-dropout case.
- `bw_2` is immediately followed by `ln_2` (`ln_2(bw_2(x))`). LayerNorm re-normalizes each token to unit variance across channels, so a roughly-uniform scale deficit is **largely absorbed** by `ln_2`. What LayerNorm does not remove is the per-channel *spread* of the deficit (e.g. a diagonal range `[0.65, 0.90]`), since LayerNorm only fixes the global per-token scale, not per-channel relative differences.
- Net: a genuine, systematic train/eval inconsistency worth fixing, but not a catastrophic breakdown. The most direct measure of real impact is eval **loss** with vs without the fix, since `ln_2` sits between BW and the model output.

### Reproducing with the verifier

The verifier exposes per-layer dropout controls (`--attn_dropout`, `--mlp_dropout`, `--resid_dropout`; each falls back to `--dropout`/`--attn_dropout` when unset). Accumulate stats in train mode with burn-in, then compare the train vs inference `bw_2` diagonal:

```bash
# Experiment A: reproduce the failure (all residual-path dropout on)
python verify_ln2_bw_orthonorm.py --real_data --dataset=shakespeare_char \
  --n_layer=4 --n_embd=256 --block_size=128 --batch_size=64 \
  --batch_center_mode=running_mean --burn_in_iters=40 \
  --iters=1 --inference_iters=3 --aggregate \
  --dropout=0.0 --attn_dropout=0.2 --mlp_dropout=0.2 --resid_dropout=0.2

# Experiment D: fix (only the full-magnitude residual writers off)
python verify_ln2_bw_orthonorm.py ... --attn_dropout=0.2 --mlp_dropout=0.0 --resid_dropout=0.0
```

Compare the `mean|diag-1=...` and `mean token var=...` fields on the `block{i}.bw_2:` lines under the `## inference aggregate ...` section against the `## train aggregate ...` section.

> **⚠️ IMPORTANT — this problem is specific to the small-model / small-dataset regime.**
>
> nanoGPT uses dropout here (`attn_dropout = mlp_dropout = 0.2` in the `shakespeare_char` config) only because it is a tiny model trained on a tiny dataset (`# we expect to overfit on this small dataset` in `train.py`), so dropout is needed as a regularizer. The train/inference mismatch that breaks BW whitening exists **only because dropout is on during training and off at inference**.
>
> In large-scale pretraining (e.g. nanoGPT's GPT-2 reproduction on OpenWebText), the default is `dropout = 0.0` (`# for pretraining 0 is good` in `train.py`): with a massive dataset the model does not overfit, so no dropout is required. With `dropout = 0.0` there is no train/eval distribution gap, and BW whitens consistently at inference. In other words, at GPT-2 scale this dropout-induced BW degradation does not arise in the first place.
