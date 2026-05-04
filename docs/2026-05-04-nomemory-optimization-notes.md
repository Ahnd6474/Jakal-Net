# 2026-05-04 Nomemory Optimization Notes

Scope:
- model: `causal_memory`
- path: `disable_memory=True`, `disable_feed_forward_layers=True`
- optimizer focus: `adamw_fused`
- main question: early plateau break and late-stage refinement in the no-memory exact path

## Key observations

### 1. Plateau break should be judged by step, not curriculum stage
The useful discriminator is not stage transition. The useful region is the plateau-break window:
- `step 150`
- `step 175`
- `step 200`
- `step 225`

Recommended metric:
- `drop_150_225 = loss@150 - loss@225`
- `drop_175_225 = loss@175 - loss@225`

Interpretation used during experiments:
- `drop_150_225 ~ 0`: stuck on plateau
- `drop_150_225 >= 0.3`: break started
- `drop_150_225 >= 0.5`: strong break

### 2. Seed sensitivity is real
Baseline short-run patterns already showed that seeds split into qualitatively different behaviors.

Representative baseline examples:
- `seed 2027`: plateau break begins around `175~225`
- `seed 1337`: remains flat through `300`
- `seed 4242`: remains flat through `300`
- `seed 1001`: remains flat through `350`
- `seed 1111`: remains flat through `350`

Concrete traces from previous runs:
- `noisycmp730-baseline-seed2027.log`
  - `150: 7.5412`
  - `175: 7.4586`
  - `200: 7.2829`
  - `225: 7.0596`
  - `300: 6.2294`
- `noisycmp730-baseline-seed1337.log`
  - `150: 7.5419`
  - `175: 7.5604`
  - `200: 7.5486`
  - `225: 7.5387`
  - `300: 7.5615`
- `noisycmp730-baseline-seed4242.log`
  - `150: 7.5474`
  - `175: 7.5676`
  - `200: 7.5507`
  - `225: 7.5417`
  - `300: 7.5500`

Working interpretation:
- the architecture appears to have a deeper good basin
- but many seeds get trapped in small attractors before entering it

### 3. NoisyAdamW helped some seeds, but not reliably enough by itself
Short noisy experiments suggested it can help some plateau seeds eventually break, but not robustly enough to treat it as the main answer.

Representative example:
- `noisycmp730-lagrate128-seed1337.log`
  - plateau-like through `300`
  - then starts falling by `325~350`
- `noisycmp730-lagrate128-seed4242.log`
  - still effectively stuck through `500`

Interpretation:
- noisy smoothing can suppress some small attractors
- but the effect is inconsistent across seeds
- initialization and effective batch still look more fundamental

### 4. Effective batch size matters
Late-stage continuation from an existing checkpoint with:
- `grad_accum_steps=4`
- `lr=1e-4`

looked materially better than the smaller-effective-batch continuation.

Observed run:
- `wiki2m-nomemory-adamw-seed2027-20260504`

Relevant later checkpoints from the log:
- `step 17000`: `train_loss=4.9039`, `val_loss=4.8988`
- `step 17500`: `train_loss=4.8878`, `val_loss=4.8949`

Interpretation:
- this model is sensitive to gradient noise
- larger effective batch helps suppress small-attractor behavior

## Code changes added for the next experiment round

The following initialization knobs were added so init experiments can be run directly from `train_causal_memory_lm.py`:

- `--pairwise-proj-init-std`
- `--pairwise-core-init-scale`
- `--prediction-proj-init-std`
- `--anchor-val-init-std`

They currently target the no-memory path components that matter most for plateau entry:
- sequence propagation pairwise modules
- prediction propagation pairwise modules
- `s_prediction_proj`
- `anchor_val`

## Recommended next experiment order

1. Baseline seed distribution under the current optimizer schedule
2. Initialization sweep on representative seeds
   - breaking seed: `2027`
   - stuck seeds: `1337`, `4242`, `1001`
3. Only after init signal is clear, revisit noisy optimizer variants

Recommended first init candidates:
- baseline
- `pairwise_proj_init_std=0.01`
- `pairwise_proj_init_std=0.005`
- `pairwise_proj_init_std=0.01`, `pairwise_core_init_scale=0.05`

Recommended short-run evaluation window:
- run to roughly `225~300` steps
- compare `drop_150_225` first
