# Rank 256 vs Rank 384 in no-FFN propagation-only runs

Date: 2026-05-02

## Scope

This note summarizes the rank-scaling experiments for the no-FFN, `direction_only_values` configuration on the remote training box. The focus was to answer two questions:

1. Was the earlier `r256` result reproducible on the current code?
2. If yes, does `r384` fail because of code drift or because the rank change itself is harmful in this regime?

## Reference configuration

The reference run was recovered from the saved remote training metadata:

- `pairwise_rank=256`
- `route_rank=256`
- `batch_size=384`
- `grad_accum_steps=1`
- `learning_rate=5e-4`
- `warmup_start_lr=1e-4`
- `warmup_steps=500`
- `lr_decay_start_step=4000`
- `lr_decay_steps=0` (code path: decay over the remaining steps)
- `lr_min_ratio=0.2` -> floor `1e-4`
- `direction_only_values=True`
- `disable_feed_forward_layers=True`
- `disable_memory_feed_forward_layers=True`

Saved source: [reports/logs/2026-05-02-old-good-r256-config.txt](/mnt/c/Users/ahnd6/codex-sync/Jakal-Net/reports/logs/2026-05-02-old-good-r256-config.txt)

## Main result

The old `r256` run is reproducible on the current code when the exact old arguments are restored.

The earlier non-reproducible `r256` attempts were not actually the same experiment. They used a different schedule, mainly:

- `lr=2e-4` instead of `5e-4`
- early decay starting at `step 500` instead of hold until `step 4000`

Once those differences were removed, the current code reproduced the original `r256` curve exactly through the early phase.

## Evidence

### 1. Original good `r256` run

Key lines are in [reports/logs/2026-05-02-old-good-r256-log.txt](/mnt/c/Users/ahnd6/codex-sync/Jakal-Net/reports/logs/2026-05-02-old-good-r256-log.txt).

Important checkpoints:

- `step 200`: `train_loss=6.8942`
- `step 500`: `train_loss=5.4767`, `val_loss=5.6941`
- `step 1000`: `train_loss=5.0490`, `val_loss=5.2514`
- `step 1500`: `train_loss=4.7975`, `val_loss=5.0433`
- `step 4000`: `val_loss=4.7356`
- `step 5000`: `val_loss=4.7027`
- `step 6500`: `val_loss=4.6513`

### 2. Current-code `r256` control run with exact old arguments

Key lines are in [reports/logs/2026-05-02-currentcode-r256-repro-log.txt](/mnt/c/Users/ahnd6/codex-sync/Jakal-Net/reports/logs/2026-05-02-currentcode-r256-repro-log.txt).

Early steps matched exactly:

- `step 25`: `8.4434`
- `step 50`: `7.7509`
- `step 75`: `7.5897`
- `step 100`: `7.5619`
- `step 125`: `7.5310`
- `step 150`: `7.4801`
- `step 175`: `7.2444`
- `step 200`: `6.8942`

Conclusion: current code is not the reason `r256` failed earlier. The mismatch came from altered LR/schedule settings.

### 3. Current-code `r384` control run under the same settings

Key lines are in [reports/logs/2026-05-02-currentcode-r384-log.txt](/mnt/c/Users/ahnd6/codex-sync/Jakal-Net/reports/logs/2026-05-02-currentcode-r384-log.txt).

Under the same schedule, `r384` is much worse:

- `step 1000`: `train_loss=6.9964`, `val_loss=7.1334`
- `step 1500`: `train_loss=6.6421`, `val_loss=6.7827`

Compared to the reproduced `r256` control:

| Step | `r256` train | `r256` val | `r384` train | `r384` val |
|---|---:|---:|---:|---:|
| 1000 | 5.0490 | 5.2514 | 6.9964 | 7.1334 |
| 1500 | 4.7975 | 5.0433 | 6.6421 | 6.7827 |

Conclusion: with code and schedule fixed, raising rank from `256` to `384` is sufficient to break the run.

## Interpretation

The current best explanation is not "code drift" and not "schedule drift". It is the rank change itself.

The working hypothesis is:

- `r256` provides a useful low-rank bottleneck.
- `r384` removes too much of that constraint.
- In practice this looks less like "more capacity helps" and more like "extra projection parameters learn unnecessary directions and interfere with optimization."

This is consistent with:

- exact reproduction of the old `r256` result on current code
- immediate degradation when only rank is increased to `384`

## Follow-up attempt

An additional `r384` run was started with square low-rank projections frozen to identity to test the "unnecessary projection learning" hypothesis more directly. That run was deprioritized before evaluation completed, so there is no claim here about its outcome.

## Current run

The active continuation run is:

- `r256`
- same old-good schedule shape
- `epochs=3.0`
- `lr_decay_start_step=4000`
- `lr_decay_steps=5153`

The fixed `lr_decay_steps=5153` preserves the original 1-epoch decay length and lets the extra epochs run at the floor instead of stretching the decay over all 3 epochs.

Startup snapshot: [reports/logs/2026-05-02-currentcode-r256-3epoch-startup.txt](/mnt/c/Users/ahnd6/codex-sync/Jakal-Net/reports/logs/2026-05-02-currentcode-r256-3epoch-startup.txt)

## Bottom line

For this regime, the key result is:

- `r256` is stable and reproducible.
- `r384` is not rescued by keeping the old-good schedule.
- The failure mode should be treated as a rank/optimization problem, not as a reproduction or code-version problem.
