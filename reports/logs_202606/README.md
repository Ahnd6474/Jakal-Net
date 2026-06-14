# 2026-06 Selected Run Logs

This archive keeps only the runs that mattered for the June knowledge-memory comparison, using short paths and filenames for easier download on other machines.

Runs:

- `base`: transformer baseline
  - source run: `20260606_202641_20260606-transformer-baseline-d1024-l10-ff3-decay-tb-v2`
  - best val: `2.7341` at step `143200`
  - last train: `2.3228` at step `144552`
- `h1`: post-lookup attention-sharing memory, `hop=1`
  - source run: `20260610_222207_20260610-knowledge-postlookup-attnshare-k4096-d1024-l10-ff3-decay-tb`
  - best val: `2.7888` at step `84200`
  - last copied train: `2.4885` at step `84814`
- `h2`: residual multi-hop memory, `hop=2`
  - source run: `20260614_195613_20260614-knowledge-postlookup-hop2res-k4096-d1024-l10-ff3-decay-tb`
  - copied very early in training
  - no validation yet at archive time

Per-run files:

- `cfg.json`: training config snapshot
- `hist.jsonl`: scalar history
- `best.json` / `last.json`: checkpoint metadata when available
- `launch.log`: detached launch log
- `tb.tfevents.gz`: compressed TensorBoard event snapshot

Clean-up applied:

- Deleted TensorBoard event logs for the failed baseline attempt `20260606_200950_...`
- Deleted TensorBoard event logs for the non-useful `qreuse` run `20260612_120537_...`

Not included:

- Full checkpoint `.pt` files, because they are too large for practical git upload
- Failed or clearly superseded experimental runs outside `base`, `h1`, and `h2`
