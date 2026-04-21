# Remote Patch Inventory

Last updated: 2026-04-21

This document inventories `C:\Users\alber\remote_patch` before code is merged into the main repo.

## Summary

`remote_patch` is not a single patch set. It is a mixed stash of:

- full or partial repo snapshots
- direct replacement files
- one-off patch scripts
- scan-level fused scan experiments
- entmax integration work
- curriculum/training-script variants

The directories are small enough to track, but they overlap heavily. The safest approach is:

1. treat `scan_cuda` as the main source for scan-level fused recovery
2. treat `entmax` as the main source for signed entmax integration
3. treat `curriculum` as a training-script variant to compare, not blindly copy
4. treat top-level `patch_*.py` files as historical one-shot migration scripts, not source of truth

## Directory Summary

| Path | Files | Size | Role |
|---|---:|---:|---|
| `20260419_sync` | 8 | 0.48 MB | remote backup fragments of native backend and CUDA files |
| `curriculum` | 8 | 0.18 MB | training-script variants for curriculum scheduling |
| `entmax` | 16 | 1.08 MB | signed entmax integration plus native/backend work |
| `Jakal-Net-dialogue` | 8 | 0.63 MB | old dialogue-side snapshot fragments |
| `Jakal-Net-docrun` | 2 | 0.07 MB | tiny docrun snapshot fragments |
| `Jakal-Net-local-push` | 505 | 7.75 MB | large repo snapshot, useful as reference but noisy |
| `Jakal-Net-transfer` | 2 | 0.02 MB | transfer fragments |
| `scan_cuda` | 21 | 1.13 MB | scan-level fused scan, checkpoint trace, backward CUDA, tests, benches |

Top-level loose files are mostly one-off patch scripts:

- `patch_jakal_native_checkpoint_trace.py`
- `patch_jakal_chunk_backward.py`
- `patch_scan_backward_cuda.py`
- `release_gil_scan_backward.py`
- `fix_scan_backward_cuda_gradmode.py`
- `skip_unstable_backward_cuda_test.py`

These are useful for archaeology, but should not be committed as canonical implementation unless we intentionally want to keep the migration scripts.

## What Is In `scan_cuda`

This is the most important directory.

Primary files:

- `scan_cuda/causal_memory_lm_remote.py`
- `scan_cuda/native_backend.py`
- `scan_cuda/jakal_net_native.cpp`
- `scan_cuda/jakal_net_native_cuda.cu`
- `scan_cuda/jakal_net_native_cuda.h`
- `scan_cuda/test_causal_memory_lm.py`
- `scan_cuda/test_native_backend.py`
- `scan_cuda/test_jakal_net.py`
- `scan_cuda/scan_smoke.py`

Key capabilities visible in this directory:

- `causal_memory_scan_fused_native`
- `_scan_memory_batch_native_fused`
- `scan_backend`
- `scan_checkpoint_chunk_size`
- `causal_memory_scan_fused_trace`
- `causal_memory_scan_fused_checkpoints`
- `causal_memory_scan_fused_backward_cuda`
- `signed_entmax15` support in the native/backend path

This directory looks like the best candidate source for restoring:

- scan-level fused fast path
- checkpoint trace path
- backward CUDA glue
- related smoke/unit coverage

## What Is In `entmax`

This directory is the strongest candidate source for signed entmax integration.

Primary files:

- `entmax/causal_memory_lm.py`
- `entmax/native_backend.py`
- `entmax/jakal_net_native.cpp`
- `entmax/kernels.py`
- `entmax/propagation.py`
- `entmax/test_jakal_net.py`
- `entmax/test_native_backend.py`

Key capabilities visible in this directory:

- `signed_entmax15`
- `signed_entmax15_backward`
- route and propagation support for `signed_entmax15`
- scan path compatibility with `signed_entmax15`
- `causal_memory_scan_fused_backward_cuda` also appears here

This directory overlaps with `scan_cuda`. We should not merge both blindly. The likely clean order is:

1. recover scan-level fused path from `scan_cuda`
2. then layer the entmax-native pieces from `entmax`

## What Is In `curriculum`

Primary files:

- `curriculum/train_causal_memory_lm.py`
- `curriculum/train_causal_memory_lm.raw`

This looks like a focused training-script branch rather than a full architecture branch.

Use it as:

- reference for curriculum schedule changes
- argument or logging comparison

Do not treat it as the canonical source for model code.

## What Is In `20260419_sync`

Primary files:

- `20260419_sync/native_backend.remote.py`
- `20260419_sync/jakal_net_native.remote.cpp`
- `20260419_sync/jakal_net_native_cuda.remote.cu`
- `20260419_sync/jakal_net_native_cuda.remote.h`

This appears to be a small remote backup snapshot. It is useful for:

- diffing against current `src/jakal_net/native_backend.py`
- checking when a symbol existed in a remote working tree
- reconstructing missing registration glue

## What Is In `Jakal-Net-local-push`

This is the biggest subtree and looks like a full repo snapshot.

Useful contents include:

- `src/jakal_net/native_backend.py`
- `src/jakal_net/causal_memory_lm.py`
- `native/jakal_net_native.cpp`
- `native/jakal_net_native_cuda.cu`
- `tests/test_causal_memory_lm.py`
- `tests/test_native_backend.py`
- benchmarks and build scripts

This is valuable as a cross-check, but too noisy to use as a direct merge source.

## Recommended Merge Order

### Phase 1: Inventory Only

Commit this inventory file first so GitHub records that `remote_patch` exists and what is inside it.

### Phase 2: Recover Scan-Level Fused Path

Use `scan_cuda` as primary source:

- `causal_memory_lm_remote.py`
- `native_backend.py`
- `jakal_net_native.cpp`
- `jakal_net_native_cuda.cu`
- `jakal_net_native_cuda.h`

Goal:

- restore `scan_backend`
- restore `_scan_memory_batch_native_fused`
- restore `causal_memory_scan_fused_native`
- restore checkpoint trace path

### Phase 3: Restore/Compare Backward CUDA

Use:

- `scan_cuda/native_backend.py`
- `patch_scan_backward_cuda.py`
- `patch_jakal_chunk_backward.py`
- `patch_jakal_native_checkpoint_trace.py`

Goal:

- identify the stable backward CUDA path
- separate stable code from one-off migration edits

### Phase 4: Reapply Entmax

Use `entmax` as primary source:

- `entmax/kernels.py`
- `entmax/propagation.py`
- `entmax/native_backend.py`
- `entmax/jakal_net_native.cpp`

Goal:

- restore signed entmax route/edge support
- ensure it coexists cleanly with the fused scan code from Phase 2

### Phase 5: Reconcile Training Script

Use:

- `curriculum/train_causal_memory_lm.py`
- current repo `scripts/train_causal_memory_lm.py`

Goal:

- align CLI
- align curriculum
- align stage-level schedule
- avoid carrying stale model constructor arguments

## Immediate Next Actions

1. commit this inventory document
2. push it to GitHub
3. start a dedicated `scan_cuda` recovery commit, separate from entmax
4. only after scan recovery is clean, bring entmax in as a second commit

## Notes

- The main repo and current remote working tree do not currently expose the scan-level fused path.
- `remote_patch/scan_cuda` clearly contains that missing work.
- `remote_patch/entmax` also contains scan-related code, but it should be treated as a second integration source because it mixes two concerns: entmax and scan recovery.
