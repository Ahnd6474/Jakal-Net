# Jakal-Net

A PyTorch playground for latent-node propagation and routing.

This repository currently holds the core data structures and reference operators. The code is intentionally plain right now. The goal is to lock down module boundaries before the implementation gets faster or more specialized.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What is Jakal-Net?](#what-is-jakal-net)
- [Why this shape?](#why-this-shape)
- [API](#api)
  - [`Layer`](#layer)
  - [`LayerDelta`](#layerdelta)
  - [`Propagation`](#propagation)
  - [`SparsePropagation`](#sparsepropagation)
  - [`Transition`](#transition)
  - [`SparseTransition`](#sparsetransition)
  - [Helper modules](#helper-modules)
- [Examples](#examples)
- [Further reading](#further-reading)
- [License](#license)
- [Contributing](#contributing)

## Installation

Clone the repository, create a virtual environment, and install the pinned dependencies.

### PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` currently installs the CPU build of PyTorch. If you want a CUDA build, install the matching `torch` wheel first and then install the rest of the dependencies around it.

For DirectML on Windows, use a separate environment:

```powershell
py -m venv .venv-dml
.\.venv-dml\Scripts\Activate.ps1
python -m pip install torch-directml numpy
```

This repository now accepts `--device directml` in the smoke test and benchmark scripts.

This project is not packaged as an installable wheel yet, so examples and scripts expect:

```powershell
$env:PYTHONPATH = "src"
```

Or in bash:

```bash
export PYTHONPATH=src
```

## Quick start

If you just want to check that the environment is wired correctly, run the smoke test from the repository root.

### PowerShell

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

You should see:

```text
using device: cpu
smoke test passed
```

To try the Intel GPU through DirectML:

```powershell
$env:PYTHONPATH = "src"
.\.venv-dml\Scripts\python.exe scripts\smoke_test.py --device directml
```

To run the progressive B example through an actual next-token training loop:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\train_progressive_b_lm.py --steps 100 --seq-len 32 --dim 48
```

The training script uses a small built-in character corpus by default. You can point it at your own UTF-8 text file with `--text-file`.

Here is the smallest useful Python example:

```python
import torch

from jakal_net import DiagonalBilinearPairwise, Layer, Propagation

layer = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,))
layer = layer.with_tensors(
    state=torch.randn_like(layer.state),
    val=torch.randn_like(layer.val),
)

propagation = Propagation(pairwise_fn=DiagonalBilinearPairwise(dim=8))
delta = propagation(layer)

print(delta.delta_state.shape)
print(delta.delta_val.shape)
```

Expected output:

```text
torch.Size([2, 16])
torch.Size([2, 16, 8])
```

## What is Jakal-Net?

Jakal-Net is a small PyTorch codebase for experimenting with node-based latent workspaces. Each layer stores a scalar `state` per node and a vector `val` per node. Same-layer propagation and cross-layer transition are separate modules on purpose.

Right now the repository focuses on these low-level operators:

- `Layer` and `LayerDelta` for storage and updates
- `Propagation` and `SparsePropagation` for same-layer message passing
- `Transition` and `SparseTransition` for cross-layer routing
- A few lightweight scorer and router modules for wiring experiments together

What is not here yet:

- Flash kernels
- Encoders or decoders
- Real tokenizer integrations
- Large-scale dataset pipelines

## Why this shape?

The main split in this codebase is between propagation and transition.

Propagation is for same-layer message passing. It uses pairwise relations, compresses only the edge scores, and does not apply an activation to `state` before transport.

Transition is for moving information from one layer to another. It builds routing logits, applies a softmax over destination nodes, and then uses an activation over source `state` to scale outgoing transport. The default activation is `softplus`.

That distinction is not cosmetic. It keeps the dense reference path and the future optimized path aligned:

- Propagation can later become a flash-style tiled reduce without storing a full edge matrix.
- Transition can later become a fused routing-and-transport kernel without storing a full routing matrix.
- Sparse variants already share the same public surface area as the dense ones.

In short, the APIs are meant to stay stable while the kernels underneath get faster.

## API

### `Layer`

`Layer` is the storage unit. It does not perform computation by itself.

| Field | Shape | Meaning |
| --- | --- | --- |
| `state` | `[..., num_nodes]` | Scalar node state |
| `val` | `[..., num_nodes, dim]` | Vector node value |

The leading `...` dimensions are free. You can use them for batch, groups, heads, or any other fixed layout.

Useful methods:

- `Layer.zeros(dim, num_nodes, batch_shape=(), device=None, dtype=None)` creates a zero-initialized layer.
- `clone()` returns a deep copy.
- `with_tensors(state=None, val=None)` returns a new `Layer` with replaced tensors.
- `apply_delta(delta, merge_mode="add")` applies a `LayerDelta` by addition or replacement.

### `LayerDelta`

`LayerDelta` holds the update tensors that match a target layer:

- `delta_state`: `[..., num_nodes]`
- `delta_val`: `[..., num_nodes, dim]`

Use `LayerDelta.zeros_like(layer)` when you need an empty accumulator.

### `Propagation`

```python
Propagation(
    pairwise_fn,
    edge_compress_fn=torch.nn.functional.softsign,
    val_proj_fn=None,
    state_proj_fn=None,
    norm_fn=None,
    residual=True,
    return_delta=True,
)
```

`Propagation` performs same-layer message passing.

Behavior:

- Calls `pairwise_fn(layer.val, layer.val)` to produce pairwise scores shaped `[..., num_nodes, num_nodes]`
- Applies `edge_compress_fn` to those scores
- Projects `state` and `val` separately
- Reduces messages into a `LayerDelta`
- Returns that delta by default

Important detail: propagation does not apply an activation to `layer.state` before transport.

Implementation notes:

- `implementation="reference"` materializes the full pairwise relation for debugging and regression checks.
- `implementation="kernel"` uses vectorized fast paths for supported scorer types such as `DiagonalBilinearPairwise` and `BilinearPairwise`.
- `implementation="streaming"` processes target and source blocks incrementally to avoid storing the full edge matrix.
- `target_block_size` and `source_block_size` control the block shape in streaming mode.
- `accumulator_dtype` lets you keep accumulation in a wider dtype such as `torch.float32`.

On DirectML, supported scorer combinations automatically fall back to the kernel path when the generic streaming path is not a good fit.

If you set `return_delta=False`, the module applies the update to the input layer and returns a `Layer`. With `residual=True`, it adds the delta. With `residual=False`, it replaces the stored tensors.

### `SparsePropagation`

```python
SparsePropagation(
    pairwise_fn,
    sparse_type,
    edge_compress_fn=torch.nn.functional.softsign,
    topk=None,
    window=None,
    val_proj_fn=None,
    state_proj_fn=None,
    norm_fn=None,
    residual=True,
    return_delta=True,
)
```

`SparsePropagation` is the restricted version of `Propagation`.

Supported sparse modes:

- `sparse_type="window"` keeps a causal window `i - window <= j <= i`
- `sparse_type="topk"` keeps the highest-scoring source nodes per target

The return behavior matches `Propagation`.

- In `implementation="kernel"`, the `window` path uses an unfold-based fused fast path for supported pairwise scorers.
- In `implementation="streaming"`, the `window` path only evaluates the necessary neighborhood.
- In `implementation="kernel"`, the `topk` path uses a specialized vectorized score-and-top-k path for supported pairwise scorers.
- In `implementation="streaming"`, the `topk` path keeps a running top-k set per target block.

### `Transition`

```python
Transition(
    route_fn,
    norm_fn=None,
    state_activation_fn=torch.nn.functional.softplus,
    val_proj_fn=None,
    state_proj_fn=None,
    merge_mode="add",
)
```

`Transition` moves information from a source layer to a destination layer.

Behavior:

- Calls `route_fn(src_layer.val)` to produce routing logits shaped `[..., src_nodes, dst_nodes]`
- Applies `softmax` over the destination axis
- Applies `state_activation_fn(src_layer.state)` to scale outgoing transport
- Projects source `state` and `val`
- Accumulates into the destination layer

`Transition.forward(src_layer, dst_layer)` returns the updated destination `Layer`.

Notes:

- `merge_mode="add"` accumulates into the existing destination tensors
- `merge_mode="replace"` overwrites them with the transported result
- `val_proj_fn` must return `[..., src_nodes, dst_dim]`
- `implementation="kernel"` uses a dense vectorized transport path and currently works best with lightweight route functions such as `LinearRoute`
- `implementation="streaming"` processes source blocks incrementally so the full routing matrix is never stored
- `src_block_size` controls how many source nodes are routed per block in streaming mode
- `accumulator_dtype` controls the internal accumulation dtype

On DirectML, `SparseTransition` uses a compatibility fallback that avoids unsupported scatter patterns.

### `SparseTransition`

```python
SparseTransition(
    route_fn,
    topk,
    norm_fn=None,
    state_activation_fn=torch.nn.functional.softplus,
    val_proj_fn=None,
    state_proj_fn=None,
    merge_mode="add",
)
```

`SparseTransition` keeps only the highest-scoring destination nodes per source before the softmax step.

That gives you:

- dense routing semantics on the kept destinations
- fixed `topk` sparsity for larger cross-layer moves
- the same update behavior as `Transition`
- a vectorized `implementation="kernel"` path for dense-logit top-k routing
- scatter-based destination accumulation in streaming mode instead of a materialized dense routing matrix

### Helper modules

These helpers are exported from `jakal_net` and are meant to make experiments less repetitive.

| Export | Purpose |
| --- | --- |
| `ScalarAffine` | Scalar projection for `state` tensors |
| `DiagonalBilinearPairwise` | Cheap diagonal bilinear scorer |
| `BilinearPairwise` | Full bilinear scorer |
| `HadamardMLPPairwise` | Hadamard interaction followed by a small MLP |
| `LinearRoute` | One-layer router from source value to destination logits |
| `MLPRoute` | Two-layer router for richer routing logits |

## Examples

### Dense propagation

```python
import torch

from jakal_net import DiagonalBilinearPairwise, Layer, Propagation, ScalarAffine

layer = Layer.zeros(dim=8, num_nodes=16, batch_shape=(2,))
layer = layer.with_tensors(
    state=torch.randn_like(layer.state),
    val=torch.randn_like(layer.val),
)

op = Propagation(
    pairwise_fn=DiagonalBilinearPairwise(dim=8),
    state_proj_fn=ScalarAffine(),
)

delta = op(layer)
updated = layer.apply_delta(delta)
```

### Window sparse propagation

```python
import torch

from jakal_net import DiagonalBilinearPairwise, Layer, SparsePropagation

layer = Layer.zeros(dim=8, num_nodes=32, batch_shape=(1,))
layer = layer.with_tensors(
    state=torch.randn_like(layer.state),
    val=torch.randn_like(layer.val),
)

op = SparsePropagation(
    pairwise_fn=DiagonalBilinearPairwise(dim=8),
    sparse_type="window",
    window=3,
)

delta = op(layer)
```

### Top-k transition between layers

```python
import torch

from jakal_net import Layer, LinearRoute, SparseTransition

src = Layer.zeros(dim=16, num_nodes=32, batch_shape=(4,))
src = src.with_tensors(
    state=torch.randn_like(src.state),
    val=torch.randn_like(src.val),
)

dst = Layer.zeros(dim=16, num_nodes=8, batch_shape=(4,))

op = SparseTransition(
    route_fn=LinearRoute(src_dim=16, dst_nodes=8),
    topk=2,
)

updated_dst = op(src, dst)
```

### Progressive B example LM

The progressive B model is kept as example code in [`scripts/progressive_b_example.py`](./scripts/progressive_b_example.py), not as part of the public `jakal_net` package surface.

That example follows this flow:

- `Embedding + position -> init(s, v)`
- `S warmup` with window sparse propagation
- `joint blocks` that expand, compress, and progressively gate the B path
- `final S refine`
- `prediction slot -> LM head`

The B path is still built entirely from the existing public modules:

- `SparsePropagation` for the S path
- `Transition` or `SparseTransition` for B expansion, compression, and S/B exchange
- `DiagonalBilinearPairwise` and `LinearRoute` as the default scorers and routers

If you want to train the example instead of only running a forward pass, the helper code also lives in [`scripts/progressive_b_example.py`](./scripts/progressive_b_example.py):

- `build_char_vocab`
- `sample_next_token_batch`
- `train_next_token_model`
- `generate_next_tokens`

## Further reading

If you want to inspect the reference implementation directly, start here:

- [`src/jakal_net/core.py`](./src/jakal_net/core.py)
- [`src/jakal_net/propagation.py`](./src/jakal_net/propagation.py)
- [`src/jakal_net/transition.py`](./src/jakal_net/transition.py)
- [`tests/test_jakal_net.py`](./tests/test_jakal_net.py)

The tests are small, but they are a good map of the intended behavior.

For timing and memory checks, run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\benchmark_operators.py --warmup 1 --iterations 5
```

That script compares `reference`, `kernel`, and `streaming` implementations for:

- dense propagation
- window sparse propagation
- top-k sparse propagation
- dense transition
- top-k sparse transition

You can run the same benchmark on DirectML like this:

```powershell
$env:PYTHONPATH = "src"
.\.venv-dml\Scripts\python.exe scripts\benchmark_operators.py --device directml --warmup 1 --iterations 5
```

## License

MIT-style usage terms do not apply here. This repository is distributed under Apache 2.0. See [LICENSE](./LICENSE) for the full text.

## Contributing

If you change the core operators, run the checks before you push anything:

### PowerShell

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\smoke_test.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected output:

```text
smoke test passed
..............
----------------------------------------------------------------------
Ran 14 tests in 0.128s

OK
```

There is no formatter or linter wired into this repository yet, so for now the test suite is the main safety check.
