# Jakal-Net

Jakal-Net is a PyTorch research playground for latent-node propagation,
sparse routing, and the Progressive-B language-model experiments built on top
of those operators.

The current codebase has two layers:

- `src/jakal_net/`: reusable operator primitives (`Layer`, propagation,
  transition, sparse routing, native backend dispatch).
- `scripts/`: experiment code, including the Progressive-B example LM,
  corpus builders, training loops, checkpointing, TensorBoard logging, and
  native extension build helpers.

## Architecture

![Jakal-Net Progressive-B architecture](docs/architecture.svg)

At a high level, the example LM encodes a fixed-length prefix into an `S`
sequence workspace, repeatedly exchanges information with a compressed/expanded
`B` bottleneck workspace, then reads from the final `S` state.

For response training, the final `S` slots condition a small GRU response
decoder:

```text
prefix ids -> S/B encoder -> response slots -> GRU decoder -> MLP head -> token logits
```

For classic next-token training, the final prediction slot goes directly to
the LM head.

## Current Capabilities

- Same-layer node message passing with `Propagation` and `SparsePropagation`.
- Cross-layer information movement with `Transition` and `SparseTransition`.
- Dense, window, top-k, and query-top-k execution paths.
- Reference, streaming, kernel, and native backend implementations.
- Native C++/CUDA extension support for selected propagation and transition
  kernels.
- Progressive-B example LM with warmup `S` propagation, lite/mid/full B stages,
  final `S` refinement, and optional prefix-response decoding.
- Training objectives:
  `last_token`, `teacher_forcing`, `full_sequence_causal`,
  `prefix_response`, and `next_sentence_response`.
- Subword tokenization through SentencePiece with dialogue special tokens for
  response objectives.
- TensorBoard logging, epoch-based eval, resumable-style checkpoint payloads,
  and best/last/final checkpoint artifacts.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/jakal_net/core.py` | `Layer`, `LayerDelta`, block helpers, validation |
| `src/jakal_net/propagation.py` | Dense and sparse same-layer propagation |
| `src/jakal_net/transition.py` | Dense and sparse cross-layer routing |
| `src/jakal_net/modules.py` | Pairwise scorers, route modules, position encoding |
| `src/jakal_net/native_backend.py` | Native extension discovery and dispatch |
| `native/` | C++/CUDA extension source |
| `scripts/progressive_b_example.py` | Progressive-B model and training utilities |
| `scripts/train_progressive_b_lm.py` | CLI training entry point |
| `scripts/build_mixed_next_sentence_corpus.py` | Mixed dialogue/science/wiki/code pair corpus builder |
| `tests/` | Unit tests and native backend checks |
| `docs/architecture.svg` | Architecture diagram used by this README |

## Installation

Create a virtual environment and install dependencies.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

For CUDA, install the matching PyTorch CUDA wheel for the target system before
running GPU experiments. The native extension build requires a working compiler
toolchain and CUDA toolkit when CUDA kernels are enabled.

## Quick Checks

Run the smoke test:

```bash
PYTHONPATH=src python scripts/smoke_test.py --device cpu
```

Run unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Build the native extension:

```bash
PYTHONPATH=src python scripts/build_native_extension.py
```

Inspect native backend support:

```bash
PYTHONPATH=src python - <<'PY'
from jakal_net.native_backend import native_status
print(native_status())
PY
```

## Training Examples

Small next-token smoke run:

```bash
PYTHONPATH=src python scripts/train_progressive_b_lm.py \
  --device cuda \
  --tokenizer subword \
  --subword-vocab-size 1024 \
  --seq-len 128 \
  --steps 100 \
  --batch-size 16 \
  --dim 128 \
  --tensorboard \
  --run-name smoke_next_token
```

Prefix/response dialogue-style run:

```bash
PYTHONPATH=src python scripts/train_progressive_b_lm.py \
  --device cuda \
  --training-objective next_sentence_response \
  --jsonl-source artifacts/data/mixed_next_sentence_dialogue_science_wiki_code.jsonl \
  --tokenizer subword \
  --tokenizer-prefix artifacts/tokenizers/en_dialogue_subword_4096 \
  --subword-vocab-size 4096 \
  --seq-len 512 \
  --response-len 128 \
  --epochs 5 \
  --eval-every-epoch \
  --batch-size 64 \
  --dim 384 \
  --warmup-layers 2 \
  --lite-layers 5 \
  --mid-layers 5 \
  --full-layers 0 \
  --final-refine-layers 3 \
  --route-topk 32 \
  --route-mode topk \
  --route-kind low_rank_bilinear \
  --route-hidden-dim 64 \
  --pairwise-kind low_rank_bilinear \
  --pairwise-hidden-dim 64 \
  --sequence-propagation window \
  --expanded-propagation topk \
  --compressed-propagation topk \
  --s-window 64 \
  --value-norm-kind layernorm \
  --norm-position pre \
  --edge-dropout-p 0.1 \
  --learning-rate 0.0002 \
  --weight-decay 0.01 \
  --precision bf16 \
  --implementation native \
  --tensorboard \
  --save-checkpoint
```

The training script writes:

- TensorBoard events under `artifacts/tensorboard/`.
- Run summaries, metrics, samples, and checkpoints under
  `artifacts/training_runs/`.
- Best and last checkpoints under `custom/checkpoints/` when
  `--save-checkpoint` is enabled.

Launch TensorBoard against the current artifact root:

```bash
tensorboard --logdir artifacts/tensorboard
```

## Operator Model

`Layer` is the base storage object:

| Field | Shape | Meaning |
| --- | --- | --- |
| `state` | `[..., num_nodes]` | Scalar node activation/state |
| `val` | `[..., num_nodes, dim]` | Vector node value |

`LayerDelta` stores matching updates:

| Field | Shape |
| --- | --- |
| `delta_state` | `[..., num_nodes]` |
| `delta_val` | `[..., num_nodes, dim]` |

`Propagation` performs same-layer message passing. It scores node pairs,
compresses edge scores, transports state/value projections, and returns a
`LayerDelta` or updated `Layer`.

`SparsePropagation` restricts same-layer message passing to a causal window or
top-k source set.

`Transition` moves information from a source layer to a destination layer by
building route logits and reducing transported source values into destination
nodes.

`SparseTransition` keeps only top-k destination routes per source before
transport. It supports edge dropout and usage-aware dropout in the experimental
paths used by Progressive-B.

## Progressive-B Example LM

The example LM in `scripts/progressive_b_example.py` is intentionally kept
outside the package surface. It composes the public operators into a staged
encoder:

1. Token embedding plus learned position encoding initializes the `S` layer.
2. `S` warmup applies window sparse propagation.
3. Each Progressive-B joint block updates `S`, expands compressed `B` memory,
   propagates through expanded `B`, sends information back to `S`, recompresses
   `B`, and optionally lets `S` update compressed `B`.
4. Final `S` refinement prepares readout slots.
5. The LM head reads the prediction slot for next-token objectives.
6. The response path reads multiple `S` slots, adds decoder-token embeddings,
   runs a GRU decoder, then projects through an MLP response head.

The default large response configuration uses low-rank bilinear pairwise
scorers and routers with `route_topk=32`, `seq_len=512`, and `response_len=128`.

## Corpus Paths

The training CLI can read:

- Plain text files with `--text-file`.
- Repeated text files, directories, or globs with `--text-source`.
- JSONL text records with `--jsonl-source --jsonl-text-key`.
- Hugging Face datasets with `--hf-dataset`.
- Explicit dialogue/prefix-response records with `prefix` and `response`
  fields.

`next_sentence_response` first converts source text or explicit JSONL records
into prefix/response pairs, then trains the response decoder with masked
cross-entropy over the response tokens.

## Development Notes

- Keep generated artifacts out of commits unless the artifact is intentionally
  part of a regression fixture.
- Prefer `artifacts/tensorboard/<run-name>` for TensorBoard runs and archive old
  runs outside the active logdir when the UI gets noisy.
- Run smoke tests and unit tests before pushing operator changes.
- If native backend behavior changes, update both `native/` and
  `tests/test_native_backend.py`.

## License

This repository is distributed under Apache 2.0. See [LICENSE](./LICENSE) for
the full text.
