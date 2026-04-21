# Jakal-Net

Jakal-Net is a PyTorch research playground for latent-node propagation,
sparse routing, and the Progressive-B language-model experiments built on top
of those operators.

The codebase is split into two layers:

- `src/jakal_net/`: reusable operator primitives such as `Layer`,
  propagation, transition, sparse routing, and native backend dispatch.
- `scripts/`: experiment code for Progressive-B, corpus builders, training
  loops, checkpointing, TensorBoard logging, and native extension build
  helpers.

## Architecture

![Jakal-Net causal memory architecture](docs/architecture.svg)

The diagram highlights the current causal-memory training path. The older
Progressive-B/query-block path remains below because it uses the same
`Layer` / `Propagation` / `Transition` operator family.

Progressive-B is not a standard decoder-only LM stack. It encodes a fixed
prefix into a sequence workspace `S`, repeatedly exchanges information with a
compressed and expanded latent memory workspace `B`, then predicts future
tokens through a separate query path.

At a high level, the current query-block path is:

```text
prefix ids
  -> token/position embedding
  -> S warmup
  -> Progressive-B joint blocks over S and B
  -> final S refinement
  -> query slots
  -> query transition
  -> query propagation
  -> query head
  -> token logits
```

There is no legacy LM head and no GRU response decoder in the current model.
The vocabulary projection belongs to the query head, so trainable parameters
track the query-based objective directly.

## Core Architecture

### 1. Sequence workspace `S`

`S` is the token-aligned workspace. Each token position owns:

- a scalar state in `Layer.state`
- a vector value in `Layer.val`

`S` starts from token embeddings plus learned positional encodings, then runs
window or dense same-layer propagation. This is the part of the model that
most closely resembles a conventional sequence encoder, except that it is
expressed with the general `Layer` / `Propagation` operator API instead of a
Transformer block API.

### 2. Bottleneck workspace `B`

`B` is the latent memory system. Each Progressive-B joint block creates or
updates two `B` views:

- expanded `B`: wider latent workspace used for richer internal propagation
- compressed `B`: smaller bottleneck workspace used to carry summarized state
  across blocks

Inside each joint block, information moves through:

1. same-layer propagation on `S`
2. `S -> expanded B` routing
3. same-layer propagation inside expanded `B`
4. `expanded B -> S` routing
5. `expanded B -> compressed B` compression
6. optional `S -> compressed B` update
7. same-layer propagation inside compressed `B`

That is the central architectural idea of this repository: sequence processing
is mediated by a reusable latent memory workspace instead of relying only on a
single token-aligned stream.

### 3. Query prediction path

Prediction does not read directly from the last token state. Instead, the model
creates a dedicated query workspace:

- one query slot per future token position
- `query_transition` routes information from encoded `S` into those slots
- `query_propagation` updates the slots causally from left to right
- `query_head` maps final query slot values to vocabulary logits

This makes the forecasting path explicit. The encoded prefix lives in `S`,
while the prediction process lives in a separate query layer with its own
causal propagation.

For `query_block` training, the current head uses teacher-forced token feedback
before query refinement:

- build one query slot per target position
- inject shifted target-token embeddings into those query slots during training
- run `query_transition` and causal `query_propagation`
- read logits from the refined query slots

At inference time there is no gold feedback, so the same path falls back to
self-conditioned soft token feedback.

### 4. Execution backends

The same architecture can run through several implementations:

- `reference`
- `streaming`
- `kernel`
- `native`

The backend choice changes execution strategy, not the high-level model graph.

## Compared With Existing LM Architectures

The most useful comparison point is a standard decoder-only Transformer LM.

| Aspect | Standard decoder-only LM | Jakal-Net Progressive-B |
| --- | --- | --- |
| Main working space | Single token-aligned hidden sequence | Token-aligned `S` plus latent `B` workspaces |
| Cross-token interaction | Self-attention over the same sequence | `Propagation` on `S`, plus routed interaction through expanded and compressed `B` |
| Long-range summarization | Implicit inside attention layers and residual stream | Explicit latent memory bottleneck carried across joint blocks |
| Prediction path | Read logits from the final token stream | Build dedicated query slots, route `S` into them, then propagate queries causally |
| Architectural unit | Attention + MLP block | General propagation + transition operators composed into joint `S/B` blocks |
| Sparsity control | Usually attention mask or custom sparse attention kernels | Window, top-k, query-top-k, dense, and routed sparse transitions |
| Execution abstraction | Usually tied closely to one attention implementation | Same operator graph can dispatch to reference, streaming, kernel, or native backends |

Another way to say it:

- A standard decoder-only LM keeps one residual stream and predicts from that
  stream directly.
- Progressive-B separates encoding, latent memory exchange, and prediction into
  different workspaces.
- The architectural bet is that explicit routed latent memory may scale
  differently from a pure token-only stack and can support more flexible sparse
  execution paths.

This repository is therefore closer to an operator research platform for
alternative LM structure than to a conventional Transformer implementation.

## Current Capabilities

- Same-layer node message passing with `Propagation` and `SparsePropagation`.
- Cross-layer information movement with `Transition` and `SparseTransition`.
- Dense, window, top-k, and query-top-k execution paths.
- Reference, streaming, kernel, and native backend implementations.
- Native C++/CUDA extension support for selected propagation and transition
  kernels.
- Progressive-B example LM with warmup `S` propagation, lite/mid/full `B`
  stages, final `S` refinement, query transition, query propagation, and
  query-only vocab prediction.
- Experimental causal hierarchical-memory LM with full causal `S` backbone,
  sequential `B` scan, and a separate causal prediction head stack.
- Training objectives: `query_next_token` and `query_block`.
- Byte-level BPE tokenization, including larger vocabularies for code and
  LaTeX-heavy corpora.
- Pretokenized token-stream loading/saving for the causal-memory training path.
- TensorBoard logging focused on total minibatch loss and overlaid
  distance-bucket perplexity curves.
- Query-block specific logging for unweighted average perplexity, grad norm,
  eval samples, and distance-bucket perplexity.
- Best/last/final checkpoint artifacts for experiment tracking.

## Current Query-Block Training Recipe

The current scratch recipe used for the teacher-forced query-feedback head is:

```bash
PYTHONPATH=src python -u scripts/train_progressive_b_lm.py \
  --device cuda \
  --training-objective query_block \
  --jsonl-source artifacts/data/query_block_instruction_dialogue_arxiv_pubmed_code_wiki_10m.jsonl \
  --balance-batch-by-source \
  --tokenizer byte_bpe \
  --subword-vocab-size 16384 \
  --tokenizer-prefix artifacts/tokenizers/query_block_mix_byte_bpe_16384 \
  --epochs 1.0 \
  --eval-interval 100 \
  --checkpoint-interval 1000 \
  --batch-size 64 \
  --grad-accum-steps 10 \
  --seq-len 512 \
  --target-len 192 \
  --dim 512 \
  --warmup-layers 0 \
  --final-refine-layers 1 \
  --query-refine-layers 4 \
  --lite-layers 2 \
  --mid-layers 5 \
  --full-layers 2 \
  --route-topk 16 \
  --query-topk 16 \
  --route-kind low_rank_bilinear \
  --pairwise-kind low_rank_bilinear \
  --route-mode topk \
  --expanded-propagation topk \
  --compressed-propagation topk \
  --sequence-propagation window \
  --precision bf16 \
  --implementation kernel \
  --s-window 32 \
  --edge-dropout-p 0.1 \
  --learning-rate-schedule cosine \
  --learning-rate-warmup-steps 1000 \
  --learning-rate-warmup-start 1e-4 \
  --learning-rate 5e-4 \
  --learning-rate-min-ratio 0.5 \
  --query-block-front-weight 1.0 \
  --block-residual \
  --query-residual \
  --share-route-families \
  --tensorboard \
  --save-checkpoint
```

On the current 95 GB GPU target, a random-token forward/backward/step check
passes at `batch_size=64` and OOMs at `batch_size=72`, so `64` is the safe
single-step batch ceiling for this configuration.

## Causal Hierarchical Memory LM

There is now a second experimental training path alongside Progressive-B:
`scripts/train_causal_memory_lm.py`.

This path keeps the existing operator philosophy (`Propagation` for same-layer
refinement, `Transition` for cross-layer movement) but changes the model graph
to match a standard causal next-token objective.

At a high level:

```text
document
  -> chunk into max_seq_len windows
  -> prepend <|bos|> + mode on the first chunk
  -> prepend <|cont|> + mode on every later chunk
  -> full causal S backbone inside each chunk
  -> per-time-step B scan inside each chunk
  -> carry B_final across chunks in the same document
  -> reset B only at document boundaries
  -> multi-level B read + aligned S residual
  -> causal prediction propagation
  -> LM head
```

The roles are intentionally separated:

- `S`: chunk-local token-aligned causal backbone
- `B`: document-persistent latent memory, updated one token step at a time
- prediction stack: causal decoder head over the readout sequence

The default memory recipe matches the current design discussion:

- `max_seq_len=2048`
- `dim=512`
- `S` layers: `2`
- `B` slots: `512 -> 128 -> 32`
- prediction layers: `2`

The current implementation updates memory in this order for each visible token:

1. read the aligned causal `S` state for the current position
2. write into `B^0`
3. propagate inside each `B` level
4. compress upward through `B^0 -> B^1 -> B^2`
5. apply weak gated skip updates (`S -> B^1`, `B^0 -> B^2`)
6. read every `B` level into one prediction vector
7. combine that read with the aligned `S` residual
8. run a separate causal prediction stack before the LM head

This keeps the causal invariant explicit: the logits for position `t+1` are
built only from the prefix snapshot available after consuming position `t`.

For training, chunk boundaries carry state but not gradient history: the
current script carries `B` across chunks within a document, then detaches the
memory state between chunk steps in the same way truncated BPTT would.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/jakal_net/core.py` | `Layer`, `LayerDelta`, block helpers, validation |
| `src/jakal_net/causal_memory_lm.py` | Experimental causal hierarchical-memory LM |
| `src/jakal_net/propagation.py` | Dense and sparse same-layer propagation |
| `src/jakal_net/transition.py` | Dense and sparse cross-layer routing |
| `src/jakal_net/modules.py` | Pairwise scorers, route modules, position encoding |
| `src/jakal_net/native_backend.py` | Native extension discovery and dispatch |
| `native/` | C++/CUDA extension source |
| `scripts/progressive_b_example.py` | Progressive-B model and training utilities |
| `scripts/train_progressive_b_lm.py` | CLI training entry point |
| `scripts/train_causal_memory_lm.py` | Separate causal-memory LM training entry point |
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

Small query-block smoke run:

```bash
PYTHONPATH=src python scripts/train_progressive_b_lm.py \
  --device cuda \
  --training-objective query_block \
  --tokenizer byte_bpe \
  --subword-vocab-size 4096 \
  --seq-len 128 \
  --target-len 32 \
  --steps 100 \
  --batch-size 16 \
  --dim 128 \
  --tensorboard \
  --run-name smoke_next_token
```

Mixed corpus query-block run with a larger byte BPE vocabulary:

```bash
PYTHONPATH=src python scripts/train_progressive_b_lm.py \
  --device cuda \
  --training-objective query_block \
  --jsonl-source artifacts/data/mixed_next_sentence_dialogue_science_wiki_code.jsonl \
  --tokenizer byte_bpe \
  --tokenizer-prefix artifacts/tokenizers/mixed_next_sentence_byte_bpe_16384 \
  --subword-vocab-size 16384 \
  --seq-len 512 \
  --target-len 128 \
  --steps 100000 \
  --eval-interval 1000 \
  --eval-steps 16 \
  --checkpoint-interval 1000 \
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
  --norm-position pre \
  --edge-dropout-p 0.1 \
  --balance-batch-by-source \
  --data-workers 8 \
  --prefetch-factor 4 \
  --pretokenize-workers 32 \
  --b-diversity-loss-weight 0.02 \
  --b-cosine-margin 0.20 \
  --route-concentration-loss-weight 0.05 \
  --route-load-cap 0.20 \
  --edge-prob-cap 0.55 \
  --precision bf16 \
  --implementation native \
  --tensorboard \
  --save-checkpoint
```

Reduced-data causal-memory bootstrap with a pretrained Qwen tokenizer and
embedding:

```bash
PYTHONPATH=src:scripts python scripts/train_causal_memory_lm.py \
  --device cuda \
  --jsonl-source artifacts/data_qwen_small/plain_dialogue_20k.jsonl \
  --tokenizer hf_auto \
  --hf-tokenizer-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --hf-embedding-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --dim 896 \
  --seq-len 512 \
  --s-layers 6 \
  --prediction-layers 3 \
  --memory-slots 384 96 24 \
  --memory-topk 24 \
  --pairwise-rank 96 \
  --route-rank 64 \
  --pairwise-kind low_rank_bilinear \
  --route-kind low_rank_bilinear \
  --implementation native \
  --scan-backend native \
  --optimizer adamw_fused \
  --batch-size 3 \
  --stage1-batch-size 3 \
  --stage2-batch-size 3 \
  --stage3-batch-size 3 \
  --curriculum-stage1-span 1 \
  --curriculum-stage2-span 2 \
  --curriculum-stage3-span 4 \
  --embedding-lr-mult 0.1 \
  --rnn-pretrain-steps 0 \
  --eval-interval 200 \
  --eval-documents 4 \
  --tensorboard \
  --run-name qwen896_small_v1
```

Notes for the Qwen path:

- `hf_auto` uses the upstream Hugging Face tokenizer directly instead of the
  repository byte-BPE tokenizer.
- No extra special tokens are added. Dialogue structure is rendered through the
  existing Qwen chat markers and plain-text section labels.
- `Qwen/Qwen2.5-Coder-0.5B-Instruct` is currently the practical pretrained
  tokenizer/embedding source for this codebase. Larger Qwen hidden sizes such
  as `1536+` were measured as too slow or OOM-prone in the current native scan
  implementation.
- The currently validated `qwen896_a` causal-memory configuration is:
  - params: about `143.6M`
  - `dim=896`
  - `s_layers=6`
  - `prediction_layers=3`
  - `memory_slots=[384, 96, 24]`
  - `pairwise_rank=96`
  - `route_rank=64`
  - safe bootstrap batch: `3`

Default document-chunked causal-memory run:

`scripts/train_causal_memory_lm.py` is pinned to `byte_bpe` for the document-chunked causal-memory path.

```bash
PYTHONPATH=src python scripts/train_causal_memory_lm.py \
  --device cuda \
  --jsonl-source artifacts/data/query_block_instruction_dialogue_arxiv_pubmed_code_wiki_10m.jsonl \
  --tokenizer byte_bpe \
  --subword-vocab-size 16384 \
  --tokenizer-prefix artifacts/tokenizers/causal_memory_byte_bpe_16384 \
  --seq-len 2048 \
  --dim 512 \
  --s-layers 2 \
  --memory-slots 512 128 32 \
  --prediction-layers 2 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --learning-rate 3e-4 \
  --warmup-steps 200 \
  --pretokenize-workers 8 \
  --tensorboard
```

The causal-memory script is intentionally separate from
`scripts/train_progressive_b_lm.py`. That keeps the older Progressive-B runs
reproducible while allowing the new architecture to evolve independently.
Its default training unit is the document, not a flat token stream.

The training script writes:

- TensorBoard events under each run directory
- run summaries, metrics, samples, and checkpoints under
  `artifacts/training_runs/`
- best and last checkpoints under `custom/checkpoints/` when
  `--save-checkpoint` is enabled

The causal-memory script writes a parallel run directory with:

- `config.json`
- `history.jsonl` and `history.csv`
- TensorBoard events when `--tensorboard` is enabled
- `checkpoints/best.pt`, `checkpoints/last.pt`, and interval checkpoints
- optional pretokenized token bundles when `--save-pretokenized` is used

### Causal-Memory Document Tokens

The document-chunked path uses explicit structural tokens and treats them as
normal next-token targets.

Minimum required structural set:

- `<|bos|>`: first chunk of a document
- `<|cont|>`: every continuation chunk of the same document
- `<|eos|>`: document end
- `<|text|>`, `<|code|>`, `<|dialogue|>`, `<|instruction|>`: document mode

Dialogue and instruction documents also use role or response structure tokens:

- `<|user|>`
- `<|assistant|>`
- `<|eot|>`
- `<|response|>`

The current causal-memory script treats `<|cont|>` as mandatory on every
non-first chunk. The first chunk uses `<|bos|> + mode`, and all later chunks
use `<|cont|> + mode`.

### Query-Block Optimization Controls

Recent query-block experiments use two extra optimization controls that are
useful when the model shows intermittent loss and gradient spikes.

- `--query-block-front-weight` applies an exponential position weight over the
  supervised query-block targets. Position `0` remains a structural
  `query_block_start` slot with zero loss weight. Positions `1..target_len`
  receive weights that decay from `front_weight` down to `1.0`, so the first
  predicted tokens matter more than later tokens while still optimizing the
  whole block.
- `--warmup-delta-scale` controls the residual gain on the `s_warmup` path.
  The current default is `0.0`, which keeps the warmup operators in the
  forward structure but removes their residual add from the learned sequence
  trunk.
- `--s-delta-scale` controls the residual gain on the sequence warmup and
  same-sequence propagation path. In current Progressive-B query-block runs,
  this has a large effect on stability because the shared `S` trunk feeds every
  later joint block and query transition.
- `--freeze-position-encoding` is a diagnostic switch that disables learning on
  the learned positional encoder. It is useful for separating positional
  instability from broader sequence-trunk instability, but it should be treated
  as an experiment flag rather than the default training recipe.

In practical terms:

- use `--query-block-front-weight 1.0` to disable front weighting
- use values such as `2.0` to `4.0` when early-token quality matters more than
  later-token quality
- use `--warmup-delta-scale 0.0` to disable the `s_warmup` residual path while
  keeping the rest of the model unchanged
- use `--s-delta-scale 0.1` as a safer default when `query_block` training
  shows repeated sequence-side gradient explosions; `0.25` is noticeably more
  aggressive on the current 512-dim setup
- use `--freeze-position-encoding` only as a debugging control when you need to
  check whether positional learning is the primary source of a spike

These controls can be combined with the existing global grad norm clip,
learning-rate warmup, and cosine decay schedule.

For post-run spike inspection, use `scripts/debug_progressive_b_events.py` to
scan a TensorBoard event file and print the dominant grad tags over a step
window.

Launch TensorBoard against the current artifact root:

```bash
tensorboard --logdir artifacts/training_runs
```

## Losses and Prediction Head

`query_block` predicts a block of future tokens from a prefix. It builds query
slots from the encoded `S` workspace, lets those slots receive routed
information from `S`, then updates query slots left-to-right through query
propagation. The query head maps the final query slot values to vocabulary
logits.

The main objective is cross entropy over the query block. For monitoring, the
same logits are also split into four distance buckets so TensorBoard can show
short-, mid-, and long-distance perplexity on one plot.

Two auxiliary losses can be enabled:

- `--b-diversity-loss-weight` penalizes compressed `B` nodes whose cosine
  similarity exceeds `--b-cosine-margin`.
- `--route-concentration-loss-weight` penalizes route or edge concentration
  only when destination load or single-edge probability exceeds configured
  caps. It is not a uniform-routing loss; it specifically discourages collapse
  onto one node.

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
transport. It supports edge dropout and usage-aware dropout in the
experimental paths used by Progressive-B.

## Progressive-B Example LM

The example LM in `scripts/progressive_b_example.py` is intentionally kept
outside the package surface. It composes the public operators into a staged
encoder:

1. Token embedding plus learned position encoding initializes the `S` layer.
2. `S` warmup applies window sparse propagation.
3. Each Progressive-B joint block updates `S`, expands compressed `B` memory,
   propagates through expanded `B`, sends information back to `S`, recompresses
   `B`, and optionally lets `S` update compressed `B`.
4. Final `S` refinement prepares the encoded prefix state.
5. Query slots are initialized for future-token positions.
6. Query transition routes `S` information into those slots.
7. Query propagation updates the slots left-to-right.
8. The query head projects query slot values to vocabulary logits.

The current large query-block configurations use bilinear, Hadamard-MLP, or
additive-style pairwise/route modules depending on the experiment, along with
byte BPE vocabularies sized for mixed dialogue, science, wiki, code, and
LaTeX-heavy text.

## Corpus Paths

The training CLI can read:

- plain text files with `--text-file`
- repeated text files, directories, or globs with `--text-source`
- JSONL text records with `--jsonl-source --jsonl-text-key`
- Hugging Face datasets with `--hf-dataset`
- explicit dialogue or prefix/response records with `prefix` and `response`
  fields

`next_sentence_response` first converts source text or explicit JSONL records
into prefix/response pairs, then trains the response decoder with masked
cross-entropy over the response tokens.

## Development Notes

- Keep generated artifacts out of commits unless the artifact is intentionally
  part of a regression fixture.
- Prefer `artifacts/tensorboard/<run-name>` for TensorBoard runs and archive
  old runs outside the active logdir when the UI gets noisy.
- Run smoke tests and unit tests before pushing operator changes.
- If native backend behavior changes, update both `native/` and
  `tests/test_native_backend.py`.

## License

This repository is distributed under Apache 2.0. See [LICENSE](./LICENSE) for
the full text.
