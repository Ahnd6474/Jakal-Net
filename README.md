# Jakal-Net

Jakal-Net is a PyTorch research repository for adding a shared, inner-product
knowledge module to a causal Transformer language model.

The current model is `CausalMemoryLM`. Older Progressive-B, propagation-stack,
and hierarchical B-memory designs are still visible in Git history and some
archived experiment files, but they are not the architecture implemented by
the current source tree.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Current architecture](#current-architecture)
- [Recurrent trace](#recurrent-trace)
- [Model configuration](#model-configuration)
- [Training](#training)
- [Current experiment status](#current-experiment-status)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [License](#license)

## Installation

Python 3.10 or newer is recommended.

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

Install a CUDA-enabled PyTorch build separately when training on NVIDIA GPUs.
The pinned base requirement is intended for the repository's existing
environment and may need to be replaced with the wheel matching the local CUDA
toolkit.

## Quick start

The library API accepts token IDs with shape `[batch, sequence]` and returns
causal next-token logits.

```python
import torch

from jakal_net import CausalMemoryLM, MemoryScanOutput

model = CausalMemoryLM(
    vocab_size=16_384,
    dim=512,
    max_seq_len=512,
    transformer_layers=6,
    transformer_heads=8,
    knowledge_memory_size=4_096,
    knowledge_hops=1,
)

input_ids = torch.randint(0, 16_384, (2, 128))
output = model(
    input_ids,
    return_memory_state=True,
)

assert isinstance(output, MemoryScanOutput)
print(output.logits.shape)          # torch.Size([2, 128, 16384])
print(output.memory_state[0].shape) # torch.Size([2, 6, 8, 4096])
```

Pass `output.memory_state` into the next chunk from the same document. Use
`reset_mask` to clear selected batch rows at document boundaries.

```python
next_ids = torch.randint(0, 16_384, (2, 128))
next_output = model(
    next_ids,
    memory_state=output.memory_state,
    reset_mask=torch.tensor([False, True]),
    return_memory_state=True,
)
```

## Current architecture

Each block is a pre-norm causal Transformer block with a knowledge update
inserted between self-attention and the feed-forward network.

```text
hidden
  -> layer norm
  -> causal multi-head self-attention
  -> shared knowledge lookup and attention mixing
  -> shared relation propagation
  -> layer-specific knowledge readout
  -> gated residual update
  -> feed-forward network
```

Let:

- `B` be batch size
- `T` be sequence length
- `D` be Transformer width
- `H` be the number of attention heads
- `K` be the number of knowledge slots

`K` in this document means knowledge-slot count. It is not the Key tensor from
the usual attention notation `Q`, `K`, and `V`.

### 1. Shared knowledge lookup

Every Transformer layer has its own attention projection weights. The model
extracts that layer's Query projection:

```text
Q_l = LayerNorm(H_l) W_Q,l                    [B, T, D]
```

It compares each query with one shared table of knowledge vectors:

```text
M                                                  [K, D]
A_l = Q_l M^T                                 [B, T, K]
```

`M` is shared by all Transformer layers. With
`knowledge_memory_size=4096`, it contains 4,096 learned vectors.

### 2. Mixing with Transformer attention

The lookup activations are propagated across tokens with the causal
self-attention probabilities from the same layer:

```text
P_l,h                                           [B, T, T]
S_l,h = P_l,h A_l                              [B, T, K]
```

The model mixes direct lookup, attention-shared activation, and the trace
carried from the previous document chunk:

```text
T_l,h = (1 - alpha_l,h) A_l
        + alpha_l,h S_l,h
        + gamma_l,h C_l,h

T_l = mean_h(T_l,h)                            [B, T, K]
```

`alpha` and `gamma` are learned per layer and per head. Their logits start at
`-2` and `-4`, respectively, so the initial sigmoid values are approximately
`0.119` and `0.018`.

### 3. Shared relation propagation

All layers and hops share one dense relation matrix:

```text
E                                                  [K, K]
Z_l = activation(T_l E)                        [B, T, K]
```

For `knowledge_hops > 1`, the model applies the same `E` again and adds each
extra hop through a learned residual gate.

### 4. Layer-specific readout

The lookup table and relation matrix are shared, but each layer has its own
readout matrix:

```text
V_l                                                [K, D]
delta_l = Z_l V_l                              [B, T, D]
```

A learned scalar controls the knowledge residual in each layer:

```text
g_l = sigmoid(a_l)
H'_l = H_attention,l + g_l delta_l
```

Each `a_l` is initialized to zero, so every layer starts with `g_l = 0.5`.
The feed-forward network consumes `H'_l`.

### Parameter sharing

| Parameter | Shape | Sharing |
| --- | --- | --- |
| Knowledge vectors `M` | `[K, D]` | Shared by every layer |
| Relation matrix `E` | `[K, K]` | Shared by every layer and hop |
| Knowledge readout `V_l` | `[K, D]` | Separate for each layer |
| Residual gate logit `a_l` | scalar | Separate for each layer |
| Attention projections | `D -> Q/K/V` | Separate for each layer |
| Trace mixing logits | `[layers, heads]` | Separate for each layer and head |

## Recurrent trace

The model retains the final token's mixed knowledge activation for every layer
and attention head:

```text
trace shape = [B, layers, H, K]
```

Training code carries this state between chunks from the same document and
detaches it between optimization steps. The trace is reset at document
boundaries. This gives the knowledge path chunk-to-chunk state without
backpropagating through the full document history.

## Model configuration

The main constructor is `jakal_net.CausalMemoryLM`.

| Argument | Default | Meaning |
| --- | ---: | --- |
| `dim` | `512` | Token and Transformer hidden width |
| `max_seq_len` | `2048` | Maximum input chunk length |
| `transformer_layers` | `6` | Number of augmented Transformer blocks |
| `transformer_heads` | `8` | Causal self-attention heads |
| `transformer_dropout` | `0.0` | Attention and FFN dropout |
| `feed_forward_hidden_mult` | `4.0` | FFN hidden-width multiplier |
| `knowledge_memory_size` | `2048` | Number of shared knowledge vectors, `K` |
| `knowledge_hops` | `1` | Number of applications of the relation graph |
| `knowledge_activation` | `relu` | `relu`, `gelu`, or `softplus` |
| `tie_embedding_head` | `True` | Share token embedding and LM-head weights |

`MemoryScanOutput` contains logits, recurrent state, and optional compatibility
`Layer` views. `ModelRecurrentState` is the typed wrapper for recurrent state.
`CausalHierarchicalMemoryLM` remains as an alias of `CausalMemoryLM` for older
callers.

## Training

The training entry point is `scripts/train_causal_memory_lm.py`. It supports
plain text, JSONL, Hugging Face datasets, and pretokenized files or shard
directories.

The memory-model CLI derives its actual Transformer depth from:

```text
total layers = --s-layers + --prediction-layers
```

Those option names are retained for checkpoint and command compatibility. The
current model does not split them into separate sequence and prediction stacks.

Example using a pretokenized dataset:

```bash
PYTHONPATH=src python scripts/train_causal_memory_lm.py \
  --pretokenized-dir artifacts/wiki6m_hf16k/pretokenized_seq512 \
  --device cuda \
  --precision bf16 \
  --model-kind causal_memory \
  --seq-len 512 \
  --dim 1024 \
  --s-layers 10 \
  --prediction-layers 0 \
  --transformer-heads 8 \
  --knowledge-memory-size 4096 \
  --knowledge-hops 1 \
  --knowledge-activation relu \
  --feed-forward-hidden-mult 3 \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --optimizer adamw_fused \
  --tensorboard \
  --run-name knowledge-k4096-h1
```

Use the explicit Transformer baseline for ablation:

```bash
PYTHONPATH=src python scripts/train_causal_memory_lm.py \
  --pretokenized-dir artifacts/wiki6m_hf16k/pretokenized_seq512 \
  --device cuda \
  --precision bf16 \
  --model-kind transformer \
  --seq-len 512 \
  --dim 1024 \
  --transformer-layers 10 \
  --transformer-heads 8 \
  --transformer-ff-mult 3 \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --learning-rate 2e-4 \
  --optimizer adamw_fused \
  --tensorboard \
  --run-name transformer-baseline
```

`--disable-memory` and `--disable-memory-read` are compatibility flags from the
older architecture. They do not provide a clean current-model ablation. Use
`--model-kind transformer` instead.

Inspect training logs with:

```bash
tensorboard --logdir artifacts/training_runs
```

## Current experiment status

The curated June 2026 runs are stored under `reports/logs_202606/`.

| Run | Model | Best validation loss | Step |
| --- | --- | ---: | ---: |
| `base` | 10-layer Transformer baseline | `2.7341` | `143200` |
| `h1` | Post-lookup attention sharing, `K=4096`, one hop | `2.7888` | `84200` |
| `h2` | Residual two-hop knowledge path | No validation captured | `100` last copied |

The runs have different lengths, so their final best values are not a fair
head-to-head comparison. At the shared 84,200-step checkpoint, `h1` reached
`2.7888` while `base` was at `2.8512`. The knowledge model was ahead by
`0.0624` validation loss at equal steps, but it was about 4.23 times slower in
wall-clock time. The two-hop run was archived too early to compare.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/jakal_net/causal_memory_lm.py` | Transformer and shared knowledge module |
| `src/jakal_net/core.py` | Compatibility `Layer` and `LayerDelta` containers |
| `scripts/train_causal_memory_lm.py` | Data loading, training, evaluation, and checkpoints |
| `scripts/tokenizer_utils.py` | Byte-BPE and Hugging Face tokenizer support |
| `scripts/pretokenize_causal_memory_shards.py` | Pretokenized shard generation |
| `tests/` | `unittest` coverage |
| `reports/logs_202606/` | Curated baseline and knowledge-module logs |
| `artifacts/` | Generated datasets, runs, checkpoints, and local logs |
| `native/` | Historical optional C++/CUDA extension source |

Treat `artifacts/` as generated data. Do not commit large datasets,
checkpoints, TensorBoard runs, or local CUDA packages.

## Testing

Run the full suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Run only the model tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_causal_memory_lm.py"
```

The test suite checks output and trace shapes, document-boundary resets,
multi-hop execution, internal diagnostics, Query-based knowledge lookup, and
pretrained embedding initialization.

## Known limitations

- The relation matrix is dense. Its parameter and compute cost grow as
  `O(K^2)`.
- `knowledge_relation_rank` is a compatibility argument. The current model
  does not implement a low-rank relation matrix.
- `knowledge_beta_dim` is retained in the constructor and CLI but is not used
  by the current attention-sharing calculation.
- The knowledge lookup uses the Transformer's Query projection. It does not
  use the attention Key projection as the lookup vector.
- Several training flags and scripts still carry names from the removed
  propagation and hierarchical-memory implementations.
- The current knowledge path uses PyTorch tensor operations. The native source
  tree is not a fused implementation of this architecture.

## License

See [LICENSE](LICENSE).
