# Jakal-Net

Latent workspace multimodal model scaffolding in PyTorch.

현재 저장소는 연산 최적화를 염두에 둔 핵심 인터페이스부터 고정한다.

- `Layer`: 노드 저장소만 담당
- `Propagation`: 같은 레이어 내부 pairwise relation 기반 전파
- `SparsePropagation`: `window` 또는 `topk` 제한 전파
- `Transition`: 레이어 간 routing + softmax + state activation 기반 이동
- `SparseTransition`: `topk` sparse routing 기반 이동

## Design Rules

- Propagation에서는 `state` 에 activation을 걸지 않는다.
- Propagation에서는 edge score만 `softsign` 같은 압축 함수를 통과시킨다.
- Transition에서는 routing 이후 송신 강도용 `activation(state)` 를 사용한다.
- Transition의 기본 state activation은 `softplus` 이다.
- dense 구현과 sparse 구현의 인터페이스를 먼저 맞춰 두고, 이후 `Flash Propagation` / `Flash Transition` 으로 교체할 수 있게 한다.

## Layout

```text
src/jakal_net/
  core.py
  propagation.py
  transition.py
  modules.py
```

## Quick Start

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

실행:

```bash
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

## Minimal Example

```python
import torch

from jakal_net import (
    DiagonalBilinearPairwise,
    Layer,
    LinearRoute,
    Propagation,
    SparseTransition,
)

layer = Layer.zeros(dim=16, num_nodes=32, batch_shape=(4,))
layer = layer.with_tensors(
    state=torch.randn_like(layer.state),
    val=torch.randn_like(layer.val),
)

propagation = Propagation(pairwise_fn=DiagonalBilinearPairwise(dim=16))
delta = propagation(layer)

dst = Layer.zeros(dim=16, num_nodes=8, batch_shape=(4,))
transition = SparseTransition(route_fn=LinearRoute(src_dim=16, dst_nodes=8), topk=2)
updated_dst = transition(layer, dst)
```

## Testing

```bash
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
