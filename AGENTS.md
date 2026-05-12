# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/jakal_net/`. Keep reusable operators, routing logic, and backend code there. Research and training entry points live in `scripts/`, including `train_progressive_b_lm.py`, `train_causal_memory_lm.py`, and corpus builders. Unit tests live in `tests/`. Native kernels are in `native/`, architecture docs in `docs/`, and experiment outputs in `artifacts/` and `reports/`. Treat `artifacts/` as generated data, not hand-edited source.

## Build, Test, and Development Commands
Set up a local environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

Key commands:

- `PYTHONPATH=src python scripts/smoke_test.py --device cpu` runs the basic CPU smoke check.
- `PYTHONPATH=src python -m unittest discover -s tests` runs the full test suite.
- `PYTHONPATH=src python scripts/build_native_extension.py` builds the optional C++/CUDA extension.
- `tensorboard --logdir artifacts/training_runs` inspects training logs.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints where practical, dataclass-based structures, and `snake_case` for functions, files, and variables. Classes use `PascalCase`. Test modules follow `test_*.py`. No dedicated formatter or linter is configured here, so match surrounding code closely and keep imports, typing, and error messages consistent with existing modules.

## Testing Guidelines
This repository uses `unittest`. Add or update tests in `tests/` whenever behavior changes in `src/jakal_net/` or training utilities. Prefer focused unit coverage for operator math, backend parity, and regression cases. Run the full suite before opening a PR; for native or CUDA work, also run the relevant smoke path or backend-specific test coverage.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `Fix dense low-rank propagation and unit-norm diagnostics`. Keep commits scoped to one change. PRs should describe the behavioral change, list validation commands, and attach logs or screenshots for training, benchmarking, or TensorBoard-visible changes. Link related issues when applicable.

## Artifact & Configuration Tips
Do not commit large generated datasets, checkpoints, TensorBoard runs, or local CUDA installer packages unless the change explicitly requires them. Prefer referencing paths under `artifacts/` in docs rather than checking in new bulky outputs.
