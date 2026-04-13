from __future__ import annotations

import argparse
import os
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-name", default="jakal_net_native")
    parser.add_argument("--build-dir", default="build_native")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "native" / "jakal_net_native.cpp"
    build_dir = repo_root / args.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    module = load(
        name=args.module_name,
        sources=[str(source)],
        build_directory=str(build_dir),
        verbose=args.verbose,
        with_cuda=False,
        extra_cflags=["/std:c++17"] if os.name == "nt" else ["-std=c++17"],
    )

    supported_ops = module.supported_ops()
    backend_name = module.backend_name()
    print(f"built module: {args.module_name}")
    print(f"backend: {backend_name}")
    print(f"supported ops: {supported_ops}")


if __name__ == "__main__":
    main()
