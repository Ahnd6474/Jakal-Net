from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load


def _cutlass_include_paths(repo_root: Path) -> list[str]:
    candidates: list[Path] = []
    env_path = os.environ.get("CUTLASS_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(repo_root / "third_party" / "cutlass")

    include_paths: list[str] = []
    seen: set[str] = set()
    for root in candidates:
        for include_dir in (root / "include", root / "tools" / "util" / "include"):
            if include_dir.exists():
                resolved = str(include_dir.resolve())
                if resolved not in seen:
                    include_paths.append(resolved)
                    seen.add(resolved)
    return include_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-name", default="jakal_net_native")
    parser.add_argument("--build-dir", default="build_native")
    parser.add_argument("--cuda", choices=("auto", "never", "always"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    native_dir = repo_root / "native"
    source = native_dir / "jakal_net_native.cpp"
    cuda_source = native_dir / "jakal_net_native_cuda.cu"
    build_dir = repo_root / args.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    sources = [str(source)]

    nvcc_available = shutil.which("nvcc") is not None or CUDA_HOME is not None
    use_cuda = False
    if args.cuda == "always":
        if not cuda_source.exists():
            raise FileNotFoundError(f"CUDA source was not found: {cuda_source}")
        if not nvcc_available:
            raise RuntimeError(
                "CUDA build requested but nvcc/CUDA_HOME is unavailable. "
                "Install the CUDA Toolkit or use --cuda never."
            )
        use_cuda = True
    elif args.cuda == "auto" and cuda_source.exists() and nvcc_available:
        use_cuda = True

    if use_cuda:
        sources.append(str(cuda_source))

    extra_cflags = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]
    if use_cuda:
        extra_cflags.append("/DWITH_CUDA") if os.name == "nt" else extra_cflags.append(
            "-DWITH_CUDA"
        )
    extra_include_paths = [str(native_dir)]
    extra_include_paths.extend(_cutlass_include_paths(repo_root))
    for candidate in (
        Path("/home/dannyahn/.local/opt/python3.12-dev-root/usr/include/python3.12"),
        Path("/home/dannyahn/.local/opt/python3.12-dev-root/usr/include"),
    ):
        if candidate.exists():
            extra_include_paths.append(str(candidate))

    module = load(
        name=args.module_name,
        sources=sources,
        build_directory=str(build_dir),
        verbose=args.verbose,
        with_cuda=use_cuda,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=["-std=c++17"] if use_cuda else None,
    )

    supported_ops = module.supported_ops()
    supported_devices = module.supported_devices()
    backend_name = module.backend_name()
    print(f"built module: {args.module_name}")
    print(f"backend: {backend_name}")
    print(f"cuda source enabled: {use_cuda}")
    print(f"supported devices: {supported_devices}")
    print(f"supported ops: {supported_ops}")


if __name__ == "__main__":
    main()
