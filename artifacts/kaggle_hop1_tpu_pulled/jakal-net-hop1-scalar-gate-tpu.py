from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_BF16", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    source_archives = list(Path("/kaggle/input").glob("**/jakal-net-tpu-source.zip"))
    if len(source_archives) != 1:
        raise RuntimeError(f"Expected one Jakal-Net source archive, found {source_archives!r}.")

    source_root = Path("/kaggle/working/Jakal-Net")
    source_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archives[0]) as archive:
        archive.extractall(source_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "datasets>=3,<5",
            "transformers>=4.46,<6",
        ],
        check=True,
    )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root / "src")
    subprocess.run(
        [
            sys.executable,
            str(source_root / "scripts" / "train_kaggle_tpu_hop1.py"),
            "--data-dir",
            "/kaggle/working/wikitext103",
            "--output-dir",
            "/kaggle/working/jakal_hop1_tpu",
            "--seq-len",
            "512",
            "--dim",
            "1024",
            "--layers",
            "10",
            "--heads",
            "8",
            "--ff-mult",
            "3",
            "--knowledge-size",
            "4096",
            "--batch-size-per-core",
            "1",
            "--max-runtime-seconds",
            "34200",
            "--eval-interval",
            "1000",
            "--checkpoint-interval",
            "1000",
        ],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
