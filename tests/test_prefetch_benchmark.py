from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_causal_memory_prefetch import (  # noqa: E402
    CommandSpec,
    VariantSpec,
    _parse_progress_line,
    parse_command_text,
    rewrite_command,
)
from summarize_causal_memory_prefetch import summarize_result  # noqa: E402


class PrefetchBenchmarkTests(unittest.TestCase):
    def test_parse_command_text_supports_env_and_line_continuations(self) -> None:
        command = parse_command_text(
            """
            PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/train_causal_memory_lm.py \
              --run-name base-run \
              --output-root artifacts/training_runs \
              --prebuild-train-batches
            """
        )

        self.assertEqual(command.env_updates, {"PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual(command.argv[0], "python")
        self.assertIn("--run-name", command.argv)
        self.assertIn("--prebuild-train-batches", command.argv)

    def test_rewrite_command_for_rolling_variant_removes_prebuild_flags(self) -> None:
        command = CommandSpec(
            env_updates={"PYTHONPATH": "src"},
            argv=(
                "python",
                "scripts/train_causal_memory_lm.py",
                "--run-name",
                "base-run",
                "--prebuild-train-batches",
                "--prebuild-workers",
                "4",
                "--prebuild-worker-threads",
                "2",
                "--pretokenized-max-loaded-shards",
                "8",
            ),
        )

        rewritten = rewrite_command(
            command,
            VariantSpec(name="rolling_cache0", mode="rolling", max_loaded_shards=0, preload_flat_shards=True),
            rolling_workers=2,
            rolling_worker_threads=1,
            rolling_block_size=16,
            rolling_blocks=4,
        )

        self.assertNotIn("--prebuild-train-batches", rewritten.argv)
        self.assertNotIn("--prebuild-workers", rewritten.argv)
        self.assertIn("--rolling-prefetch-workers", rewritten.argv)
        self.assertIn("--preload-flat-shards", rewritten.argv)
        self.assertIn("base-run-rolling_cache0", rewritten.argv)
        self.assertIn("--tensorboard", rewritten.argv)

    def test_rewrite_command_for_prebuild_variant_removes_rolling_flags(self) -> None:
        command = CommandSpec(
            env_updates={},
            argv=(
                "python",
                "scripts/train_causal_memory_lm.py",
                "--run-name",
                "base-run",
                "--rolling-prefetch-workers",
                "2",
                "--rolling-prefetch-worker-threads",
                "1",
                "--rolling-prefetch-block-size",
                "16",
                "--rolling-prefetch-blocks",
                "4",
            ),
        )

        rewritten = rewrite_command(
            command,
            VariantSpec(name="baseline_prebuild_cache8", mode="prebuild", max_loaded_shards=8, preload_flat_shards=False),
            rolling_workers=2,
            rolling_worker_threads=1,
            rolling_block_size=16,
            rolling_blocks=4,
        )

        self.assertIn("--prebuild-train-batches", rewritten.argv)
        self.assertNotIn("--rolling-prefetch-workers", rewritten.argv)
        self.assertNotIn("--preload-flat-shards", rewritten.argv)
        self.assertIn("base-run-baseline_prebuild_cache8", rewritten.argv)

    def test_parse_progress_line_extracts_expected_metrics(self) -> None:
        row = _parse_progress_line(
            "progress | step=   25/500 | stage=stage1 | span=1 | train_loss=1.2345 | lr=0.0003 | "
            "cpu_batch_ms=14.5 | prefetch_q=3 | elapsed=52.4s"
        )

        assert row is not None
        self.assertEqual(row["step"], 25)
        self.assertEqual(row["stage"], "stage1")
        self.assertEqual(row["span"], 1)
        self.assertAlmostEqual(row["cpu_batch_ms"], 14.5)
        self.assertAlmostEqual(row["prefetch_q"], 3.0)

    def test_summarize_result_uses_stdout_progress_window(self) -> None:
        result = {
            "variant": {
                "name": "rolling_cache0",
                "mode": "rolling",
                "max_loaded_shards": 0,
                "preload_flat_shards": True,
            },
            "log_path": "stdout.log",
            "run_dir": None,
            "tensorboard_dir": None,
            "stopped_at_target_step": True,
            "progress_rows": [
                {"step": 1, "elapsed_s": 10.0, "wall_time_s": 15.0, "cpu_batch_ms": 20.0, "prefetch_q": 2.0},
                {"step": 25, "elapsed_s": 40.0, "wall_time_s": 45.0, "cpu_batch_ms": 12.0, "prefetch_q": 3.0},
                {"step": 50, "elapsed_s": 55.0, "wall_time_s": 60.0, "cpu_batch_ms": 11.0, "prefetch_q": 2.0},
                {"step": 75, "elapsed_s": 70.0, "wall_time_s": 75.0, "cpu_batch_ms": 10.0, "prefetch_q": 1.0},
                {"step": 100, "elapsed_s": 85.0, "wall_time_s": 90.0, "cpu_batch_ms": 9.0, "prefetch_q": 1.0},
                {"step": 125, "elapsed_s": 100.0, "wall_time_s": 105.0, "cpu_batch_ms": 8.0, "prefetch_q": 0.0},
            ],
        }

        summary = summarize_result(result, steady_start_step=25, steady_end_step=125)

        self.assertAlmostEqual(summary["startup_seconds_to_step1"], 15.0)
        self.assertAlmostEqual(summary["steady_window_throughput_steps_per_s"], 100.0 / 60.0)
        self.assertAlmostEqual(summary["stdout_cpu_batch_ms_mean"], 10.0)
        self.assertAlmostEqual(summary["stdout_prefetch_q_median"], 1.0)
        self.assertEqual(summary["stdout_prefetch_q_zero_count"], 1)

    def test_summarize_result_reads_tensorboard_when_available(self) -> None:
        with TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir)
            event_file = benchmark_dir / "events.out.tfevents.fake"
            event_file.write_text("", encoding="utf-8")
            result = {
                "variant": {
                    "name": "rolling_cache8",
                    "mode": "rolling",
                    "max_loaded_shards": 8,
                    "preload_flat_shards": False,
                },
                "tensorboard_dir": str(benchmark_dir),
                "progress_rows": [],
            }

            with mock.patch(
                "summarize_causal_memory_prefetch._scalar_by_step",
                return_value={
                    "train/cpu_batch_ms": {25: 12.0, 50: 10.0, 125: 8.0},
                    "train/prefetch_queue_size": {25: 3.0, 50: 2.0, 125: 1.0},
                },
            ):
                summary = summarize_result(result, steady_start_step=25, steady_end_step=125)

            self.assertAlmostEqual(summary["tb_cpu_batch_ms_mean"], 10.0)
            self.assertAlmostEqual(summary["tb_prefetch_q_median"], 2.0)


if __name__ == "__main__":
    unittest.main()
