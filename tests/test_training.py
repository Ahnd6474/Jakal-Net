import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from progressive_b_example import (  # noqa: E402
    ProgressiveBExampleLM,
    build_char_vocab,
    compute_next_token_loss,
    estimate_next_token_loss,
    generate_next_tokens,
    sample_next_token_batch,
    split_train_val,
    train_next_token_model,
)


from train_progressive_b_lm import (  # noqa: E402
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    QUERY_BLOCK_START_TOKEN,
    USER_TOKEN,
    build_tokenizer,
    generate_next_tokens_with_sampling,
)
from train_causal_memory_lm import (  # noqa: E402
    BOS_TOKEN,
    CODE_TOKEN,
    CONT_TOKEN,
    DocumentChunk,
    DocumentBatch,
    FlatDocumentChunkBatcher,
    FlatIndexPrebuiltBatchBlock,
    DIALOGUE_TOKEN,
    EOT_TOKEN,
    INSTRUCTION_TOKEN,
    MATH_TOKEN,
    PrebuiltDocumentBatchProvider,
    RESPONSE_TOKEN,
    TEXT_TOKEN,
    TokenizedDocument,
    TrainingCurriculumStage,
    NoisyAdamWController,
    _content_target_visibility,
    build_noisy_adamw_controller,
    _normalize_dialogue_body,
    build_prebuilt_batch_blocks_parallel,
    build_special_token_id_map,
    load_flat_pretokenized_directory,
    make_document_chunks,
    apply_training_curriculum,
    preload_flat_pretokenized_directory,
    resolve_curriculum_stage,
    save_pretokenized_bundle,
    split_document_batch_three,
    split_train_val_flat_documents_with_collection,
)
from build_segmented_dialogue_corpus import dialogue_has_mixed_segments, segment_message_content_exact  # noqa: E402
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM  # noqa: E402

class TrainingTests(unittest.TestCase):
    def test_split_document_batch_three_partitions_batch(self) -> None:
        batch = DocumentBatch(
            context=torch.arange(24, dtype=torch.long).view(6, 4),
            target=torch.arange(24, dtype=torch.long).view(6, 4) + 1,
            loss_mask=torch.ones(6, 4, dtype=torch.float32),
            reset_mask=torch.tensor([True, False, False, True, False, False], dtype=torch.bool),
        )

        clean_batch, noise_batch1, noise_batch2 = split_document_batch_three(batch)

        self.assertEqual(clean_batch.context.shape[0], 2)
        self.assertEqual(noise_batch1.context.shape[0], 2)
        self.assertEqual(noise_batch2.context.shape[0], 2)
        torch.testing.assert_close(
            torch.cat((clean_batch.context, noise_batch1.context, noise_batch2.context), dim=0),
            batch.context,
        )

    def test_noisy_adamw_controller_updates_and_decays_noise_state(self) -> None:
        class TinyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pairwise_weight = torch.nn.Parameter(torch.full((4, 4), 0.5))
                self.bias = torch.nn.Parameter(torch.zeros(4))

        module = TinyModule()
        controller = build_noisy_adamw_controller(
            module,
            update_ema_beta=0.95,
            lag=0.95,
            noise_mult=1.0,
            noise_floor=1.0e-5,
            abs_floor=1.0e-4,
            abs_floor_steps=4,
            decay_steps=8,
            clean_weight=0.5,
            noisy_weight=0.25,
        )

        self.assertEqual(len(controller.named_parameters), 1)
        self.assertEqual(controller.named_parameters[0][0], "pairwise_weight")
        before = controller.capture_parameter_snapshots()
        with torch.no_grad():
            module.pairwise_weight.add_(0.1)
        controller.update_after_optimizer_step(before)
        perturbations = controller.sample_branch_noise(step=1, branch_seed=7)
        self.assertEqual(len(perturbations), 1)
        self.assertGreater(controller.last_rate, 0.0)
        self.assertGreater(controller.last_noise_rms, 0.0)

    def _write_flat_test_shards(self, root: Path) -> Path:
        corpus_info = {
            "special_tokens": {"pad": 0, "eos": 4, "cont": 3},
            "tokenized_summary": {"documents": 4, "chunks": 4, "tokens": 16},
        }
        shard1 = (
            TokenizedDocument(
                kind="text",
                source="wiki:test:1",
                token_count=4,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 2, 10, 0], dtype=torch.long),
                        target=torch.tensor([2, 10, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
            TokenizedDocument(
                kind="text",
                source="wiki:test:2",
                token_count=4,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 5, 11, 0], dtype=torch.long),
                        target=torch.tensor([5, 11, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
        )
        shard2 = (
            TokenizedDocument(
                kind="text",
                source="wiki:test:3",
                token_count=4,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 6, 12, 0], dtype=torch.long),
                        target=torch.tensor([6, 12, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
            TokenizedDocument(
                kind="text",
                source="wiki:test:4",
                token_count=4,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 7, 13, 0], dtype=torch.long),
                        target=torch.tensor([7, 13, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
        )
        save_pretokenized_bundle(
            root / "shard_0000.pt",
            documents=shard1,
            vocab_size=128,
            tokenizer_label="test",
            tokenizer_model_path=None,
            corpus_info=corpus_info,
        )
        save_pretokenized_bundle(
            root / "shard_0001.pt",
            documents=shard2,
            vocab_size=128,
            tokenizer_label="test",
            tokenizer_model_path=None,
            corpus_info=corpus_info,
        )
        return root

    def test_segment_message_content_exact_marks_code_and_math(self) -> None:
        content = "Explain this:\n```python\nprint(1)\n```\nThen solve $$x^2 + 1 = 0$$ please."

        segments = segment_message_content_exact(content)

        self.assertEqual([segment["kind"] for segment in segments], ["text", "code", "text", "math", "text"])
        self.assertIn("```python", segments[1]["text"])
        self.assertIn("$$x^2 + 1 = 0$$", segments[3]["text"])

    def test_segment_message_content_exact_marks_inline_dollar_math_and_execute_blocks(self) -> None:
        content = "Use <execute>print(1)</execute> and solve $x+1=2$."

        segments = segment_message_content_exact(content)

        self.assertEqual([segment["kind"] for segment in segments], ["text", "code", "text", "math", "text"])
        self.assertEqual(segments[1]["text"], "<execute>print(1)</execute>")
        self.assertEqual(segments[3]["text"], "$x+1=2$")

    def test_normalize_dialogue_body_inserts_text_segments(self) -> None:
        body = _normalize_dialogue_body(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        )

        assert body is not None
        self.assertIn(USER_TOKEN, body)
        self.assertIn(ASSISTANT_TOKEN, body)
        self.assertEqual(body.count(TEXT_TOKEN), 2)

    def test_normalize_dialogue_body_uses_explicit_segments(self) -> None:
        body = _normalize_dialogue_body(
            [
                {
                    "role": "assistant",
                    "segments": [
                        {"kind": "text", "text": "Let's derive it."},
                        {"kind": "math", "text": "$$E = mc^2$$"},
                        {"kind": "code", "text": "```python\nprint(1)\n```"},
                    ],
                }
            ]
        )

        assert body is not None
        self.assertIn(TEXT_TOKEN, body)
        self.assertIn(MATH_TOKEN, body)
        self.assertIn(CODE_TOKEN, body)

    def test_dialogue_has_mixed_segments_requires_text_and_code_or_math(self) -> None:
        self.assertTrue(
            dialogue_has_mixed_segments(
                [
                    {"role": "assistant", "segments": [{"kind": "text", "text": "Explain"}, {"kind": "code", "text": "```python\nprint(1)\n```"}]}
                ]
            )
        )
        self.assertFalse(
            dialogue_has_mixed_segments(
                [
                    {"role": "assistant", "segments": [{"kind": "code", "text": "```python\nprint(1)\n```"}]}
                ]
            )
        )

    def test_assistant_only_visibility_marks_assistant_span(self) -> None:
        special = {
            "assistant": 10,
            "response": 11,
            "user": 12,
            "eot": 13,
        }
        token_ids = torch.tensor([12, 20, 21, 13, 10, 30, 31, 13], dtype=torch.long)

        visible = _content_target_visibility(token_ids, loss_mode="assistant_only", special_token_ids=special)

        torch.testing.assert_close(
            visible,
            torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        )

    def test_make_document_chunks_respects_assistant_only_mask(self) -> None:
        token_ids = torch.tensor([12, 20, 21, 13, 10, 30, 31, 13], dtype=torch.long)
        visible = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        chunks = make_document_chunks(
            content_ids=token_ids,
            content_target_visibility=visible,
            mode_token_id=2,
            seq_len=16,
            bos_token_id=1,
            cont_token_id=3,
            eos_token_id=4,
            pad_token_id=0,
        )
        self.assertEqual(len(chunks), 1)
        self.assertGreater(float(chunks[0].loss_mask.sum().item()), 0.0)
        self.assertEqual(float(chunks[0].loss_mask[1].item()), 0.0)
        self.assertEqual(float(chunks[0].loss_mask[5].item()), 1.0)

    def test_curriculum_stage_resolution(self) -> None:
        stage1 = resolve_curriculum_stage(
            step=1,
            total_steps=100,
            stage1_ratio=0.1,
            stage2_ratio=0.4,
            stage2_span=4,
            stage3_span=8,
        )
        stage2 = resolve_curriculum_stage(
            step=20,
            total_steps=100,
            stage1_ratio=0.1,
            stage2_ratio=0.4,
            stage2_span=4,
            stage3_span=8,
        )
        stage3 = resolve_curriculum_stage(
            step=80,
            total_steps=100,
            stage1_ratio=0.1,
            stage2_ratio=0.4,
            stage2_span=4,
            stage3_span=8,
        )

        self.assertEqual(stage1, TrainingCurriculumStage("stage1", 1, True, True, True))
        self.assertEqual(stage2, TrainingCurriculumStage("stage2", 4, False, True, True))
        self.assertEqual(stage3, TrainingCurriculumStage("stage3", 8, False, False, False))

    def test_apply_training_curriculum_freezes_memory_paths(self) -> None:
        model = CausalHierarchicalMemoryLM(vocab_size=64, dim=16, max_seq_len=16, memory_slots=(8, 4))

        apply_training_curriculum(model, TrainingCurriculumStage("stage1", 1, True, True, True))

        self.assertFalse(model.memory_levels[0].init_state.requires_grad)
        self.assertFalse(next(model.memory_levels[0].write.parameters()).requires_grad)
        self.assertFalse(next(model.read_projections[0].parameters()).requires_grad)
        self.assertFalse(model.skip_gates["token_to_1"].requires_grad)

        apply_training_curriculum(model, TrainingCurriculumStage("stage2", 4, False, True, True))

        self.assertTrue(model.memory_levels[0].init_state.requires_grad)
        self.assertTrue(next(model.memory_levels[0].write.parameters()).requires_grad)
        self.assertFalse(next(model.memory_levels[0].propagation.parameters()).requires_grad)
        self.assertFalse(model.skip_gates["token_to_1"].requires_grad)

    def test_build_tokenizer_byte_bpe_with_special_tokens(self) -> None:
        text = (
            f"{USER_TOKEN}\nhello\n{ASSISTANT_TOKEN}\nworld\n{EOS_TOKEN}\n"
            f"{USER_TOKEN}\nbyte bpe test\n{ASSISTANT_TOKEN}\nworks\n{PAD_TOKEN}\n"
            f"{QUERY_BLOCK_START_TOKEN}\n"
        )
        with TemporaryDirectory() as tmpdir:
            vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
                text,
                text_path=None,
                tokenizer="byte_bpe",
                subword_vocab_size=128,
                subword_model_type="bpe",
                tokenizer_prefix=str(Path(tmpdir) / "byte_bpe_test"),
                subword_character_coverage=0.9995,
                user_defined_symbols=(
                    USER_TOKEN,
                    ASSISTANT_TOKEN,
                    EOS_TOKEN,
                    PAD_TOKEN,
                    QUERY_BLOCK_START_TOKEN,
                ),
            )

            self.assertEqual(tokenizer_label, "byte_bpe")
            self.assertIsNotNone(tokenizer_model_path)
            assert tokenizer_model_path is not None
            self.assertTrue(tokenizer_model_path.exists())
            encoded = vocab.encode(text)
            self.assertGreater(encoded.numel(), 0)
            self.assertGreaterEqual(vocab.token_id(USER_TOKEN), 0)
            self.assertGreaterEqual(vocab.token_id(QUERY_BLOCK_START_TOKEN), 0)
            decoded = vocab.decode(encoded[:16].tolist())
            self.assertIsInstance(decoded, str)

    def test_char_vocab_roundtrip(self) -> None:
        vocab = build_char_vocab("abca")
        encoded = vocab.encode("caba")
        decoded = vocab.decode(encoded.tolist())

        self.assertEqual(vocab.size, 3)
        self.assertEqual(decoded, "caba")

    def test_sampling_returns_context_and_target(self) -> None:
        tokens = torch.arange(20, dtype=torch.long)

        batch = sample_next_token_batch(tokens, seq_len=5, batch_size=4, device="cpu")

        self.assertEqual(batch.context.shape, (4, 5))
        self.assertEqual(batch.target.shape, (4,))

    def test_query_block_sampling_returns_generated_tokens(self) -> None:
        torch.manual_seed(9)
        model = ProgressiveBExampleLM(
            vocab_size=32,
            dim=8,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )
        prompt = torch.randint(0, 32, (8,), dtype=torch.long)
        generated = generate_next_tokens_with_sampling(
            model,
            prompt,
            max_new_tokens=4,
            seq_len=8,
            device="cpu",
            temperature=None,
            sample_topk=None,
            training_objective="query_block",
            target_len=4,
            query_block_start_token_id=31,
        )

        self.assertEqual(generated.shape, (12,))

    def test_preload_flat_pretokenized_directory_overrides_loaded_shard_limit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shard_dir = self._write_flat_test_shards(Path(tmpdir))
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=1,
                integrity_mode="meta",
                integrity_workers=1,
            )

            self.assertEqual(collection.max_loaded_shards, 1)

            preload_flat_pretokenized_directory(collection)

            self.assertEqual(collection.max_loaded_shards, 0)
            self.assertEqual(len(collection._loaded_shard_lru), 0)
            self.assertTrue(all(shard._loaded for shard in collection.shards))

    def test_flat_process_prebuild_blocks_complete_with_spawned_fresh_workers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = self._write_flat_test_shards(root / "shards")
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=1,
                integrity_mode="meta",
                integrity_workers=1,
            )
            train_documents_flat, _ = split_train_val_flat_documents_with_collection(
                collection,
                collection.document_refs,
                train_fraction=0.75,
            )
            batcher = FlatDocumentChunkBatcher(
                collection,
                tuple(ref for ref in train_documents_flat if collection.get_shard(ref.shard_index).source[ref.document_index].startswith("wiki:")),
                batch_size=1,
                device=torch.device("cpu"),
                active_shards_per_bucket=1,
                shard_rotation_interval=4,
            )

            blocks, seconds = build_prebuilt_batch_blocks_parallel(
                batcher,
                total_steps=16,
                start_step=0,
                stage1_ratio=1.0,
                stage2_ratio=1.0,
                stage1_span=1,
                stage2_span=1,
                stage3_span=1,
                stage1_batch_size=1,
                stage2_batch_size=1,
                stage3_batch_size=1,
                stage1_grad_accum_steps=1,
                stage2_grad_accum_steps=1,
                stage3_grad_accum_steps=1,
                pin_memory=False,
                workers=2,
                worker_threads=1,
                cache_dir=root / "unused_cache",
                flat_train_fraction=0.75,
            )

            self.assertGreaterEqual(seconds, 0.0)
            self.assertGreaterEqual(sum(block.batch_count for block in blocks), 16)
            self.assertGreater(len(blocks), 2)
            self.assertIsInstance(blocks[0], FlatIndexPrebuiltBatchBlock)
            block = blocks[0]
            assert isinstance(block, FlatIndexPrebuiltBatchBlock)
            self.assertEqual(block.shard_index.shape[1:], (1,))
            self.assertFalse(any((root / "unused_cache").glob("batch_block_*.pt")))

    def test_flat_index_prebuilt_provider_reconstructs_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shard_dir = self._write_flat_test_shards(Path(tmpdir))
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=0,
                integrity_mode="meta",
                integrity_workers=1,
            )
            block = FlatIndexPrebuiltBatchBlock(
                shard_index=torch.tensor([[0, 0]], dtype=torch.int32),
                document_index=torch.tensor([[0, 1]], dtype=torch.int32),
                chunk_index=torch.tensor([[0, 1]], dtype=torch.int32),
                reset_mask=torch.tensor([[True, False]], dtype=torch.bool),
                build_seconds=0.0,
            )
            provider = PrebuiltDocumentBatchProvider(
                device=torch.device("cpu"),
                build_seconds=0.0,
                blocks=[block],
                flat_collection=collection,
            )
            batch = provider.next_batch()

            torch.testing.assert_close(
                batch.context,
                torch.tensor([[1, 2, 10, 0], [1, 5, 11, 0]], dtype=torch.long),
            )
            torch.testing.assert_close(
                batch.target,
                torch.tensor([[2, 10, 4, 0], [5, 11, 4, 0]], dtype=torch.long),
            )
            torch.testing.assert_close(
                batch.loss_mask,
                torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
            )
            torch.testing.assert_close(
                batch.reset_mask,
                torch.tensor([True, False], dtype=torch.bool),
            )

    def test_flat_index_prebuilt_provider_caches_materialized_block(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shard_dir = self._write_flat_test_shards(Path(tmpdir))
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=0,
                integrity_mode="meta",
                integrity_workers=1,
            )
            block = FlatIndexPrebuiltBatchBlock(
                shard_index=torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
                document_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
                chunk_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
                reset_mask=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
                build_seconds=0.0,
            )
            provider = PrebuiltDocumentBatchProvider(
                device=torch.device("cpu"),
                build_seconds=0.0,
                blocks=[block],
                flat_collection=collection,
            )
            original_get_shard = type(collection).get_shard
            call_count = 0

            def counted_get_shard(instance, shard_index: int):
                nonlocal call_count
                call_count += 1
                return original_get_shard(instance, shard_index)

            with patch.object(type(collection), "get_shard", autospec=True, side_effect=counted_get_shard):
                first = provider.next_batch()
                first_count = call_count
                second = provider.next_batch()
                second_count = call_count

            self.assertGreater(first_count, 0)
            self.assertEqual(first_count, second_count)
            self.assertEqual(first.context.shape, (2, 4))
            self.assertEqual(second.context.shape, (2, 4))

    def test_flat_batcher_reuses_loaded_shard_within_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shard_dir = self._write_flat_test_shards(Path(tmpdir))
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=0,
                integrity_mode="meta",
                integrity_workers=1,
            )
            batcher = FlatDocumentChunkBatcher(
                collection,
                collection.document_refs[:2],
                batch_size=2,
                device=torch.device("cpu"),
                active_shards_per_bucket=1,
                shard_rotation_interval=4,
            )
            batcher.current_doc = [0, 1]
            batcher.current_chunk = [0, 0]
            batcher.needs_reset = [False, False]

            original_get_shard = type(collection).get_shard
            call_count = 0

            def counted_get_shard(instance, shard_index: int):
                nonlocal call_count
                call_count += 1
                return original_get_shard(instance, shard_index)

            with patch.object(type(collection), "get_shard", autospec=True, side_effect=counted_get_shard):
                batch = batcher.next_batch()

            self.assertEqual(batch.context.shape, (2, 4))
            self.assertEqual(call_count, 1)

    def test_flat_index_batcher_uses_metadata_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            shard_dir = self._write_flat_test_shards(Path(tmpdir))
            collection = load_flat_pretokenized_directory(
                shard_dir,
                load_workers=1,
                max_loaded_shards=0,
                integrity_mode="meta",
                integrity_workers=1,
            )
            batcher = FlatDocumentChunkBatcher(
                collection,
                collection.document_refs[:2],
                batch_size=2,
                device=torch.device("cpu"),
                active_shards_per_bucket=1,
                shard_rotation_interval=4,
            )
            batcher.current_doc = [0, 1]
            batcher.current_chunk = [0, 0]
            batcher.needs_reset = [False, False]
            shard_index = torch.zeros((2,), dtype=torch.int32)
            document_index = torch.zeros((2,), dtype=torch.int32)
            chunk_index = torch.zeros((2,), dtype=torch.int32)
            reset_mask = torch.zeros((2,), dtype=torch.bool)

            with patch.object(type(collection), "get_shard", autospec=True, side_effect=AssertionError("unexpected shard load")):
                batcher.next_index_batch_into(
                    shard_index_out=shard_index,
                    document_index_out=document_index,
                    chunk_index_out=chunk_index,
                    reset_mask_out=reset_mask,
                )

            torch.testing.assert_close(shard_index, torch.tensor([0, 0], dtype=torch.int32))
            torch.testing.assert_close(document_index, torch.tensor([0, 1], dtype=torch.int32))
            torch.testing.assert_close(chunk_index, torch.tensor([0, 1], dtype=torch.int32))
            torch.testing.assert_close(reset_mask, torch.tensor([False, False], dtype=torch.bool))

    def test_query_block_sampling_prepends_start_token(self) -> None:
        tokens = torch.arange(40, dtype=torch.long)

        batch = sample_next_token_batch(
            tokens,
            seq_len=5,
            batch_size=4,
            device="cpu",
            target_len=4,
            query_block_start_token_id=99,
        )

        self.assertEqual(batch.target.shape, (4, 5))
        self.assertTrue(torch.equal(batch.target[:, 0], torch.full((4,), 99, dtype=torch.long)))
        torch.testing.assert_close(batch.target[:, 1], batch.context[:, -1] + 1)
        torch.testing.assert_close(batch.target[:, 2], batch.context[:, -1] + 2)
        torch.testing.assert_close(batch.target[:, 3], batch.context[:, -1] + 3)
        torch.testing.assert_close(batch.target[:, 4], batch.context[:, -1] + 4)

    def test_full_sequence_causal_sampling_and_loss(self) -> None:
        torch.manual_seed(7)
        text = "full sequence causal objective keeps every position supervised. " * 8
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        batch = sample_next_token_batch(
            tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            full_sequence_causal=True,
        )
        loss, logits = compute_next_token_loss(
            model,
            batch,
            full_sequence_causal=True,
        )

        self.assertEqual(batch.context.shape, (2, 8))
        self.assertEqual(batch.target.shape, (2, 8))
        self.assertEqual(logits.shape, (2, 8, vocab.size))
        self.assertTrue(torch.isfinite(loss))

    def test_teacher_forcing_chunked_matches_full_teacher_forcing(self) -> None:
        torch.manual_seed(11)
        text = "teacher forcing with chunking should preserve logits exactly. " * 8
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        batch = sample_next_token_batch(
            tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            teacher_forcing=True,
        )
        full_loss, full_logits = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=True,
        )
        chunked_loss, chunked_logits = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=True,
            teacher_forcing_chunk_size=3,
        )

        torch.testing.assert_close(chunked_logits, full_logits)
        torch.testing.assert_close(chunked_loss, full_loss)

    def test_training_loop_runs_and_returns_history(self) -> None:
        torch.manual_seed(30)
        text = "progressive b activation helps stable training. " * 12
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.8)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        history = train_next_token_model(
            model,
            train_tokens,
            val_tokens,
            seq_len=8,
            batch_size=4,
            device="cpu",
            steps=4,
            eval_interval=2,
            eval_steps=2,
            learning_rate=1e-3,
        )

        self.assertGreaterEqual(len(history.train_losses), 2)
        self.assertEqual(len(history.train_losses), len(history.val_losses))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.train_losses))))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.val_losses))))

    def test_teacher_forcing_chunked_training_loop_runs(self) -> None:
        torch.manual_seed(32)
        text = "chunked teacher forcing should keep the B path active while limiting memory. " * 8
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.8)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        history = train_next_token_model(
            model,
            train_tokens,
            val_tokens,
            seq_len=8,
            batch_size=4,
            device="cpu",
            steps=3,
            eval_interval=2,
            eval_steps=1,
            learning_rate=1e-3,
            teacher_forcing=True,
            teacher_forcing_chunk_size=3,
        )

        self.assertGreaterEqual(len(history.train_losses), 2)
        self.assertEqual(len(history.train_losses), len(history.val_losses))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.train_losses))))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.val_losses))))

        batch = sample_next_token_batch(train_tokens, seq_len=8, batch_size=2, device="cpu")
        loss, logits = compute_next_token_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.shape, (2, vocab.size))

        estimated = estimate_next_token_loss(
            model,
            val_tokens,
            seq_len=8,
            batch_size=2,
            device="cpu",
            eval_steps=2,
        )
        self.assertTrue(torch.isfinite(torch.tensor(estimated)))

        prompt = train_tokens[:8]
        generated = generate_next_tokens(
            model,
            prompt,
            max_new_tokens=5,
            seq_len=8,
            device="cpu",
        )
        self.assertEqual(generated.shape, (13,))

    def test_full_sequence_causal_training_loop_runs(self) -> None:
        torch.manual_seed(31)
        text = "causal sequence objective should train without prefix expansion. " * 10
        vocab = build_char_vocab(text)
        tokens = vocab.encode(text)
        train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.8)
        model = ProgressiveBExampleLM(
            vocab_size=vocab.size,
            dim=12,
            seq_nodes=8,
            warmup_layers=1,
            final_refine_layers=1,
        )

        history = train_next_token_model(
            model,
            train_tokens,
            val_tokens,
            seq_len=8,
            batch_size=4,
            device="cpu",
            steps=3,
            eval_interval=2,
            eval_steps=1,
            learning_rate=1e-3,
            full_sequence_causal=True,
        )

        self.assertGreaterEqual(len(history.train_losses), 2)
        self.assertEqual(len(history.train_losses), len(history.val_losses))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.train_losses))))
        self.assertTrue(all(torch.isfinite(torch.tensor(history.val_losses))))


if __name__ == "__main__":
    unittest.main()
