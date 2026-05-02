import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_causal_memory_lm import (  # noqa: E402
    DocumentChunk,
    DocumentChunkBatcher,
    TokenizedDocument,
    load_pretokenized_bundle,
    make_document_chunks,
    save_pretokenized_bundle,
)

from jakal_net.causal_memory_lm import (  # noqa: E402
    CausalHierarchicalMemoryLM,
    MemoryScanOutput,
    ModelRecurrentState,
    ValueNormStateProjection,
    _nomemory_exact_prediction_forward_with_postmix,
    _nomemory_exact_run_layers_with_postmix,
    _nomemory_exact_stack_fused,
)
from jakal_net._architectural_common import apply_delta, softsign_state  # noqa: E402
from jakal_net.core import Layer, LayerDelta  # noqa: E402
from jakal_net.kernel_common import pairwise_kernel_spec  # noqa: E402
from jakal_net.latent_graph import KModule  # noqa: E402
from jakal_net.modules import MultiHeadPairwise  # noqa: E402
from jakal_net.modules import LowRankBilinearPairwise  # noqa: E402
from jakal_net.native_backend import (  # noqa: E402
    _native_scan_uses_legacy_low_rank_extension,
    native_supports,
    nomemory_causal_stack_ffn_fused_native_available,
    nomemory_causal_stack_fused_native_available,
)


class CausalMemoryLMTests(unittest.TestCase):
    def test_forward_returns_sequence_logits_and_memory(self) -> None:
        torch.manual_seed(7)
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4, 2),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True, return_layers=True)

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertEqual(len(output.memory_state), 3)
        self.assertEqual(output.memory_state[0].val.shape, (2, 6, 8))
        self.assertTrue(model.feed_forward_layers)
        self.assertTrue(model.s_module.feed_forward_layers)
        self.assertTrue(model.b_module.feed_forward_layers)
        self.assertEqual(model.feed_forward_hidden_mult, 2.0)
        assert output.sequence_layer is not None
        assert output.query_layer is not None
        self.assertEqual(output.sequence_layer.val.shape, (2, 6, 8))
        self.assertEqual(output.query_layer.val.shape, (2, 5, 8))

    def test_can_disable_only_memory_feed_forward_layers(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            memory_feed_forward_layers=False,
        )

        self.assertTrue(model.feed_forward_layers)
        self.assertTrue(model.s_module.feed_forward_layers)
        self.assertFalse(model.b_module.feed_forward_layers)
        self.assertTrue(model._native_scan_supported_config())

    def test_memory_ablation_flags_forward(self) -> None:
        token_ids = torch.randint(0, 32, (2, 5))
        for kwargs in (
            {"disable_memory": True},
            {"disable_memory_read": True},
            {"disable_memory_propagation": True},
        ):
            model = CausalHierarchicalMemoryLM(
                vocab_size=32,
                dim=8,
                max_seq_len=12,
                s_layers=1,
                memory_slots=(6, 4),
                prediction_layers=1,
                memory_topk=2,
                pairwise_rank=4,
                route_rank=4,
                **kwargs,
            )

            output = model(token_ids, return_memory_state=True)

            self.assertIsInstance(output, MemoryScanOutput)
            self.assertEqual(output.logits.shape, (2, 5, 32))
            self.assertEqual(len(output.memory_state), 2)
            if kwargs.get("disable_memory_propagation"):
                self.assertFalse(model.b_module.memory_propagation_layers)

    def test_reset_mask_preserves_fresh_path_for_selected_items(self) -> None:
        torch.manual_seed(8)
        model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        token_ids = torch.randint(0, 24, (2, 4))

        fresh_output = model(token_ids, return_memory_state=True)
        assert isinstance(fresh_output, MemoryScanOutput)
        carried_output = model(
            token_ids,
            memory_state=fresh_output.memory_state,
            reset_mask=torch.tensor([True, False]),
            return_memory_state=True,
        )
        assert isinstance(carried_output, MemoryScanOutput)

        self.assertTrue(torch.allclose(carried_output.logits[0], fresh_output.logits[0]))
        self.assertFalse(torch.allclose(carried_output.logits[1], fresh_output.logits[1]))

    def test_s_microbatch_matches_full_batch_forward(self) -> None:
        torch.manual_seed(9)
        base_model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=8,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        microbatched_model = CausalHierarchicalMemoryLM(
            vocab_size=24,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(5, 3),
            prediction_layers=1,
            s_window=8,
            s_microbatch_size=2,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
        )
        microbatched_model.load_state_dict(base_model.state_dict())
        token_ids = torch.randint(0, 24, (4, 5))

        base_output = base_model(token_ids, return_memory_state=True)
        micro_output = microbatched_model(token_ids, return_memory_state=True)

        assert isinstance(base_output, MemoryScanOutput)
        assert isinstance(micro_output, MemoryScanOutput)
        self.assertTrue(torch.allclose(base_output.logits, micro_output.logits))
        self.assertEqual(len(base_output.memory_state), len(micro_output.memory_state))
        for base_layer, micro_layer in zip(base_output.memory_state, micro_output.memory_state):
            self.assertTrue(torch.allclose(base_layer.state, micro_layer.state))
            self.assertTrue(torch.allclose(base_layer.val, micro_layer.val))

    def test_forward_logits_remain_finite(self) -> None:
        torch.manual_seed(10)
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=2,
            memory_slots=(8, 4, 2),
            prediction_layers=2,
            s_window=8,
            s_microbatch_size=1,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
        )
        token_ids = torch.randint(0, 64, (2, 8))

        logits = model(token_ids)

        self.assertTrue(torch.isfinite(logits).all().item())

    def test_constructor_accepts_scan_backend_compatibility_args(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4, 2),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            scan_backend="native",
            scan_checkpoint_chunk_size=32,
        )

        self.assertEqual(model.scan_backend, "native")
        self.assertEqual(model.scan_checkpoint_chunk_size, 32)

    def test_feed_forward_layers_are_supported_by_native_scan(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
        )

        self.assertTrue(model._native_scan_supported_config())

    def test_direction_only_values_use_softsign_state_without_value_l2_normalization(self) -> None:
        layer = Layer(
            dim=4,
            num_nodes=3,
            state=torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32),
            val=torch.tensor(
                [[[3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 2.0, 1.0], [0.5, 0.5, 0.5, 0.5]]],
                dtype=torch.float32,
            ),
        )
        delta = LayerDelta(
            delta_state=torch.tensor([[1.5, 0.5, -0.5]], dtype=torch.float32),
            delta_val=torch.tensor(
                [[[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 3.0]]],
                dtype=torch.float32,
            ),
        )

        updated = apply_delta(layer, delta, residual=True, direction_only_values=True)

        self.assertTrue(torch.allclose(updated.state, softsign_state(layer.state + delta.delta_state)))
        self.assertTrue(torch.equal(updated.val, layer.val + delta.delta_val))

    def test_direction_only_values_keep_sequence_dense_fastpath(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            scan_backend="native",
            direction_only_values=True,
            feed_forward_layers=False,
        )
        layer = Layer(
            dim=16,
            num_nodes=17,
            state=torch.randn(2, 17),
            val=torch.randn(2, 17, 16),
        )
        delta = LayerDelta(
            delta_state=torch.randn_like(layer.state),
            delta_val=torch.randn_like(layer.val),
        )

        self.assertTrue(model.s_module._can_use_dense_apply_fastpath(layer, model.s_module.sequence_layers[0]))
        expected = layer.with_tensors(
            state=softsign_state(layer.state + delta.delta_state),
            val=model.s_module.sequence_norms[0](layer.val + delta.delta_val),
        )
        actual = model.s_module._apply_dense_delta_fastpath(
            layer,
            delta.delta_state,
            delta.delta_val,
            model.s_module.sequence_norms[0],
            model.s_module.sequence_layers[0],
        )
        self.assertTrue(torch.allclose(actual.state, expected.state))
        self.assertTrue(torch.allclose(actual.val, expected.val))

    def test_direction_only_values_keep_prediction_dense_fastpath(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            scan_backend="native",
            direction_only_values=True,
            feed_forward_layers=False,
        )
        layer = Layer(
            dim=16,
            num_nodes=16,
            state=torch.randn(2, 16),
            val=torch.randn(2, 16, 16),
        )
        delta = LayerDelta(
            delta_state=torch.randn_like(layer.state),
            delta_val=torch.randn_like(layer.val),
        )

        self.assertTrue(model._can_use_dense_apply_fastpath(layer, model.prediction_layers[0]))
        expected = layer.with_tensors(
            state=softsign_state(layer.state + delta.delta_state),
            val=model.prediction_norms[0](layer.val + delta.delta_val),
        )
        actual = model._apply_dense_delta_fastpath(
            layer,
            delta.delta_state,
            delta.delta_val,
            model.prediction_norms[0],
            model.prediction_layers[0],
        )
        self.assertTrue(torch.allclose(actual.state, expected.state))
        self.assertTrue(torch.allclose(actual.val, expected.val))

    def test_nomemory_exact_stack_fastpath_requires_explicit_env_gate(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            disable_memory=True,
            feed_forward_layers=False,
        )
        token_ids = torch.randint(0, 64, (2, 16))

        self.assertFalse(
            model._supports_nomemory_exact_stack_fastpath(
                token_ids,
                return_layers=False,
            )
        )

    def test_forward_can_dispatch_to_nomemory_exact_stack_fastpath(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            disable_memory=True,
            feed_forward_layers=False,
        )
        token_ids = torch.randint(0, 32, (2, 5))
        fast_query_val = torch.zeros(2, 5, 8)

        with patch.object(
            model,
            "_supports_nomemory_exact_stack_fastpath",
            return_value=True,
        ) as supports_mock, patch.object(
            model,
            "_run_nomemory_exact_stack_fastpath",
            return_value=fast_query_val,
        ) as run_mock:
            logits = model(token_ids)

        supports_mock.assert_called_once()
        run_mock.assert_called_once()
        self.assertEqual(logits.shape, (2, 5, 32))

    def test_nomemory_exact_stack_ffn_fastpath_requires_explicit_env_gate(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            disable_memory=True,
            feed_forward_layers=True,
        )
        token_ids = torch.randint(0, 64, (2, 16))

        self.assertFalse(
            model._supports_nomemory_exact_stack_ffn_fastpath(
                token_ids,
                return_layers=False,
            )
        )

    def test_forward_can_dispatch_to_nomemory_exact_stack_ffn_fastpath(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            disable_memory=True,
            feed_forward_layers=True,
        )
        token_ids = torch.randint(0, 32, (2, 5))
        fast_query_val = torch.zeros(2, 5, 8)

        with patch.object(
            model,
            "_supports_nomemory_exact_stack_fastpath",
            return_value=False,
        ) as supports_noffn_mock, patch.object(
            model,
            "_supports_nomemory_exact_stack_ffn_fastpath",
            return_value=True,
        ) as supports_ffn_mock, patch.object(
            model,
            "_run_nomemory_exact_stack_ffn_fastpath",
            return_value=fast_query_val,
        ) as run_mock:
            logits = model(token_ids)

        supports_noffn_mock.assert_called_once()
        supports_ffn_mock.assert_called_once()
        run_mock.assert_called_once()
        self.assertEqual(logits.shape, (2, 5, 32))

    def test_nomemory_exact_stack_linear_fastpath_requires_explicit_env_gate(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            disable_memory=True,
            feed_forward_layers=True,
            feed_forward_kind="linear",
        )
        token_ids = torch.randint(0, 64, (2, 16))

        self.assertFalse(
            model._supports_nomemory_exact_stack_linear_fastpath(
                token_ids,
                return_layers=False,
            )
        )

    def test_forward_can_dispatch_to_nomemory_exact_stack_linear_fastpath(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            disable_memory=True,
            feed_forward_layers=True,
            feed_forward_kind="linear",
        )
        token_ids = torch.randint(0, 32, (2, 5))
        fast_query_val = torch.zeros(2, 5, 8)

        with patch.object(
            model,
            "_supports_nomemory_exact_stack_fastpath",
            return_value=False,
        ) as supports_noffn_mock, patch.object(
            model,
            "_supports_nomemory_exact_stack_ffn_fastpath",
            return_value=False,
        ) as supports_ffn_mock, patch.object(
            model,
            "_supports_nomemory_exact_stack_linear_fastpath",
            return_value=True,
        ) as supports_linear_mock, patch.object(
            model,
            "_run_nomemory_exact_stack_linear_fastpath",
            return_value=fast_query_val,
        ) as run_mock:
            logits = model(token_ids)

        supports_noffn_mock.assert_called_once()
        supports_ffn_mock.assert_called_once()
        supports_linear_mock.assert_called_once()
        run_mock.assert_called_once()
        self.assertEqual(logits.shape, (2, 5, 32))

    def test_nomemory_exact_stack_native_fastpath_matches_reference_and_gradients(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for native nomemory exact stack parity coverage.")
        if not nomemory_causal_stack_fused_native_available("cuda"):
            self.skipTest("Native nomemory exact stack fused op is unavailable on CUDA.")
        if not native_supports("nomemory_causal_stack_fused_trace"):
            self.skipTest("Native nomemory exact stack trace op is unavailable.")
        if not native_supports("nomemory_causal_stack_fused_backward_cuda"):
            self.skipTest("Native nomemory exact stack backward op is unavailable.")

        device = torch.device("cuda")
        torch.manual_seed(23)
        input_ids = torch.randint(0, 64, (2, 8), device=device)
        target = torch.randint(0, 64, (2, 8), device=device)

        def _run_step(model: CausalHierarchicalMemoryLM, *, fastpath_enabled: bool) -> tuple[torch.Tensor, float, dict[str, int], dict[str, torch.Tensor]]:
            env = {
                "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1" if fastpath_enabled else "0",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1" if fastpath_enabled else "0",
            }
            model.zero_grad(set_to_none=True)
            model.reset_dense_apply_stats()
            with patch.dict(os.environ, env, clear=False):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
            sequence_pairwise = model.s_module.sequence_layers[0].pairwise_fn
            prediction_pairwise = model.prediction_layers[0].pairwise_fn
            if isinstance(sequence_pairwise, MultiHeadPairwise):
                sequence_pairwise = sequence_pairwise.heads[0]
            if isinstance(prediction_pairwise, MultiHeadPairwise):
                prediction_pairwise = prediction_pairwise.heads[0]
            grads = {
                "s_prediction": model.s_prediction_proj.weight.grad.detach().cpu(),
                "prediction_input_norm": model.prediction_input_norm.weight.grad.detach().cpu(),
                "sequence_core": sequence_pairwise.weight.grad.detach().cpu(),
                "prediction_core": prediction_pairwise.weight.grad.detach().cpu(),
            }
            return logits.detach().cpu(), float(loss.detach()), model.dense_apply_stats(), grads

        def _run_token_val_grad_parity(model: CausalHierarchicalMemoryLM) -> torch.Tensor:
            token_val = model.s_module.token_embedding(input_ids)
            token_val = token_val + model.s_module.position_encoding(
                input_ids.shape[1],
                device=token_val.device,
                dtype=token_val.dtype,
            ).unsqueeze(0)
            token_val = model.s_module.sequence_input_norm(token_val)

            sequence_specs = tuple(
                (
                    int(model._nomemory_exact_edge_compress_kind(propagation) or 0),
                    int(propagation.window or 0),
                    int(propagation.target_block_size or (input_ids.shape[1] + 1)),
                    int(propagation.source_block_size or (input_ids.shape[1] + 1)),
                )
                for propagation in model.s_module.sequence_layers
            )
            prediction_specs = tuple(
                (
                    int(model._nomemory_exact_edge_compress_kind(propagation) or 0),
                    int(propagation.window or 0),
                    int(propagation.target_block_size or input_ids.shape[1]),
                    int(propagation.source_block_size or input_ids.shape[1]),
                )
                for propagation in model.prediction_layers
            )
            sequence_tensors = tuple(
                tensor
                for propagation, norm in zip(model.s_module.sequence_layers, model.s_module.sequence_norms)
                for spec in (pairwise_kernel_spec(propagation.pairwise_fn),)
                for tensor in (
                    model._tensor_or_empty(spec.in_weight, token_val),
                    model._tensor_or_empty(spec.out_weight, token_val),
                    spec.weight.to(device=token_val.device, dtype=token_val.dtype),
                    model._tensor_or_empty(spec.bias, token_val),
                    model.s_module._norm_param_or_empty(norm, "weight", token_val),
                    model.s_module._norm_param_or_empty(norm, "bias", token_val),
                )
            )
            prediction_tensors = tuple(
                tensor
                for propagation, norm in zip(model.prediction_layers, model.prediction_norms)
                for spec in (pairwise_kernel_spec(propagation.pairwise_fn),)
                for tensor in (
                    model._tensor_or_empty(spec.in_weight, token_val),
                    model._tensor_or_empty(spec.out_weight, token_val),
                    spec.weight.to(device=token_val.device, dtype=token_val.dtype),
                    model._tensor_or_empty(spec.bias, token_val),
                    model._norm_param_or_empty(norm, "weight", token_val),
                    model._norm_param_or_empty(norm, "bias", token_val),
                )
            )

            if not all(
                isinstance(propagation.pairwise_fn, LowRankBilinearPairwise)
                for propagation in (*model.s_module.sequence_layers, *model.prediction_layers)
            ):
                token_val_ref = token_val.detach().clone().requires_grad_(True)
                anchor_val = model.s_module.anchor_val.expand(input_ids.shape[0], 1, -1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                anchor_state = model.s_module.anchor_state.expand(input_ids.shape[0], 1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                token_state = model.value_to_state(token_val_ref).squeeze(-1)
                if model.direction_only_values:
                    anchor_state = softsign_state(anchor_state)
                    token_state = softsign_state(token_state)
                sequence_layer = Layer(
                    dim=model.dim,
                    num_nodes=input_ids.shape[1] + 1,
                    state=torch.cat((anchor_state, token_state), dim=1),
                    val=torch.cat((anchor_val, token_val_ref), dim=1),
                )
                for propagation, norm, ffn in zip(
                    model.s_module.sequence_layers,
                    model.s_module.sequence_norms,
                    model.s_module.sequence_ffns,
                ):
                    delta = propagation.compute_delta(sequence_layer)
                    if model.s_module._can_use_dense_apply_fastpath(sequence_layer, propagation):
                        sequence_layer = model.s_module._apply_dense_delta_fastpath(
                            sequence_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        sequence_layer = apply_delta(
                            sequence_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=model.direction_only_values,
                        )
                    sequence_layer = sequence_layer.with_tensors(val=ffn(sequence_layer.val))
                aligned_s = sequence_layer.val[:, 1:, :]
                query_layer = model._memoryless_query_layer(aligned_s)
                for propagation, norm, ffn in zip(
                    model.prediction_layers,
                    model.prediction_norms,
                    model.prediction_ffns,
                ):
                    delta = propagation.compute_delta(query_layer)
                    if model._can_use_dense_apply_fastpath(query_layer, propagation):
                        query_layer = model._apply_dense_delta_fastpath(
                            query_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        query_layer = apply_delta(
                            query_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=model.direction_only_values,
                        )
                    query_layer = query_layer.with_tensors(val=ffn(query_layer.val))
                query_layer.val.square().mean().backward()
                return token_val_ref.grad.detach().cpu()

            token_val_fast = token_val.detach().clone().requires_grad_(True)
            query_fast = _nomemory_exact_stack_fused(
                token_val_fast,
                anchor_state=model.s_module.anchor_state.expand(input_ids.shape[0], 1).to(
                    device=token_val_fast.device,
                    dtype=token_val_fast.dtype,
                ),
                anchor_val=model.s_module.anchor_val.expand(input_ids.shape[0], 1, -1).to(
                    device=token_val_fast.device,
                    dtype=token_val_fast.dtype,
                ),
                s_prediction_weight=model.s_prediction_proj.weight.to(
                    device=token_val_fast.device,
                    dtype=token_val_fast.dtype,
                ),
                prediction_input_norm_weight=model._norm_param_or_empty(
                    model.prediction_input_norm,
                    "weight",
                    token_val_fast,
                ),
                prediction_input_norm_bias=model._norm_param_or_empty(
                    model.prediction_input_norm,
                    "bias",
                    token_val_fast,
                ),
                sequence_tensors=sequence_tensors,
                prediction_tensors=prediction_tensors,
                sequence_specs=sequence_specs,
                prediction_specs=prediction_specs,
                state_activation_name="softsign" if model.direction_only_values else "signed_softmax",
            )
            loss_fast = query_fast.square().mean()
            loss_fast.backward()
            return token_val_fast.grad.detach().cpu()

        for direction_only_values in (False, True):
            with self.subTest(direction_only_values=direction_only_values):
                base_model = CausalHierarchicalMemoryLM(
                    vocab_size=64,
                    dim=32,
                    max_seq_len=16,
                    s_layers=6,
                    memory_slots=(8, 4),
                    prediction_layers=3,
                    memory_topk=2,
                    pairwise_rank=16,
                    route_rank=16,
                    pairwise_heads=4,
                    route_heads=4,
                    disable_memory=True,
                    feed_forward_layers=False,
                    scan_backend="native",
                    implementation="native",
                    direction_only_values=direction_only_values,
                ).to(device)
                reference_model = CausalHierarchicalMemoryLM(
                    vocab_size=64,
                    dim=32,
                    max_seq_len=16,
                    s_layers=6,
                    memory_slots=(8, 4),
                    prediction_layers=3,
                    memory_topk=2,
                    pairwise_rank=16,
                    route_rank=16,
                    pairwise_heads=4,
                    route_heads=4,
                    disable_memory=True,
                    feed_forward_layers=False,
                    scan_backend="native",
                    implementation="native",
                    direction_only_values=direction_only_values,
                ).to(device)
                reference_model.load_state_dict(base_model.state_dict())
                base_model.train()
                reference_model.train()

                with patch.dict(
                    os.environ,
                    {
                        "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1",
                        "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1",
                    },
                    clear=False,
                ):
                    self.assertTrue(
                        base_model._supports_nomemory_exact_stack_fastpath(
                            input_ids,
                            return_layers=False,
                        )
                    )

                fast_logits, fast_loss, fast_stats, fast_grads = _run_step(base_model, fastpath_enabled=True)
                reference_logits, reference_loss, reference_stats, reference_grads = _run_step(
                    reference_model,
                    fastpath_enabled=False,
                )
                token_val_grad = _run_token_val_grad_parity(base_model)
                reference_token_val = token_val_grad.new_zeros(token_val_grad.shape)
                token_val = reference_model.s_module.token_embedding(input_ids)
                token_val = token_val + reference_model.s_module.position_encoding(
                    input_ids.shape[1],
                    device=token_val.device,
                    dtype=token_val.dtype,
                ).unsqueeze(0)
                token_val = reference_model.s_module.sequence_input_norm(token_val)
                token_val_ref = token_val.detach().clone().requires_grad_(True)
                anchor_val = reference_model.s_module.anchor_val.expand(input_ids.shape[0], 1, -1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                anchor_state = reference_model.s_module.anchor_state.expand(input_ids.shape[0], 1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                token_state = reference_model.value_to_state(token_val_ref).squeeze(-1)
                if direction_only_values:
                    anchor_state = softsign_state(anchor_state)
                    token_state = softsign_state(token_state)
                sequence_layer = Layer(
                    dim=reference_model.dim,
                    num_nodes=input_ids.shape[1] + 1,
                    state=torch.cat((anchor_state, token_state), dim=1),
                    val=torch.cat((anchor_val, token_val_ref), dim=1),
                )
                for propagation, norm, ffn in zip(
                    reference_model.s_module.sequence_layers,
                    reference_model.s_module.sequence_norms,
                    reference_model.s_module.sequence_ffns,
                ):
                    delta = propagation.compute_delta(sequence_layer)
                    if reference_model.s_module._can_use_dense_apply_fastpath(sequence_layer, propagation):
                        sequence_layer = reference_model.s_module._apply_dense_delta_fastpath(
                            sequence_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        sequence_layer = apply_delta(
                            sequence_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=direction_only_values,
                        )
                    sequence_layer = sequence_layer.with_tensors(val=ffn(sequence_layer.val))
                aligned_s = sequence_layer.val[:, 1:, :]
                query_layer = reference_model._memoryless_query_layer(aligned_s)
                for propagation, norm, ffn in zip(
                    reference_model.prediction_layers,
                    reference_model.prediction_norms,
                    reference_model.prediction_ffns,
                ):
                    delta = propagation.compute_delta(query_layer)
                    if reference_model._can_use_dense_apply_fastpath(query_layer, propagation):
                        query_layer = reference_model._apply_dense_delta_fastpath(
                            query_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        query_layer = apply_delta(
                            query_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=direction_only_values,
                        )
                    query_layer = query_layer.with_tensors(val=ffn(query_layer.val))
                query_layer.val.square().mean().backward()
                reference_token_val = token_val_ref.grad.detach().cpu()

                self.assertTrue(torch.allclose(fast_logits, reference_logits, atol=1e-4, rtol=1e-4))
                self.assertAlmostEqual(fast_loss, reference_loss, places=5)
                for name in fast_grads:
                    self.assertTrue(
                        torch.allclose(fast_grads[name], reference_grads[name], atol=1e-4, rtol=1e-4),
                        msg=f"Gradient mismatch for {name} with direction_only_values={direction_only_values}.",
                    )
                self.assertTrue(
                    torch.allclose(token_val_grad, reference_token_val, atol=1e-4, rtol=1e-4),
                    msg=f"token_val gradient mismatch with direction_only_values={direction_only_values}.",
                )
                self.assertEqual(fast_stats["nomemory_exact_stack_fastpath_calls"], 1)
                self.assertEqual(fast_stats["sequence_native_calls"], 6)
                self.assertEqual(fast_stats["prediction_native_calls"], 3)
                self.assertEqual(reference_stats["nomemory_exact_stack_fastpath_calls"], 0)

    def test_nomemory_exact_stack_linear_fastpath_matches_reference_and_gradients(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for native nomemory exact stack linear parity coverage.")

        device = torch.device("cuda")
        torch.manual_seed(23)
        input_ids = torch.randint(0, 64, (2, 8), device=device)
        target = torch.randint(0, 64, (2, 8), device=device)

        def _run_step(
            model: CausalHierarchicalMemoryLM,
            *,
            fastpath_enabled: bool,
        ) -> tuple[torch.Tensor, float, dict[str, int], dict[str, torch.Tensor]]:
            env = {
                "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1" if fastpath_enabled else "0",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1" if fastpath_enabled else "0",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_LINEAR_FASTPATH": "1" if fastpath_enabled else "0",
            }
            model.zero_grad(set_to_none=True)
            model.reset_dense_apply_stats()
            with patch.dict(os.environ, env, clear=False):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
            grads = {
                "s_prediction": model.s_prediction_proj.weight.grad.detach().cpu(),
                "prediction_input_norm": model.prediction_input_norm.weight.grad.detach().cpu(),
                "sequence_linear_norm": model.s_module.sequence_ffns[0].input_norm.weight.grad.detach().cpu(),
                "sequence_linear": model.s_module.sequence_ffns[0].linear.weight.grad.detach().cpu(),
                "prediction_linear_norm": model.prediction_ffns[0].input_norm.weight.grad.detach().cpu(),
                "prediction_linear": model.prediction_ffns[0].linear.weight.grad.detach().cpu(),
            }
            return logits.detach().cpu(), float(loss.detach()), model.dense_apply_stats(), grads

        def _run_token_val_grad_parity(model: CausalHierarchicalMemoryLM) -> torch.Tensor:
            token_val = model.s_module.token_embedding(input_ids)
            token_val = token_val + model.s_module.position_encoding(
                input_ids.shape[1],
                device=token_val.device,
                dtype=token_val.dtype,
            ).unsqueeze(0)
            token_val = model.s_module.sequence_input_norm(token_val)

            sequence_specs = tuple(
                (
                    int(model._nomemory_exact_edge_compress_kind(propagation) or 0),
                    int(propagation.window or 0),
                    int(propagation.target_block_size or (input_ids.shape[1] + 1)),
                    int(propagation.source_block_size or (input_ids.shape[1] + 1)),
                )
                for propagation in model.s_module.sequence_layers
            )
            prediction_specs = tuple(
                (
                    int(model._nomemory_exact_edge_compress_kind(propagation) or 0),
                    int(propagation.window or 0),
                    int(propagation.target_block_size or input_ids.shape[1]),
                    int(propagation.source_block_size or input_ids.shape[1]),
                )
                for propagation in model.prediction_layers
            )
            sequence_tensors = tuple(
                tensor
                for propagation, norm in zip(model.s_module.sequence_layers, model.s_module.sequence_norms)
                for spec in (pairwise_kernel_spec(propagation.pairwise_fn),)
                for tensor in (
                    model._tensor_or_empty(spec.in_weight, token_val),
                    model._tensor_or_empty(spec.out_weight, token_val),
                    spec.weight.to(device=token_val.device, dtype=token_val.dtype),
                    model._tensor_or_empty(spec.bias, token_val),
                    model.s_module._norm_param_or_empty(norm, "weight", token_val),
                    model.s_module._norm_param_or_empty(norm, "bias", token_val),
                )
            )
            prediction_tensors = tuple(
                tensor
                for propagation, norm in zip(model.prediction_layers, model.prediction_norms)
                for spec in (pairwise_kernel_spec(propagation.pairwise_fn),)
                for tensor in (
                    model._tensor_or_empty(spec.in_weight, token_val),
                    model._tensor_or_empty(spec.out_weight, token_val),
                    spec.weight.to(device=token_val.device, dtype=token_val.dtype),
                    model._tensor_or_empty(spec.bias, token_val),
                    model._norm_param_or_empty(norm, "weight", token_val),
                    model._norm_param_or_empty(norm, "bias", token_val),
                )
            )

            if not all(
                isinstance(propagation.pairwise_fn, LowRankBilinearPairwise)
                for propagation in (*model.s_module.sequence_layers, *model.prediction_layers)
            ):
                token_val_ref = token_val.detach().clone().requires_grad_(True)
                anchor_val = model.s_module.anchor_val.expand(input_ids.shape[0], 1, -1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                anchor_state = model.s_module.anchor_state.expand(input_ids.shape[0], 1).to(
                    device=token_val_ref.device,
                    dtype=token_val_ref.dtype,
                )
                token_state = model.value_to_state(token_val_ref).squeeze(-1)
                if model.direction_only_values:
                    anchor_state = softsign_state(anchor_state)
                    token_state = softsign_state(token_state)
                sequence_layer = Layer(
                    dim=model.dim,
                    num_nodes=input_ids.shape[1] + 1,
                    state=torch.cat((anchor_state, token_state), dim=1),
                    val=torch.cat((anchor_val, token_val_ref), dim=1),
                )
                for propagation, norm, ffn in zip(
                    model.s_module.sequence_layers,
                    model.s_module.sequence_norms,
                    model.s_module.sequence_ffns,
                ):
                    delta = propagation.compute_delta(sequence_layer)
                    if model.s_module._can_use_dense_apply_fastpath(sequence_layer, propagation):
                        sequence_layer = model.s_module._apply_dense_delta_fastpath(
                            sequence_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        sequence_layer = apply_delta(
                            sequence_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=model.direction_only_values,
                        )
                    sequence_layer = sequence_layer.with_tensors(val=ffn(sequence_layer.val))
                aligned_s = sequence_layer.val[:, 1:, :]
                query_layer = model._memoryless_query_layer(aligned_s)
                for propagation, norm, ffn in zip(
                    model.prediction_layers,
                    model.prediction_norms,
                    model.prediction_ffns,
                ):
                    delta = propagation.compute_delta(query_layer)
                    if model._can_use_dense_apply_fastpath(query_layer, propagation):
                        query_layer = model._apply_dense_delta_fastpath(
                            query_layer,
                            delta.delta_state,
                            delta.delta_val,
                            norm,
                            propagation,
                        )
                    else:
                        query_layer = apply_delta(
                            query_layer,
                            delta,
                            residual=True,
                            val_norm=norm,
                            direction_only_values=model.direction_only_values,
                        )
                    query_layer = query_layer.with_tensors(val=ffn(query_layer.val))
                query_layer.val.square().mean().backward()
                return token_val_ref.grad.detach().cpu()

            token_val_fast = token_val.detach().clone().requires_grad_(True)
            batch_size = input_ids.shape[0]
            anchor_val = model.s_module.anchor_val.expand(batch_size, 1, -1).to(
                device=token_val_fast.device,
                dtype=token_val_fast.dtype,
            )
            anchor_state = model.s_module.anchor_state.expand(batch_size, 1).to(
                device=token_val_fast.device,
                dtype=token_val_fast.dtype,
            )
            token_state = model.value_to_state(token_val_fast).squeeze(-1)
            if model.direction_only_values:
                anchor_state = softsign_state(anchor_state)
                token_state = softsign_state(token_state)
            sequence_state = torch.cat((anchor_state, token_state), dim=1)
            sequence_val = torch.cat((anchor_val, token_val_fast), dim=1)
            state_activation_name = "softsign" if model.direction_only_values else "signed_softmax"
            sequence_state, sequence_val = _nomemory_exact_run_layers_with_postmix(
                sequence_state,
                sequence_val,
                layer_tensors=sequence_tensors,
                layer_specs=sequence_specs,
                postmix_modules=model.s_module.sequence_ffns,
                state_activation_name=state_activation_name,
                checkpoint_layers=False,
            )
            aligned_s = sequence_val[:, 1:, :]
            query_fast = _nomemory_exact_prediction_forward_with_postmix(
                aligned_s,
                s_prediction_weight=model.s_prediction_proj.weight.to(
                    device=token_val_fast.device,
                    dtype=token_val_fast.dtype,
                ),
                prediction_input_norm_weight=model._norm_param_or_empty(
                    model.prediction_input_norm,
                    "weight",
                    token_val_fast,
                ),
                prediction_input_norm_bias=model._norm_param_or_empty(
                    model.prediction_input_norm,
                    "bias",
                    token_val_fast,
                ),
                prediction_tensors=prediction_tensors,
                prediction_specs=prediction_specs,
                postmix_modules=model.prediction_ffns,
                state_activation_name=state_activation_name,
                checkpoint_layers=False,
            )
            loss_fast = query_fast.square().mean()
            loss_fast.backward()
            return token_val_fast.grad.detach().cpu()

        base_model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=32,
            max_seq_len=16,
            s_layers=6,
            memory_slots=(8, 4),
            prediction_layers=3,
            memory_topk=2,
            pairwise_rank=16,
            route_rank=16,
            pairwise_heads=4,
            route_heads=4,
            disable_memory=True,
            feed_forward_layers=True,
            feed_forward_kind="linear",
            scan_backend="native",
            implementation="native",
            direction_only_values=True,
        ).to(device)
        reference_model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=32,
            max_seq_len=16,
            s_layers=6,
            memory_slots=(8, 4),
            prediction_layers=3,
            memory_topk=2,
            pairwise_rank=16,
            route_rank=16,
            pairwise_heads=4,
            route_heads=4,
            disable_memory=True,
            feed_forward_layers=True,
            feed_forward_kind="linear",
            scan_backend="native",
            implementation="native",
            direction_only_values=True,
        ).to(device)
        reference_model.load_state_dict(base_model.state_dict())
        base_model.train()
        reference_model.train()

        with patch.dict(
            os.environ,
            {
                "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_LINEAR_FASTPATH": "1",
            },
            clear=False,
        ):
            self.assertTrue(
                base_model._supports_nomemory_exact_stack_linear_fastpath(
                    input_ids,
                    return_layers=False,
                )
            )

        fast_logits, fast_loss, fast_stats, fast_grads = _run_step(base_model, fastpath_enabled=True)
        reference_logits, reference_loss, reference_stats, reference_grads = _run_step(
            reference_model,
            fastpath_enabled=False,
        )
        token_val_grad = _run_token_val_grad_parity(base_model)
        reference_token_val = token_val_grad.new_zeros(token_val_grad.shape)
        token_val = reference_model.s_module.token_embedding(input_ids)
        token_val = token_val + reference_model.s_module.position_encoding(
            input_ids.shape[1],
            device=token_val.device,
            dtype=token_val.dtype,
        ).unsqueeze(0)
        token_val = reference_model.s_module.sequence_input_norm(token_val)
        token_val_ref = token_val.detach().clone().requires_grad_(True)
        anchor_val = reference_model.s_module.anchor_val.expand(input_ids.shape[0], 1, -1).to(
            device=token_val_ref.device,
            dtype=token_val_ref.dtype,
        )
        anchor_state = reference_model.s_module.anchor_state.expand(input_ids.shape[0], 1).to(
            device=token_val_ref.device,
            dtype=token_val_ref.dtype,
        )
        token_state = reference_model.value_to_state(token_val_ref).squeeze(-1)
        anchor_state = softsign_state(anchor_state)
        token_state = softsign_state(token_state)
        sequence_layer = Layer(
            dim=reference_model.dim,
            num_nodes=input_ids.shape[1] + 1,
            state=torch.cat((anchor_state, token_state), dim=1),
            val=torch.cat((anchor_val, token_val_ref), dim=1),
        )
        for propagation, norm, ffn in zip(
            reference_model.s_module.sequence_layers,
            reference_model.s_module.sequence_norms,
            reference_model.s_module.sequence_ffns,
        ):
            delta = propagation.compute_delta(sequence_layer)
            if reference_model.s_module._can_use_dense_apply_fastpath(sequence_layer, propagation):
                sequence_layer = reference_model.s_module._apply_dense_delta_fastpath(
                    sequence_layer,
                    delta.delta_state,
                    delta.delta_val,
                    norm,
                    propagation,
                )
            else:
                sequence_layer = apply_delta(
                    sequence_layer,
                    delta,
                    residual=True,
                    val_norm=norm,
                    direction_only_values=True,
                )
            sequence_layer = sequence_layer.with_tensors(val=ffn(sequence_layer.val))
        aligned_s = sequence_layer.val[:, 1:, :]
        query_layer = reference_model._memoryless_query_layer(aligned_s)
        for propagation, norm, ffn in zip(
            reference_model.prediction_layers,
            reference_model.prediction_norms,
            reference_model.prediction_ffns,
        ):
            delta = propagation.compute_delta(query_layer)
            if reference_model._can_use_dense_apply_fastpath(query_layer, propagation):
                query_layer = reference_model._apply_dense_delta_fastpath(
                    query_layer,
                    delta.delta_state,
                    delta.delta_val,
                    norm,
                    propagation,
                )
            else:
                query_layer = apply_delta(
                    query_layer,
                    delta,
                    residual=True,
                    val_norm=norm,
                    direction_only_values=True,
                )
            query_layer = query_layer.with_tensors(val=ffn(query_layer.val))
        query_layer.val.square().mean().backward()
        reference_token_val = token_val_ref.grad.detach().cpu()

        self.assertTrue(torch.allclose(fast_logits, reference_logits, atol=1e-4, rtol=1e-4))
        self.assertAlmostEqual(fast_loss, reference_loss, places=5)
        for name in fast_grads:
            self.assertTrue(
                torch.allclose(fast_grads[name], reference_grads[name], atol=1e-4, rtol=1e-4),
                msg=f"Gradient mismatch for {name}.",
            )
        self.assertTrue(
            torch.allclose(token_val_grad, reference_token_val, atol=1e-4, rtol=1e-4),
            msg="token_val gradient mismatch for linear fastpath.",
        )
        self.assertEqual(fast_stats["nomemory_exact_stack_linear_fastpath_calls"], 1)
        self.assertEqual(fast_stats["sequence_native_calls"], 6)
        self.assertEqual(fast_stats["prediction_native_calls"], 3)
        self.assertEqual(reference_stats["nomemory_exact_stack_linear_fastpath_calls"], 0)

    def test_nomemory_exact_stack_ffn_native_fastpath_matches_reference_and_gradients(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for native nomemory exact stack FFN parity coverage.")
        if not nomemory_causal_stack_ffn_fused_native_available("cuda"):
            self.skipTest("Native nomemory exact stack FFN fused op is unavailable on CUDA.")
        if not native_supports("nomemory_causal_stack_ffn_fused_trace"):
            self.skipTest("Native nomemory exact stack FFN trace op is unavailable.")
        if not native_supports("nomemory_causal_stack_ffn_fused_backward_cuda"):
            self.skipTest("Native nomemory exact stack FFN backward op is unavailable.")

        device = torch.device("cuda")
        torch.manual_seed(23)
        input_ids = torch.randint(0, 64, (2, 8), device=device)
        target = torch.randint(0, 64, (2, 8), device=device)

        def _run_step(
            model: CausalHierarchicalMemoryLM,
            *,
            fastpath_enabled: bool,
        ) -> tuple[torch.Tensor, float, dict[str, int], dict[str, torch.Tensor]]:
            env = {
                "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1" if fastpath_enabled else "0",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1" if fastpath_enabled else "0",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FFN_FASTPATH": "1" if fastpath_enabled else "0",
            }
            model.zero_grad(set_to_none=True)
            model.reset_dense_apply_stats()
            with patch.dict(os.environ, env, clear=False):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss.backward()
            grads = {
                "s_prediction": model.s_prediction_proj.weight.grad.detach().cpu(),
                "prediction_input_norm": model.prediction_input_norm.weight.grad.detach().cpu(),
                "sequence_ffn_norm": model.s_module.sequence_ffns[0].input_norm.weight.grad.detach().cpu(),
                "sequence_ffn_in": model.s_module.sequence_ffns[0].net[0].weight.grad.detach().cpu(),
                "sequence_ffn_out": model.s_module.sequence_ffns[0].net[3].weight.grad.detach().cpu(),
                "prediction_ffn_norm": model.prediction_ffns[0].input_norm.weight.grad.detach().cpu(),
                "prediction_ffn_in": model.prediction_ffns[0].net[0].weight.grad.detach().cpu(),
                "prediction_ffn_out": model.prediction_ffns[0].net[3].weight.grad.detach().cpu(),
            }
            return logits.detach().cpu(), float(loss.detach()), model.dense_apply_stats(), grads

        base_model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=32,
            max_seq_len=16,
            s_layers=6,
            memory_slots=(8, 4),
            prediction_layers=3,
            memory_topk=2,
            pairwise_rank=16,
            route_rank=16,
            pairwise_heads=4,
            route_heads=4,
            disable_memory=True,
            feed_forward_layers=True,
            scan_backend="native",
            implementation="native",
            direction_only_values=True,
        ).to(device)
        reference_model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=32,
            max_seq_len=16,
            s_layers=6,
            memory_slots=(8, 4),
            prediction_layers=3,
            memory_topk=2,
            pairwise_rank=16,
            route_rank=16,
            pairwise_heads=4,
            route_heads=4,
            disable_memory=True,
            feed_forward_layers=True,
            scan_backend="native",
            implementation="native",
            direction_only_values=True,
        ).to(device)
        reference_model.load_state_dict(base_model.state_dict())
        base_model.train()
        reference_model.train()

        with patch.dict(
            os.environ,
            {
                "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING": "1",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FASTPATH": "1",
                "JAKAL_NET_ENABLE_EXPERIMENTAL_NOMEMORY_EXACT_STACK_FFN_FASTPATH": "1",
            },
            clear=False,
        ):
            self.assertTrue(
                base_model._supports_nomemory_exact_stack_ffn_fastpath(
                    input_ids,
                    return_layers=False,
                )
            )

        fast_logits, fast_loss, fast_stats, fast_grads = _run_step(base_model, fastpath_enabled=True)
        reference_logits, reference_loss, reference_stats, reference_grads = _run_step(
            reference_model,
            fastpath_enabled=False,
        )

        self.assertTrue(torch.allclose(fast_logits, reference_logits, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(fast_loss, reference_loss, places=6)
        for name in fast_grads:
            self.assertTrue(
                torch.allclose(fast_grads[name], reference_grads[name], atol=1e-5, rtol=1e-5),
                msg=f"Gradient mismatch for {name}.",
            )
        self.assertEqual(fast_stats["nomemory_exact_stack_ffn_fastpath_calls"], 1)
        self.assertEqual(fast_stats["sequence_native_calls"], 6)
        self.assertEqual(fast_stats["prediction_native_calls"], 3)
        self.assertEqual(reference_stats["nomemory_exact_stack_ffn_fastpath_calls"], 0)

    def test_direction_only_values_memoryless_query_uses_softsign_state(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            direction_only_values=True,
            feed_forward_layers=False,
        )
        aligned_s = torch.randn(2, 5, 16)

        query_layer = model._memoryless_query_layer(aligned_s)
        query_state_source = model.prediction_input_norm(model.s_prediction_proj(aligned_s))
        expected_state = softsign_state(model.value_to_state(query_state_source).squeeze(-1))

        self.assertTrue(torch.allclose(query_layer.state, expected_state))
        self.assertTrue(torch.equal(query_layer.val, query_state_source))

    def test_dense_apply_stats_track_sequence_and_prediction_calls(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            direction_only_values=True,
            feed_forward_layers=False,
        )
        model.reset_dense_apply_stats()
        input_ids = torch.randint(0, 64, (2, 8))

        _ = model(input_ids)

        stats = model.dense_apply_stats()
        self.assertEqual(stats["sequence_native_calls"] + stats["sequence_python_calls"], 1)
        self.assertEqual(stats["prediction_native_calls"] + stats["prediction_python_calls"], 1)

    def test_memory_train_eval_modes_select_dense_or_topk(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4),
            prediction_layers=1,
            memory_topk=3,
            memory_train_mode="dense",
            memory_eval_mode="topk",
            eval_topk=2,
            pairwise_rank=8,
            route_rank=8,
            feed_forward_layers=False,
        )
        aligned_s = torch.randn(2, 4, 16)
        memory_state = model.initialize_memory_state(2, device=aligned_s.device, dtype=aligned_s.dtype)

        model.train()
        train_packed = model._pack_native_scan_inputs(aligned_s, memory_state)
        self.assertEqual(train_packed["propagation_topks"], (0, 0))

        model.eval()
        eval_packed = model._pack_native_scan_inputs(aligned_s, memory_state)
        self.assertEqual(eval_packed["propagation_topks"], (2, 2))

    def test_multi_head_config_is_supported_by_native_fused_wrapper(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4, 2),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            pairwise_heads=2,
            route_heads=2,
            scan_backend="native",
            feed_forward_layers=False,
        )

        self.assertTrue(model._native_scan_supported_config())

    def test_diagonal_anchor_heads_are_supported_by_native_fused_wrapper(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4, 2),
            prediction_layers=1,
            memory_topk=2,
            pairwise_kind="diagonal_bilinear",
            route_kind="diagonal_bilinear",
            pairwise_rank=8,
            route_rank=8,
            pairwise_heads=4,
            route_heads=4,
            pairwise_anchor_heads=1,
            route_anchor_heads=1,
            pairwise_anchor_kind="diagonal_bilinear",
            route_anchor_kind="diagonal_bilinear",
            scan_backend="native",
            feed_forward_layers=False,
        )

        self.assertTrue(model._native_scan_supported_config())
        self.assertFalse(
            _native_scan_uses_legacy_low_rank_extension(
                "multihead_max_diagonal_bilinear_route",
                "multihead_max_diagonal_bilinear",
            )
        )

    def test_constant_one_anchor_heads_stay_on_low_rank_native_path(self) -> None:
        model = CausalHierarchicalMemoryLM(
            vocab_size=64,
            dim=16,
            max_seq_len=16,
            s_layers=1,
            memory_slots=(8, 4, 2),
            prediction_layers=1,
            memory_topk=2,
            pairwise_rank=8,
            route_rank=8,
            pairwise_heads=4,
            route_heads=4,
            pairwise_anchor_heads=1,
            route_anchor_heads=1,
            pairwise_anchor_kind="constant_one",
            route_anchor_kind="constant_one",
            scan_backend="native",
            feed_forward_layers=False,
        )
        aligned_s = torch.randn(2, 4, 16)
        memory_state = model.initialize_memory_state(2, device=aligned_s.device, dtype=aligned_s.dtype)

        packed = model._pack_native_scan_inputs(aligned_s, memory_state)

        self.assertTrue(model._native_scan_supported_config())
        self.assertEqual(packed["route_kind_name"], "multihead_max_low_rank_bilinear_route")
        self.assertEqual(packed["propagation_pairwise_kind"], "multihead_max_low_rank_bilinear")
        self.assertTrue(
            _native_scan_uses_legacy_low_rank_extension(
                packed["route_kind_name"],
                packed["propagation_pairwise_kind"],
            )
        )
        write_source = packed["write_source_weights"][0]
        write_source_by_head = write_source.reshape(write_source.shape[0], -1).abs().amax(dim=1)
        self.assertTrue(torch.any(write_source_by_head == 0).item())
        self.assertTrue(torch.any(torch.isclose(packed["write_biases"][0].reshape(-1), torch.ones(()))).item())

    def test_value_norm_state_projection_uses_vector_norm(self) -> None:
        projection = ValueNormStateProjection()
        val = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])

        state = projection(val)

        self.assertTrue(torch.allclose(state, torch.tensor([[[5.0], [13.0]]])))

    def test_multi_head_low_rank_scan_uses_legacy_extension_path(self) -> None:
        self.assertTrue(
            _native_scan_uses_legacy_low_rank_extension(
                "multihead_max_low_rank_bilinear_route",
                "multihead_max_low_rank_bilinear",
            )
        )

    def test_forward_supports_optional_knowledge_module(self) -> None:
        torch.manual_seed(11)
        knowledge_module = KModule(
            dim=8,
            num_nodes=10,
            route_rank=4,
            pairwise_rank=4,
            route_topk=4,
            propagation_topk=3,
            implementation="reference",
        )
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4, 2),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            knowledge_module=knowledge_module,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True, return_layers=True)

        self.assertIsInstance(output, MemoryScanOutput)
        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertIsNotNone(output.knowledge_state)
        assert output.knowledge_state is not None
        self.assertEqual(output.knowledge_state.val.shape, (2, 10, 8))
        recurrent_state = output.recurrent_state
        self.assertIsInstance(recurrent_state, ModelRecurrentState)
        self.assertEqual(recurrent_state.memory_state[0].val.shape, (2, 6, 8))
        assert recurrent_state.knowledge_state is not None
        self.assertEqual(recurrent_state.knowledge_state.val.shape, (2, 10, 8))

    def test_model_can_build_internal_knowledge_module(self) -> None:
        torch.manual_seed(12)
        model = CausalHierarchicalMemoryLM(
            vocab_size=32,
            dim=8,
            max_seq_len=12,
            s_layers=1,
            memory_slots=(6, 4, 2),
            prediction_layers=1,
            s_window=12,
            prediction_window=4,
            memory_topk=2,
            pairwise_rank=4,
            route_rank=4,
            knowledge_nodes=9,
            knowledge_route_topk=3,
            knowledge_propagation_topk=3,
            knowledge_propagation_layers=2,
        )
        token_ids = torch.randint(0, 32, (2, 5))

        output = model(token_ids, return_memory_state=True)

        self.assertIsNotNone(model.knowledge_module)
        self.assertIsInstance(output, MemoryScanOutput)
        self.assertIsNotNone(output.knowledge_state)

    def test_document_chunks_insert_continuation_prefix(self) -> None:
        chunks = make_document_chunks(
            content_ids=torch.tensor([10, 11, 12, 13, 14, 15, 16], dtype=torch.long),
            mode_token_id=2,
            seq_len=5,
            bos_token_id=1,
            cont_token_id=3,
            eos_token_id=4,
            pad_token_id=0,
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(torch.equal(chunks[0].context[:5], torch.tensor([1, 2, 10, 11, 12])))
        self.assertTrue(torch.equal(chunks[1].context[:5], torch.tensor([3, 2, 13, 14, 15])))
        self.assertTrue(torch.equal(chunks[2].context[:4], torch.tensor([3, 2, 16, 0])))
        self.assertEqual(int(chunks[0].target[4].item()), 3)
        self.assertEqual(int(chunks[2].target[2].item()), 4)
        self.assertFalse(chunks[0].is_continuation)
        self.assertTrue(chunks[1].is_continuation)

    def test_document_batcher_carries_within_document_and_resets_on_new_document(self) -> None:
        doc1 = TokenizedDocument(
            kind="text",
            source="doc1",
            token_count=5,
            chunks=(
                DocumentChunk(
                    context=torch.tensor([1, 2, 10, 0], dtype=torch.long),
                    target=torch.tensor([2, 10, 3, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=False,
                ),
                DocumentChunk(
                    context=torch.tensor([3, 2, 11, 0], dtype=torch.long),
                    target=torch.tensor([2, 11, 4, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=True,
                ),
            ),
        )
        doc2 = TokenizedDocument(
            kind="text",
            source="doc2",
            token_count=2,
            chunks=(
                DocumentChunk(
                    context=torch.tensor([1, 2, 20, 0], dtype=torch.long),
                    target=torch.tensor([2, 20, 4, 0], dtype=torch.long),
                    loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    is_continuation=False,
                ),
            ),
        )
        batcher = DocumentChunkBatcher((doc1, doc2), batch_size=1, device=torch.device("cpu"))
        batcher.current_doc[0] = 0
        batcher.current_chunk[0] = 0
        batcher.needs_reset[0] = False

        batch1 = batcher.next_batch()
        batch2 = batcher.next_batch()
        batch3 = batcher.next_batch()

        self.assertTrue(torch.equal(batch1.reset_mask, torch.tensor([False])))
        self.assertTrue(torch.equal(batch2.reset_mask, torch.tensor([False])))
        self.assertTrue(torch.equal(batch1.context[0], doc1.chunks[0].context))
        self.assertTrue(torch.equal(batch2.context[0], doc1.chunks[1].context))
        self.assertTrue(torch.equal(batch3.reset_mask, torch.tensor([True])))

    def test_pretokenized_bundle_roundtrip_uses_flat_storage(self) -> None:
        documents = (
            TokenizedDocument(
                kind="dialogue",
                source="doc1",
                token_count=6,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 2, 10, 11, 0, 0], dtype=torch.long),
                        target=torch.tensor([2, 10, 11, 3, 0, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                    DocumentChunk(
                        context=torch.tensor([3, 2, 12, 13, 14, 0], dtype=torch.long),
                        target=torch.tensor([2, 12, 13, 14, 4, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
                        is_continuation=True,
                    ),
                ),
            ),
            TokenizedDocument(
                kind="text",
                source="doc2",
                token_count=2,
                chunks=(
                    DocumentChunk(
                        context=torch.tensor([1, 5, 20, 0, 0, 0], dtype=torch.long),
                        target=torch.tensor([5, 20, 4, 0, 0, 0], dtype=torch.long),
                        loss_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                        is_continuation=False,
                    ),
                ),
            ),
        )
        corpus_info = {
            "special_tokens": {"pad": 0, "eos": 4, "cont": 3},
            "tokenized_summary": {"documents": 2, "chunks": 3, "tokens": 8},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.pt"
            save_pretokenized_bundle(
                bundle_path,
                documents=documents,
                vocab_size=16384,
                tokenizer_label="test",
                tokenizer_model_path=None,
                corpus_info=corpus_info,
            )
            raw_bundle = torch.load(bundle_path, map_location="cpu")
            self.assertEqual(raw_bundle["storage_format"], "flat_v2")
            self.assertEqual(int(raw_bundle["seq_len"]), 6)
            self.assertNotIn("target", str(raw_bundle))
            restored = load_pretokenized_bundle(bundle_path)

        self.assertEqual(len(restored["documents"]), len(documents))
        for restored_document, original_document in zip(restored["documents"], documents):
            self.assertEqual(restored_document.kind, original_document.kind)
            self.assertEqual(restored_document.source, original_document.source)
            self.assertEqual(restored_document.token_count, original_document.token_count)
            self.assertEqual(len(restored_document.chunks), len(original_document.chunks))
            for restored_chunk, original_chunk in zip(restored_document.chunks, original_document.chunks):
                self.assertTrue(torch.equal(restored_chunk.context, original_chunk.context))
                self.assertTrue(torch.equal(restored_chunk.target, original_chunk.target))
                self.assertTrue(torch.equal(restored_chunk.loss_mask, original_chunk.loss_mask))
                self.assertEqual(restored_chunk.is_continuation, original_chunk.is_continuation)


if __name__ == "__main__":
    unittest.main()
