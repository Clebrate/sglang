"""Paged addressing, offset causality, ancestry isolation and local-head TP."""

import unittest
from math import gcd
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.kernels.ops.attention.flash_attention_v3 import (
    _is_fa3_supported,
    flash_attn_with_kvcache,
)
from sglang.kernels.ops.attention.gr_paged_attention import gr_paged_decode_attention
from sglang.srt.beam_search.paged_kv import GRPagedForwardMetadata
from sglang.srt.layers.attention.gr_paged_backend import (
    GRPagedAttentionBackend,
    _torch_attention,
    torch_decode_attention,
    torch_prefill_attention,
    validate_gr_paged_configuration,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


def dense_attention(q, k, v, scale, offset=None):
    """Independent FP64 oracle, with explicit per-query/per-head legal keys."""
    result = torch.empty((*q.shape[:2], v.shape[-1]), dtype=q.dtype)
    groups = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        end = k.shape[0] if offset is None else offset + row + 1
        for head in range(q.shape[1]):
            key = k[:end, head // groups].double()
            value = v[:end, head // groups].double()
            scores = torch.mv(key, q[row, head].double()) * scale
            result[row, head] = torch.softmax(scores, 0) @ value
    return result


def prefill_references(q, k, v, scale, prefix):
    """Independent logical-KV FP64 reference and a precomputed BF16 error bound.

    Adapt FA2 v2.8.3.post2's forward-test criterion: at most twice the max
    error of a low-precision reference. Use FP64 instead of its FP32 high reference.
    This is distinct from the original atol=.002/rtol=.008 pointwise check.
    """
    groups = q.shape[1] // k.shape[1]
    key, value = (x.repeat_interleave(groups, dim=1) for x in (k, v))
    key_positions = torch.arange(k.shape[0], device=q.device)
    reference = torch.empty_like(q, dtype=torch.float64)
    key64 = key.double().permute(1, 2, 0)
    value64 = value.double().permute(1, 0, 2)
    for begin in range(0, q.shape[0], 64):
        end = min(begin + 64, q.shape[0])
        scores = q[begin:end].double().permute(1, 0, 2) @ key64 * scale
        positions = prefix + torch.arange(begin, end, device=q.device)
        scores.masked_fill_(key_positions[None, :] > positions[:, None], -torch.inf)
        reference[begin:end] = (scores.softmax(-1) @ value64).permute(1, 0, 2)

    # Scale K first and retain BF16 QK, softmax output and PV intermediates,
    # matching FA2's independent low-precision reference order.
    scores = torch.einsum("thd,shd->hts", q, key * scale)
    positions = prefix + torch.arange(q.shape[0], device=q.device)
    scores.masked_fill_(key_positions[None, :] > positions[:, None], -torch.inf)
    low = torch.einsum("hts,shd->thd", scores.softmax(-1).to(v.dtype), value)
    bound = 2 * (low.double() - reference.to(q.dtype).double()).abs().max().item()
    return reference, bound


def layout(context_k, context_v, beam_k, beam_v, page_size, ancestry):
    """Scatter known logical KV onto nonadjacent pages; poison padding."""
    context_len, kv_heads, dim = context_k.shape
    device = context_k.device
    steps, width = beam_k.shape[:2]
    context_page_count = (context_len + page_size - 1) // page_size
    step_page_count = (width + page_size - 1) // page_size
    page_ids = torch.arange(
        1, 2 * (context_page_count + steps * step_page_count), 2, device=device
    )
    context_pages = page_ids[:context_page_count]
    beam_pages = page_ids[context_page_count:].reshape(steps, step_page_count)
    pool_shape = (int(page_ids[-1] + 1) * page_size, kv_heads, dim)
    key_pool = torch.full(pool_shape, torch.nan, dtype=context_k.dtype, device=device)
    value_pool = torch.full_like(key_pool, torch.nan)
    positions = torch.arange(context_len, device=device)
    context_locs = (
        context_pages[positions // page_size] * page_size + positions % page_size
    )
    key_pool[context_locs] = context_k
    value_pool[context_locs] = context_v
    columns = torch.arange(width, device=device)
    for step in range(steps):
        slots = beam_pages[step, columns // page_size] * page_size + columns % page_size
        key_pool[slots] = beam_k[step]
        value_pool[slots] = beam_v[step]
    metadata = GRPagedForwardMetadata(
        page_size=page_size,
        context_len=context_len,
        beam_width=width,
        step=steps - 1,
        context_locs=context_locs,
        context_pages=context_pages,
        beam_pages=beam_pages,
        topk_indices=ancestry,
        out_cache_loc=slots,
    )
    return key_pool, value_pool, metadata


def cute_decode_attention(q, key_pool, value_pool, metadata, scale):
    """Adapt the existing single-group cases to the batched CUDA interface."""
    device = q.device
    return gr_paged_decode_attention(
        q=q.unsqueeze(0),
        key_pool=key_pool,
        value_pool=value_pool,
        context_pages=metadata.context_pages.to(device).unsqueeze(0),
        beam_pages=metadata.beam_pages.to(device).unsqueeze(0),
        ancestor_columns=metadata.topk_indices.to(device).unsqueeze(0),
        context_lens=torch.tensor(
            [metadata.context_len], dtype=torch.int32, device=device
        ),
        history_lens=torch.tensor(
            [metadata.step + 1], dtype=torch.int32, device=device
        ),
        page_size=metadata.page_size,
        scale=scale,
    ).squeeze(0)


class TestGRPagedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_offset_causal_extension(self):
        # Suffix query 0 is absolute position 3, not position 0. The later
        # value 1000 must remain invisible to it, including on a partial page.
        q = torch.zeros(2, 4, 2)
        key_pool = torch.full((32, 1, 2), torch.nan)
        value_pool = torch.full_like(key_pool, torch.nan)
        pages = torch.tensor([3, 6])
        slots = torch.tensor([12, 13, 14, 15, 24])
        key_pool[slots] = 0
        values = torch.tensor([1, 2, 3, 4, 1000.0])
        value_pool[slots] = values[:, None, None]
        actual = torch_prefill_attention(q, key_pool, value_pool, pages, 5, 4, 3, 1.0)
        expected = torch.tensor([2.5, 202.0])[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_shared_parent_without_sibling_or_padding_visibility(self):
        context = torch.zeros(3, 1, 2)
        beam_k = torch.zeros(2, 2, 1, 2)
        beam_v = torch.tensor([[2.0, 1000.0], [4.0, 8.0]])[:, :, None, None].expand_as(
            beam_k
        )
        ancestry = torch.tensor([[0, 0], [0, 1]])
        key, value, meta = layout(context, context, beam_k, beam_v, 4, ancestry)
        actual = torch_decode_attention(torch.zeros(2, 4, 2), key, value, meta, 1.0)
        expected = torch.tensor([1.2, 2.0])[:, None, None].expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_pages_and_tp_are_exact_with_fixed_arithmetic(self):
        for width in (2, 5, 17):
            with self.subTest(width=width):
                context_k = torch.randn(35, 4, 8).bfloat16()
                context_v = torch.randn_like(context_k)
                beam_k = torch.randn(3, width, 4, 8).bfloat16()
                beam_v = torch.randn_like(beam_k)
                q = torch.randn(width, 16, 8).bfloat16()
                ancestry = torch.stack(
                    (
                        torch.arange(width) % 2,
                        torch.zeros(width, dtype=torch.long),
                        torch.arange(width),
                    ),
                    1,
                )
                oracle = None
                for page_size in (1, 4, 16, 32):
                    key, value, meta = layout(
                        context_k, context_v, beam_k, beam_v, page_size, ancestry
                    )
                    actual = torch_decode_attention(q, key, value, meta, 8**-0.5)
                    if oracle is None:
                        oracle = actual
                    torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
                    for tp in (2, 4):
                        outputs = [
                            torch_decode_attention(qpart, kpart, vpart, meta, 8**-0.5)
                            for qpart, kpart, vpart in zip(
                                q.chunk(tp, 1), key.chunk(tp, 1), value.chunk(tp, 1)
                            )
                        ]
                        torch.testing.assert_close(
                            torch.cat(outputs, 1), oracle, atol=0, rtol=0
                        )
                for beam in range(width):
                    k = torch.cat((context_k, beam_k[torch.arange(3), ancestry[beam]]))
                    v = torch.cat((context_v, beam_v[torch.arange(3), ancestry[beam]]))
                    expected = dense_attention(q[beam : beam + 1], k, v, 8**-0.5)
                    torch.testing.assert_close(
                        oracle[beam : beam + 1], expected, atol=0.002, rtol=0.008
                    )

    def test_prefill_matches_dense_oracle_across_query_blocks(self):
        q = torch.randn(37, 8, 8)
        k = torch.randn(42, 2, 8)
        v = torch.randn_like(k)
        expected = dense_attention(q, k, v, 8**-0.5, offset=5)
        actual = _torch_attention(q, k, v, 8**-0.5, query_offset=5)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)

    def test_prefill_consumes_previously_written_kv(self):
        context_k = torch.zeros(3, 1, 2)
        context_v = torch.tensor([2.0, 4.0, 6.0])[:, None, None].expand_as(context_k)
        beam_k = torch.zeros(1, 2, 1, 2)
        beam_v = torch.tensor([8.0, 10.0])[None, :, None, None].expand_as(beam_k)
        key, value, meta = layout(
            context_k, context_v, beam_k, beam_v, 16, torch.tensor([[0], [1]])
        )
        backend = GRPagedAttentionBackend.__new__(GRPagedAttentionBackend)
        backend.page_size = 16
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(key, value)), set_kv_buffer=Mock()
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=4,
            head_dim=2,
            scaling=1.0,
            is_cross_attention=False,
            logit_cap=0,
            sliding_window_size=-1,
        )
        backend.forward_metadata = SimpleNamespace(
            context_pages=meta.context_pages, context_len=3, prefix_len=0
        )
        # Keep this CPU test focused on the fused writer's KV ownership.
        backend._prefill_attention = Mock(
            side_effect=lambda q, k, v, meta, scale: torch_prefill_attention(
                q,
                k,
                v,
                meta.context_pages,
                meta.context_len,
                16,
                meta.prefix_len,
                scale,
            )
        )
        expected = torch.tensor([2.0, 3.0, 4.0])
        q = torch.zeros(len(expected), 8)
        # Absent or stale input K/V must not replace a fused writer's data.
        for kv in (None, torch.full((len(expected), 1, 2), torch.nan)):
            with self.subTest(missing_kv=kv is None):
                actual = backend.forward_extend(
                    q, kv, kv, layer, SimpleNamespace(), save_kv_cache=False
                )
                torch.testing.assert_close(actual, expected[:, None].expand_as(q))
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()

    def test_configuration_scope(self):
        base = dict(
            enable_gr=True,
            attention_backend="gr_paged",
            page_size=16,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled",
        )
        for tp in (1, 2, 4):
            validate_gr_paged_configuration({**base, "tp_size": tp})
        for key, value in dict(
            enable_gr=False,
            page_size=0,
            tp_size=8,
            pp_size=2,
            dcp_size=2,
            attn_cp_size=2,
            enable_hierarchical_cache=True,
            enable_lora=True,
            chunked_prefill_size=256,
            disable_overlap_schedule=False,
            decode_attention_backend="flashinfer",
            cuda_graph_backend_prefill="full",
            quantization="fp8",
        ).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_gr_paged_configuration({**base, key: value})


@unittest.skipUnless(torch.cuda.is_available(), "FA3 prefill requires CUDA")
class TestFA3GRPagedPrefillAttention(unittest.TestCase):
    def setUp(self):
        if not _is_fa3_supported():
            self.skipTest("SGLang FA3 does not support this device")
        torch.manual_seed(17)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _backend(self, q, key, value, metadata, prefix):
        backend = GRPagedAttentionBackend.__new__(GRPagedAttentionBackend)
        backend.page_size = metadata.page_size
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=metadata.context_locs[None]
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=Mock(return_value=(key, value)), set_kv_buffer=Mock()
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[len(q)],
            extend_prefix_lens_cpu=[prefix],
            req_pool_indices=torch.tensor([0], device=q.device),
        )
        backend.init_forward_metadata(batch)
        meta = backend.forward_metadata
        self.assertEqual(meta.fa3_page_size, gcd(metadata.page_size, 64))
        self.assertEqual(meta.cache_seqlens.tolist(), [metadata.context_len])
        self.assertEqual(meta.cu_seqlens_q.tolist(), [0, len(q)])
        torch.testing.assert_close(meta.context_pages, metadata.context_pages)
        positions = torch.arange(metadata.context_len, device=q.device)
        slots = (
            meta.block_table[0, positions // meta.fa3_page_size].long()
            * meta.fa3_page_size
            + positions % meta.fa3_page_size
        )
        torch.testing.assert_close(slots, metadata.context_locs, atol=0, rtol=0)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=q.shape[1],
            head_dim=q.shape[2],
            scaling=q.shape[2] ** -0.5,
            is_cross_attention=False,
            logit_cap=0,
            sliding_window_size=-1,
        )
        return backend, layer, batch

    def _check_precision(self, actual, reference, bound):
        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all().item())
        rounded = reference.to(actual.dtype).double()
        error = (actual.double() - rounded).abs()
        strict_count = (error > 0.002 + 0.008 * rounded.abs()).sum().item()
        if strict_count:
            print(
                f"FA3 prefill {tuple(actual.shape)}: original pointwise tolerance "
                f"failed at {strict_count}/{actual.numel()} elements; "
                f"max error={error.max().item():.8g}, BF16 reference bound={bound:.8g}"
            )
        self.assertLessEqual(error.max().item(), bound)

    @torch.inference_mode()
    def test_pages_prefix_gqa_and_strided_pool(self):
        for dim in (64, 128):
            for page_size in (1, 2, 4, 8, 16, 32, 64, 96, 128, 256):
                for prefix in (0, 17):
                    with self.subTest(dim=dim, page_size=page_size, prefix=prefix):
                        length = page_size + 17
                        q = torch.randn(
                            length - prefix, 4, dim, device="cuda"
                        ).bfloat16()
                        k = torch.randn(length, 2, dim, device="cuda").bfloat16()
                        v = torch.randn_like(k)
                        reference, bound = prefill_references(
                            q, k, v, dim**-0.5, prefix
                        )
                        # Reuse the PR's fragmented layout; unused KV stays NaN.
                        unused = torch.full_like(k[:1, None], torch.nan)
                        key, value, metadata = layout(
                            k,
                            v,
                            unused,
                            unused,
                            page_size,
                            torch.zeros(1, 1, dtype=torch.int64, device=q.device),
                        )
                        if prefix:
                            pools = []
                            for pool in (key, value):
                                backing = pool.new_full(
                                    (len(pool), 2, 2, dim), torch.nan
                                )
                                backing[:, 0].copy_(pool)
                                pools.append(backing[:, 0])
                            key, value = pools
                            self.assertFalse(key.is_contiguous())
                        backend, layer, batch = self._backend(
                            q, key, value, metadata, prefix
                        )
                        actual = backend.forward_extend(
                            q.flatten(1), None, None, layer, batch, save_kv_cache=False
                        ).reshape_as(q)
                        self._check_precision(actual, reference, bound)
                        dense = flash_attn_with_kvcache(
                            q=q,
                            k_cache=k[None],
                            v_cache=v[None],
                            cache_seqlens=backend.forward_metadata.cache_seqlens,
                            cu_seqlens_q=backend.forward_metadata.cu_seqlens_q,
                            max_seqlen_q=len(q),
                            softmax_scale=layer.scaling,
                            causal=True,
                            num_splits=0,
                        )
                        torch.testing.assert_close(
                            actual, dense, atol=0.002, rtol=0.008
                        )
                        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()

    @torch.inference_mode()
    def test_offset_causality_and_previously_written_kv(self):
        for dim in (64, 128):
            for page_size in (16, 128, 256):
                with self.subTest(dim=dim, page_size=page_size):
                    k = torch.zeros(5, 1, dim, device="cuda", dtype=torch.bfloat16)
                    v = torch.tensor([1, 2, 3, 4, 1000], device=k.device, dtype=k.dtype)
                    v = v[:, None, None].expand_as(k)
                    unused = torch.full_like(k[:1, None], torch.nan)
                    key, value, metadata = layout(
                        k,
                        v,
                        unused,
                        unused,
                        page_size,
                        torch.zeros(1, 1, dtype=torch.int64, device=k.device),
                    )
                    q = k.new_zeros(2, 8 * dim)[:, : 4 * dim]
                    backend, layer, batch = self._backend(
                        q.reshape(2, 4, dim), key, value, metadata, 3
                    )
                    expected = q.new_tensor([2.5, 202])[:, None].expand_as(q)
                    for kv in (None, torch.full_like(k[3:], torch.nan)):
                        with self.subTest(missing_kv=kv is None):
                            actual = backend.forward_extend(
                                q, kv, kv, layer, batch, save_kv_cache=False
                            )
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    backend.token_to_kv_pool.set_kv_buffer.assert_not_called()

    @torch.inference_mode()
    def test_recsys_default_shape_precision(self):
        # Preserve the full shape/RNG of the two observed near-zero deviations.
        # Improved kernels may eliminate them: do not require a failure count
        # or hard-code historical FA3 outputs.
        torch.manual_seed(42)
        q = torch.randn(4700, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(4700, 8, 128, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        reference, bound = prefill_references(q, k, v, 128**-0.5, 0)
        unused = torch.full_like(k[:1, None], torch.nan)
        key, value, metadata = layout(
            k,
            v,
            unused,
            unused,
            16,
            torch.zeros(1, 1, dtype=torch.int64, device=q.device),
        )
        backend, layer, batch = self._backend(q, key, value, metadata, 0)
        actual = backend.forward_extend(
            q.flatten(1), None, None, layer, batch, save_kv_cache=False
        ).reshape_as(q)
        self._check_precision(actual, reference, bound)
        dense = flash_attn_with_kvcache(
            q=q,
            k_cache=k[None],
            v_cache=v[None],
            cache_seqlens=backend.forward_metadata.cache_seqlens,
            cu_seqlens_q=backend.forward_metadata.cu_seqlens_q,
            max_seqlen_q=len(q),
            softmax_scale=layer.scaling,
            causal=True,
            num_splits=0,
        )
        torch.testing.assert_close(actual, dense, atol=0.002, rtol=0.008)
        original = torch_prefill_attention(
            q, key, value, metadata.context_pages, 4700, 16, 0, layer.scaling
        )
        torch.testing.assert_close(
            original, reference.to(q.dtype), atol=0.002, rtol=0.008
        )


@unittest.skipUnless(torch.cuda.is_available(), "CuTe decode requires CUDA")
class TestCuTeGRPagedAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_shared_parent_without_sibling_or_padding_visibility(self):
        for dim in (64, 128):
            with self.subTest(dim=dim):
                context = torch.zeros(3, 1, dim, dtype=torch.bfloat16)
                beam_k = torch.zeros(2, 2, 1, dim, dtype=torch.bfloat16)
                beam_v = torch.tensor(
                    [[2.0, 1000.0], [4.0, 8.0]], dtype=torch.bfloat16
                )[:, :, None, None].expand_as(beam_k)
                ancestry = torch.tensor([[0, 0], [0, 1]])
                key, value, meta = layout(context, context, beam_k, beam_v, 4, ancestry)
                q = torch.zeros(2, 4, dim, device="cuda", dtype=torch.bfloat16)
                actual = cute_decode_attention(q, key.cuda(), value.cuda(), meta, 1.0)
                expected = torch.tensor([1.2, 2.0], dtype=q.dtype)[:, None, None]
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(
                    actual.cpu(), expected.expand(2, 4, dim), atol=0.002, rtol=0.008
                )

    def test_pages_and_tp_match_dense_oracle(self):
        for dim in (64, 128):
            for width in (2, 5, 17):
                context_k = torch.randn(35, 4, dim).bfloat16()
                context_v = torch.randn_like(context_k)
                beam_k = torch.randn(3, width, 4, dim).bfloat16()
                beam_v = torch.randn_like(beam_k)
                q = torch.randn(width, 16, dim).bfloat16()
                ancestry = torch.stack(
                    (
                        torch.arange(width) % 2,
                        torch.zeros(width, dtype=torch.long),
                        torch.arange(width),
                    ),
                    dim=1,
                )
                expected = torch.empty_like(q)
                for beam in range(width):
                    k = torch.cat((context_k, beam_k[torch.arange(3), ancestry[beam]]))
                    v = torch.cat((context_v, beam_v[torch.arange(3), ancestry[beam]]))
                    expected[beam : beam + 1] = dense_attention(
                        q[beam : beam + 1], k, v, dim**-0.5
                    )
                q_cuda = q.cuda()
                for page_size in (1, 4, 16, 32):
                    key, value, meta = layout(
                        context_k, context_v, beam_k, beam_v, page_size, ancestry
                    )
                    # Transfer before splitting heads so TP retains pool strides.
                    key_cuda, value_cuda = key.cuda(), value.cuda()
                    for tp in (1, 2, 4):
                        with self.subTest(
                            dim=dim, width=width, page_size=page_size, tp=tp
                        ):
                            outputs = []
                            for qpart, kpart, vpart in zip(
                                q_cuda.chunk(tp, 1),
                                key_cuda.chunk(tp, 1),
                                value_cuda.chunk(tp, 1),
                            ):
                                if tp > 1:
                                    self.assertFalse(qpart.is_contiguous())
                                    self.assertFalse(kpart.is_contiguous())
                                outputs.append(
                                    cute_decode_attention(
                                        qpart, kpart, vpart, meta, dim**-0.5
                                    )
                                )
                            actual = torch.cat(outputs, dim=1)
                            self.assertTrue(torch.isfinite(actual).all().item())
                            torch.testing.assert_close(
                                actual.cpu(), expected, atol=0.002, rtol=0.008
                            )

    def test_ragged_batch_and_partial_cta(self):
        context_lens, history_lens = (1, 17, 35), (1, 2, 3)
        batch, width, heads, kv_heads, dim, page_size = 3, 5, 2, 1, 64, 16
        q = torch.randn(batch, width, heads, dim).bfloat16()
        expected = torch.empty_like(q)
        context_pages = torch.zeros(batch, 3, dtype=torch.int64)
        beam_pages = torch.zeros(batch, 3, 1, dtype=torch.int64)
        ancestors = torch.zeros(batch, width, 3, dtype=torch.int64)
        keys, values, page_offset = [], [], 0
        for b, (context_len, steps) in enumerate(zip(context_lens, history_lens)):
            context_k = torch.randn(context_len, kv_heads, dim).bfloat16()
            context_v = torch.randn_like(context_k)
            beam_k = torch.randn(steps, width, kv_heads, dim).bfloat16()
            beam_v = torch.randn_like(beam_k)
            step = torch.arange(steps)
            ancestry = torch.arange(width)[:, None] // (2 ** (steps - 1 - step))
            key, value, meta = layout(
                context_k, context_v, beam_k, beam_v, page_size, ancestry
            )
            context_pages[b, : len(meta.context_pages)] = (
                meta.context_pages + page_offset
            )
            beam_pages[b, :steps] = meta.beam_pages + page_offset
            ancestors[b, :, :steps] = ancestry
            keys.append(key)
            values.append(value)
            page_offset += key.shape[0] // page_size
            for beam in range(width):
                k = torch.cat((context_k, beam_k[step, ancestry[beam]]))
                v = torch.cat((context_v, beam_v[step, ancestry[beam]]))
                expected[b, beam : beam + 1] = dense_attention(
                    q[b, beam : beam + 1], k, v, dim**-0.5
                )
        # 3 * 5 * 2 = 30 warps: the last CTA has two inactive warps.
        actual = gr_paged_decode_attention(
            q.cuda(),
            torch.cat(keys).cuda(),
            torch.cat(values).cuda(),
            context_pages.cuda(),
            beam_pages.cuda(),
            ancestors.cuda(),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device="cuda"),
            history_lens=torch.tensor(history_lens, dtype=torch.int32, device="cuda"),
            page_size=page_size,
            scale=dim**-0.5,
        )
        self.assertEqual(actual.shape, q.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.008)

    def test_decode_consumes_previously_written_kv(self):
        for dim in (64, 128):
            with self.subTest(dim=dim):
                context_k = torch.zeros(3, 1, dim, device="cuda", dtype=torch.bfloat16)
                context_v = torch.tensor(
                    [2.0, 4.0, 6.0], device="cuda", dtype=torch.bfloat16
                )[:, None, None].expand_as(context_k)
                beam_k = torch.zeros(1, 2, 1, dim, device="cuda", dtype=torch.bfloat16)
                beam_v = torch.tensor([8.0, 10.0], device="cuda", dtype=torch.bfloat16)[
                    None, :, None, None
                ].expand_as(beam_k)
                key, value, meta = layout(
                    context_k,
                    context_v,
                    beam_k,
                    beam_v,
                    16,
                    torch.tensor([[0], [1]], device="cuda"),
                )
                backend = GRPagedAttentionBackend.__new__(GRPagedAttentionBackend)
                backend.page_size = 16
                backend.token_to_kv_pool = SimpleNamespace(
                    get_kv_buffer=Mock(return_value=(key, value)), set_kv_buffer=Mock()
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE, gr_paged_metadata=meta
                )
                backend.init_forward_metadata(batch)
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=4,
                    head_dim=dim,
                    scaling=1.0,
                    is_cross_attention=False,
                    logit_cap=0,
                    sliding_window_size=-1,
                )
                q = torch.zeros(2, 12 * dim, device="cuda", dtype=torch.bfloat16)[
                    :, : 4 * dim
                ]
                expected = torch.tensor([5.0, 5.5], dtype=q.dtype)[:, None].expand(
                    q.shape
                )
                for kv in (None, torch.full_like(beam_k[0], torch.nan)):
                    with self.subTest(missing_kv=kv is None):
                        actual = backend.forward_decode(
                            q, kv, kv, layer, batch, save_kv_cache=False
                        )
                        self.assertTrue(torch.isfinite(actual).all().item())
                        torch.testing.assert_close(
                            actual.cpu(), expected, atol=0.002, rtol=0.008
                        )
                backend.token_to_kv_pool.set_kv_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
