import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives
from cutlass.memory import SmemAllocator
from quack import layout_utils

_LOG2_E = 1.4426950408889634
_COMPILED = {}


def gr_paged_decode_attention(
    q,
    key_pool,
    value_pool,
    context_pages,
    beam_pages,
    ancestor_columns,
    *,
    context_lens,
    history_lens,
    page_size,
    scale,
):
    inputs = (
        q,
        key_pool,
        value_pool,
        context_pages,
        beam_pages,
        ancestor_columns,
        context_lens,
        history_lens,
    )
    tensor_specs = tuple((t.dtype, t.shape, t.stride()) for t in inputs)
    # Base alignment can change even when a pool's shape and strides do not.
    aligned_kv = ((key_pool.data_ptr() | value_pool.data_ptr()) & 15) == 0
    cache_key = (q.device, page_size, aligned_kv, tensor_specs)
    compiled, num_splits = _COMPILED.get(cache_key, (None, 0))
    batch, width, q_heads, dim = tensor_specs[0][1]

    if compiled is None:
        kv_heads = key_pool.shape[1]
        # Target 256 CTAs on Hopper; limit splits using the context capacity.
        num_ctas = batch * q_heads * ((width + 63) // 64)
        context_capacity = context_pages.shape[1] * page_size
        num_splits = min(16, max(1, 256 // num_ctas), max(1, context_capacity // 256))
        async_kv = aligned_kv and all(
            t.stride(-1) == 1
            and all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1])
            for t in (key_pool, value_pool)
        )
        # TensorMap eligibility is fixed by the cached shape/stride signature.
        tma_kv = (
            async_kv
            and page_size == 16
            and dim in (64, 128)
            and all(
                all(0 < n <= 2**31 for n in t.shape)
                and all(0 < s * t.element_size() < 2**40 for s in t.stride()[:-1])
                for t in (key_pool, value_pool)
            )
        )

    out = torch.empty_like(q)
    partials = (
        out
        if num_splits == 1
        else torch.empty(
            (batch, width, q_heads, num_splits, dim + 2),
            dtype=torch.float32,
            device=q.device,
        )
    )

    tensors = (*inputs, partials, out)
    if compiled is None:
        with torch.cuda.device(q.device):
            cute_tensors = [
                from_dlpack(
                    t.detach(),
                    assumed_align=16 if tma_kv and i in (1, 2) else t.element_size(),
                    use_32bit_stride=False,
                    enable_tvm_ffi=True,
                )
                for i, t in enumerate(tensors)
            ]

            kernel = _GRPagedDecode(
                dim,
                q_heads // kv_heads,
                page_size,
                num_splits,
                async_kv,
                tma_kv,
            )

            compiled = cute.compile(
                kernel,
                *cute_tensors,
                cutlass.Float32(scale),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        _COMPILED[cache_key] = compiled, num_splits

    # TVM FFI selects the CUDA device and stream from the input tensors.
    compiled(*tensors, float(scale))

    return out


class _GRPagedDecode:
    def __init__(
        self,
        head_dim,
        q_heads_per_kv_head,
        page_size,
        num_splits,
        async_kv,
        tma_kv=False,
    ):
        self.head_dim = head_dim
        self.q_heads_per_kv_head = q_heads_per_kv_head
        self.page_size = page_size
        self.num_splits = num_splits
        self.combine_group_bits = (num_splits - 1).bit_length()
        self.combine_group_size = 1 << self.combine_group_bits
        self.async_kv = async_kv
        self.tma_kv = tma_kv
        self.num_stages = 2 if async_kv else 1
        self.tile_m = 64
        self.tile_n = 64
        self.num_threads = 256 if tma_kv else 128

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        key_pool: cute.Tensor,
        value_pool: cute.Tensor,
        context_pages: cute.Tensor,
        beam_pages: cute.Tensor,
        ancestor_columns: cute.Tensor,
        context_lens: cute.Tensor,
        history_lens: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        smem_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            cutlass.BFloat16,
        )

        sQ_layout = cute.tile_to_shape(
            smem_atom,
            (self.tile_m, self.head_dim),
            order=(0, 1),
        )

        sK_layout = cute.tile_to_shape(
            smem_atom,
            (self.tile_n, self.head_dim, self.num_stages),
            order=(0, 1, 2),
        )

        sV_layout = cute.tile_to_shape(
            smem_atom,
            (self.tile_n, self.head_dim, self.num_stages),
            order=(0, 1, 2),
        )

        tma_atom_k, tma_key = None, None
        tma_atom_v, tma_value = None, None
        if cutlass.const_expr(self.tma_kv):
            # A SW128 piece spans 64 BF16 values along D. Keep the pool's
            # original strides; this view only places the head mode last.
            tma_layout = cute.tile_to_shape(smem_atom, (16, 64), order=(0, 1))
            tma_atom_k, tma_key = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                cute.make_tensor(
                    key_pool.iterator, cute.select(key_pool.layout, [0, 2, 1])
                ),
                tma_layout,
                (16, 64),
            )
            tma_atom_v, tma_value = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                cute.make_tensor(
                    value_pool.iterator, cute.select(value_pool.layout, [0, 2, 1])
                ),
                tma_layout,
                (16, 64),
            )

        mma_qk_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (64, self.tile_n, 16),
            warpgroup.OperandSource.SMEM,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
        )

        tiled_mma_qk = cute.make_tiled_mma(
            cute.make_mma_atom(mma_qk_op),
            (1, 1, 1),
        )

        mma_pv_op = warpgroup.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (64, self.head_dim, 16),
            warpgroup.OperandSource.RMEM,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
        )

        tiled_mma_pv = cute.make_tiled_mma(
            cute.make_mma_atom(mma_pv_op),
            (1, 1, 1),
        )

        self.kernel(
            q,
            key_pool,
            value_pool,
            context_pages,
            beam_pages,
            ancestor_columns,
            context_lens,
            history_lens,
            partials,
            out,
            scale,
            tma_atom_k,
            tma_key,
            tma_atom_v,
            tma_value,
            sQ_layout,
            sK_layout,
            sV_layout,
            tiled_mma_qk,
            tiled_mma_pv,
        ).launch(
            grid=(
                cute.ceil_div(q.shape[1], self.tile_m),
                q.shape[2],
                q.shape[0] * self.num_splits,
            ),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=2 if self.tma_kv else 1,
            # Hopper rounds these preferences to 164/100 KiB for two CTAs.
            preferred_smem_carveout=(70 if self.head_dim == 128 else 40)
            if self.tma_kv
            else None,
            stream=stream,
        )

        if cutlass.const_expr(self.num_splits > 1):
            self.combine(partials, out).launch(
                grid=(cute.ceil_div(q.shape[1], 4), q.shape[2], q.shape[0]),
                block=(128, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        key_pool: cute.Tensor,
        value_pool: cute.Tensor,
        context_pages: cute.Tensor,
        beam_pages: cute.Tensor,
        ancestor_columns: cute.Tensor,
        context_lens: cute.Tensor,
        history_lens: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        scale: cutlass.Float32,
        tma_atom_k,
        tma_key,
        tma_atom_v,
        tma_value,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
    ):
        allocator = SmemAllocator()

        sQ = allocator.allocate_tensor(
            cutlass.BFloat16,
            sQ_layout.outer,
            byte_alignment=1024,
            swizzle=sQ_layout.inner,
        )

        sK = allocator.allocate_tensor(
            cutlass.BFloat16,
            sK_layout.outer,
            byte_alignment=1024,
            swizzle=sK_layout.inner,
        )

        sV = allocator.allocate_tensor(
            cutlass.BFloat16,
            sV_layout.outer,
            byte_alignment=1024,
            swizzle=sV_layout.inner,
        )

        sVt = cute.make_tensor(
            sV.iterator,
            cute.select(sV.layout, mode=[1, 0, 2]),
        )

        beam_tile, head, batch_split = cute.arch.block_idx()
        batch = batch_split // self.num_splits
        split = batch_split % self.num_splits
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)

        context_barriers = None
        empty_barriers = None
        if cutlass.const_expr(self.tma_kv):
            context_barriers = allocator.allocate_array(cutlass.Uint64, self.num_stages)
            empty_barriers = allocator.allocate_array(cutlass.Uint64, self.num_stages)
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.num_stages):
                        cute.arch.mbarrier_init(context_barriers + stage, 1)
                        cute.arch.mbarrier_init(empty_barriers + stage, 4)
                cpasync.prefetch_descriptor(tma_atom_k)
                cpasync.prefetch_descriptor(tma_atom_v)
            cute.arch.mbarrier_init_fence()

        beam_begin = beam_tile * self.tile_m
        kv_head = head // self.q_heads_per_kv_head

        context_len = context_lens[batch]
        history_len = history_lens[batch]
        num_context_tiles = cute.ceil_div(context_len, self.tile_n)
        first_tile = num_context_tiles * split // self.num_splits
        last_tile = num_context_tiles * (split + 1) // self.num_splits

        for i in cutlass.range_constexpr(
            self.tile_m * self.head_dim // self.num_threads
        ):
            linear = thread_idx + i * self.num_threads

            local_beam = linear // self.head_dim
            dim = linear % self.head_dim
            beam = beam_begin + local_beam

            q_value = cutlass.BFloat16(0.0)
            if beam < q.shape[1]:
                q_value = q[batch, beam, head, dim]

            sQ[local_beam, dim] = q_value

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        if cutlass.const_expr(self.tma_kv):
            if warp_idx >= 4:
                cute.arch.setmaxregister_decrease(32)
                if warp_idx == 4:
                    last_full_tile = last_tile
                    if last_tile * self.tile_n > context_len:
                        last_full_tile -= 1
                    for tile in cutlass.range(first_tile, last_full_tile):
                        write_stage = (tile - first_tile) % self.num_stages
                        empty_phase = ((tile - first_tile) // self.num_stages) % 2 ^ 1
                        cute.arch.mbarrier_wait(
                            empty_barriers + write_stage, empty_phase
                        )
                        self.load_context_tile(
                            key_pool,
                            value_pool,
                            context_pages,
                            sK[None, None, write_stage],
                            sV[None, None, write_stage],
                            batch,
                            kv_head,
                            tile * self.tile_n,
                            context_len,
                            tma_atom_k,
                            tma_key,
                            tma_atom_v,
                            tma_value,
                            context_barriers + write_stage,
                        )
                        # TMA completion supplies the expected transaction bytes.
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(context_barriers + write_stage)

                if first_tile < last_tile and last_tile * self.tile_n > context_len:
                    tile = last_tile - 1
                    write_stage = (tile - first_tile) % self.num_stages
                    empty_phase = ((tile - first_tile) // self.num_stages) % 2 ^ 1
                    if warp_idx == 4:
                        cute.arch.mbarrier_wait(
                            empty_barriers + write_stage, empty_phase
                        )
                    # Only the tail needs all 128 producer threads for cp.async.
                    primitives.barrier_cta_sync(1, thread_count=128)
                    self.load_context_tile_async(
                        key_pool,
                        value_pool,
                        context_pages,
                        sK[None, None, write_stage],
                        sV[None, None, write_stage],
                        batch,
                        kv_head,
                        tile * self.tile_n,
                        context_len,
                    )
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    primitives.barrier_cta_sync(1, thread_count=128)
                    if warp_idx == 4:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(context_barriers + write_stage)

        if warp_idx < 4:
            if cutlass.const_expr(self.tma_kv):
                cute.arch.setmaxregister_increase(224)
            # Descriptors are shared by one warpgroup; C fragments are per thread.
            warpgroup_qk = tiled_mma_qk.get_slice(0)
            fragment_q = tiled_mma_qk.make_fragment_A(warpgroup_qk.partition_A(sQ))
            fragment_k = tiled_mma_qk.make_fragment_B(warpgroup_qk.partition_B(sK))
            warpgroup_pv = tiled_mma_pv.get_slice(0)
            fragment_p = tiled_mma_pv.make_fragment_A(
                warpgroup_pv.partition_shape_A((self.tile_m, self.tile_n))
            )
            fragment_v = tiled_mma_pv.make_fragment_B(warpgroup_pv.partition_B(sVt))

            score_coordinates = tiled_mma_qk.get_slice(thread_idx).partition_C(
                cute.make_identity_tensor((self.tile_m, self.tile_n))
            )
            output_coordinates = tiled_mma_pv.get_slice(thread_idx).partition_C(
                cute.make_identity_tensor((self.tile_m, self.head_dim))
            )
            scores = cute.make_rmem_tensor(score_coordinates.shape, cutlass.Float32)
            accum = cute.make_rmem_tensor(output_coordinates.shape, cutlass.Float32)
            accum.fill(0.0)
            scores_mn = layout_utils.reshape_acc_to_mn(scores)
            accum_mn = layout_utils.reshape_acc_to_mn(accum)
            score_coordinates_mn = layout_utils.reshape_acc_to_mn(score_coordinates)
            output_coordinates_mn = layout_utils.reshape_acc_to_mn(output_coordinates)

            num_rows = cute.size(accum_mn, mode=[0])
            num_score_columns = cute.size(scores_mn, mode=[1])
            num_output_columns = cute.size(accum_mn, mode=[1])
            row_max = cute.make_rmem_tensor((num_rows,), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((num_rows,), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)
            tiled_mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            tiled_mma_pv.set(warpgroup.Field.ACCUMULATE, True)

            if cutlass.const_expr(self.async_kv and not self.tma_kv):
                if first_tile < last_tile:
                    self.load_context_tile(
                        key_pool,
                        value_pool,
                        context_pages,
                        sK[None, None, 0],
                        sV[None, None, 0],
                        batch,
                        kv_head,
                        first_tile * self.tile_n,
                        context_len,
                        tma_atom_k,
                        tma_key,
                        tma_atom_v,
                        tma_value,
                        None,
                    )
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()

            for context_tile in cutlass.range(first_tile, last_tile):
                context_begin = context_tile * self.tile_n
                read_stage = (context_tile - first_tile) % self.num_stages
                if cutlass.const_expr(self.tma_kv):
                    phase = ((context_tile - first_tile) // self.num_stages) % 2
                    cute.arch.mbarrier_wait(context_barriers + read_stage, phase)
                    cute.arch.fence_view_async_shared()
                elif cutlass.const_expr(self.async_kv):
                    next_tile = context_tile + 1
                    if next_tile < last_tile:
                        self.load_context_tile(
                            key_pool,
                            value_pool,
                            context_pages,
                            sK[None, None, 1 - read_stage],
                            sV[None, None, 1 - read_stage],
                            batch,
                            kv_head,
                            next_tile * self.tile_n,
                            context_len,
                            tma_atom_k,
                            tma_key,
                            tma_atom_v,
                            tma_value,
                            None,
                        )
                else:
                    for i in cutlass.range_constexpr(
                        self.tile_n * self.head_dim // self.num_threads
                    ):
                        linear = thread_idx + i * self.num_threads
                        local_token = linear // self.head_dim
                        dim = linear % self.head_dim
                        token = context_begin + local_token
                        key_value = cutlass.BFloat16(0.0)
                        value_value = cutlass.BFloat16(0.0)
                        if token < context_len:
                            page = cutlass.Int64(
                                context_pages[batch, token // self.page_size]
                            )
                            slot = page * cutlass.Int64(self.page_size) + cutlass.Int64(
                                token % self.page_size
                            )
                            key_value = key_pool[slot, kv_head, dim]
                            value_value = value_pool[slot, kv_head, dim]
                        sK[local_token, dim, 0] = key_value
                        sV[local_token, dim, 0] = value_value
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()

                scores.fill(0.0)
                warpgroup.fence()
                for k in cutlass.range_constexpr(cute.size(fragment_q, mode=[2])):
                    cute.gemm(
                        tiled_mma_qk,
                        scores,
                        fragment_q[None, None, k],
                        fragment_k[None, None, k, read_stage],
                        scores,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                for r in cutlass.range_constexpr(num_rows):
                    local_beam = score_coordinates_mn[r, 0][0]
                    beam = beam_begin + local_beam
                    new_max = row_max[r]
                    for c in cutlass.range_constexpr(num_score_columns):
                        token = context_begin + score_coordinates_mn[r, c][1]
                        score = -cutlass.Float32.inf
                        if beam < q.shape[1] and token < context_len:
                            score = scores_mn[r, c] * scale
                        scores_mn[r, c] = score
                        new_max = cute.arch.fmax(new_max, score)
                    for i in cutlass.range_constexpr(2):
                        new_max = cute.arch.fmax(
                            new_max,
                            cute.arch.shuffle_sync_bfly(new_max, offset=1 << i),
                        )
                    safe_max = new_max
                    if new_max == -cutlass.Float32.inf:
                        safe_max = cutlass.Float32(0.0)
                    old_weight = cute.math.exp2((row_max[r] - safe_max) * _LOG2_E)
                    tile_sum = cutlass.Float32(0.0)
                    for c in cutlass.range_constexpr(num_score_columns):
                        weight = cute.math.exp2((scores_mn[r, c] - safe_max) * _LOG2_E)
                        scores_mn[r, c] = weight
                        tile_sum += weight
                    # Reduce only this tile; previous row_sum is already replicated.
                    for i in cutlass.range_constexpr(2):
                        tile_sum += cute.arch.shuffle_sync_bfly(tile_sum, offset=1 << i)
                    row_sum[r] = row_sum[r] * old_weight + tile_sum
                    row_max[r] = new_max
                    for c in cutlass.range_constexpr(num_output_columns):
                        accum_mn[r, c] = accum_mn[r, c] * old_weight

                # Reinterpret QK's accumulator layout as PV's register A layout.
                weights = layout_utils.reshape_acc_to_frgA(scores)
                fragment_p.store(weights.load().to(cutlass.BFloat16))
                warpgroup.fence()
                for k in cutlass.range_constexpr(cute.size(fragment_p, mode=[2])):
                    cute.gemm(
                        tiled_mma_pv,
                        accum,
                        fragment_p[None, None, k],
                        fragment_v[None, None, k, read_stage],
                        accum,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                if cutlass.const_expr(self.tma_kv):
                    # Release this stage only after PV has finished all shared reads.
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_barriers + read_stage)
                else:
                    if cutlass.const_expr(self.async_kv):
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                    cute.arch.sync_threads()

            # Distribute private history across splits; four threads share a beam row.
            quad_lane = thread_idx % 4
            for history_step in cutlass.range(split, history_len, self.num_splits):
                for r in cutlass.range_constexpr(num_rows):
                    local_beam = output_coordinates_mn[r, 0][0]
                    beam = beam_begin + local_beam
                    slot = cutlass.Int64(0)
                    score = cutlass.Float32(0.0)
                    if beam < q.shape[1]:
                        column = cutlass.Int64(
                            ancestor_columns[batch, beam, history_step]
                        )
                        page = cutlass.Int64(
                            beam_pages[batch, history_step, column // self.page_size]
                        )
                        slot = (
                            page * cutlass.Int64(self.page_size)
                            + column % self.page_size
                        )
                        for k in cutlass.range_constexpr(self.head_dim // 4):
                            dim = quad_lane + k * 4
                            score += cutlass.Float32(
                                sQ[local_beam, dim]
                            ) * cutlass.Float32(key_pool[slot, kv_head, dim])
                    # Keep shuffles outside the tail-beam predicate.
                    for i in cutlass.range_constexpr(2):
                        score += cute.arch.shuffle_sync_bfly(score, offset=1 << i)
                    score *= scale
                    if beam < q.shape[1]:
                        new_max = cute.arch.fmax(row_max[r], score)
                        old_weight = cute.math.exp2((row_max[r] - new_max) * _LOG2_E)
                        token_weight = cute.math.exp2((score - new_max) * _LOG2_E)
                        row_sum[r] = row_sum[r] * old_weight + token_weight
                        row_max[r] = new_max
                        for c in cutlass.range_constexpr(num_output_columns):
                            dim = output_coordinates_mn[r, c][1]
                            value = cutlass.Float32(value_pool[slot, kv_head, dim])
                            accum_mn[r, c] = (
                                accum_mn[r, c] * old_weight + token_weight * value
                            )

            for r in cutlass.range_constexpr(num_rows):
                beam = beam_begin + output_coordinates_mn[r, 0][0]
                if beam < q.shape[1]:
                    if cutlass.const_expr(self.num_splits == 1):
                        inv_sum = cutlass.Float32(1.0) / row_sum[r]
                        for c in cutlass.range_constexpr(num_output_columns):
                            dim = output_coordinates_mn[r, c][1]
                            out[batch, beam, head, dim] = cutlass.BFloat16(
                                accum_mn[r, c] * inv_sum
                            )
                    else:
                        if quad_lane == 0:
                            partials[batch, beam, head, split, 0] = row_max[r]
                            partials[batch, beam, head, split, 1] = row_sum[r]
                        for c in cutlass.range_constexpr(num_output_columns):
                            dim = output_coordinates_mn[r, c][1]
                            partials[batch, beam, head, split, dim + 2] = accum_mn[r, c]

    @cute.jit
    def load_context_tile(
        self,
        key_pool: cute.Tensor,
        value_pool: cute.Tensor,
        context_pages: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        batch: cutlass.Int32,
        kv_head: cutlass.Int32,
        context_begin: cutlass.Int32,
        context_len: cutlass.Int32,
        tma_atom_k,
        tma_key,
        tma_atom_v,
        tma_value,
        context_barrier,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)
        if cutlass.const_expr(self.tma_kv):
            if context_begin + self.tile_n <= context_len:
                if warp_idx % 4 == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(
                            context_barrier, self.tile_n * self.head_dim * 2 * 2
                        )
                    for page_in_tile in cutlass.range_constexpr(self.tile_n // 16):
                        physical_page = context_pages[
                            batch, context_begin // 16 + page_in_tile
                        ]
                        for half in cutlass.range_constexpr(self.head_dim // 64):
                            src_k = cute.local_tile(
                                tma_key[None, None, kv_head],
                                (16, 64),
                                (physical_page, half),
                            )
                            src_v = cute.local_tile(
                                tma_value[None, None, kv_head],
                                (16, 64),
                                (physical_page, half),
                            )
                            dst_k = cute.local_tile(sK, (16, 64), (page_in_tile, half))
                            dst_v = cute.local_tile(sV, (16, 64), (page_in_tile, half))
                            copy_k, coord_k = cpasync.tma_partition(
                                tma_atom_k,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(dst_k, 0, 2),
                                cute.group_modes(src_k, 0, 2),
                            )
                            copy_v, coord_v = cpasync.tma_partition(
                                tma_atom_v,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(dst_v, 0, 2),
                                cute.group_modes(src_v, 0, 2),
                            )
                            # The TMA copy atom elects its issuing lane itself.
                            cute.copy(
                                tma_atom_k, coord_k, copy_k, tma_bar_ptr=context_barrier
                            )
                            cute.copy(
                                tma_atom_v, coord_v, copy_v, tma_bar_ptr=context_barrier
                            )
            else:
                self.load_context_tile_async(
                    key_pool,
                    value_pool,
                    context_pages,
                    sK,
                    sV,
                    batch,
                    kv_head,
                    context_begin,
                    context_len,
                )
        else:
            self.load_context_tile_async(
                key_pool,
                value_pool,
                context_pages,
                sK,
                sV,
                batch,
                kv_head,
                context_begin,
                context_len,
            )

    @cute.jit
    def load_context_tile_async(
        self,
        key_pool: cute.Tensor,
        value_pool: cute.Tensor,
        context_pages: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        batch: cutlass.Int32,
        kv_head: cutlass.Int32,
        context_begin: cutlass.Int32,
        context_len: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        # Eight threads cover D=64; each thread copies eight BF16 values.
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((16, 8), stride=(8, 1)),
            cute.make_layout((1, 8)),
        )
        thread_copy = tiled_copy.get_slice(thread_idx % 128)
        copy_coordinates = thread_copy.partition_S(
            cute.make_identity_tensor((self.tile_n, self.head_dim))
        )
        copy_k = thread_copy.partition_D(sK)
        copy_v = thread_copy.partition_D(sV)

        for row in cutlass.range_constexpr(cute.size(copy_k, mode=[1])):
            local_token = copy_coordinates[0, row, 0][0]
            token = context_begin + local_token
            slot = cutlass.Int64(0)
            if token < context_len:
                page = cutlass.Int64(context_pages[batch, token // self.page_size])
                slot = page * cutlass.Int64(self.page_size) + cutlass.Int64(
                    token % self.page_size
                )
            for col in cutlass.range_constexpr(cute.size(copy_k, mode=[2])):
                dim = copy_coordinates[0, row, col][1]
                if token < context_len:
                    # partition_D retains the shared pointer's SW128 mapping.
                    src_k = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            cute.domain_offset(
                                (slot, kv_head, dim), key_pool
                            ).iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        copy_k[None, row, col].layout,
                    )
                    src_v = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            cute.domain_offset(
                                (slot, kv_head, dim), value_pool
                            ).iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        copy_v[None, row, col].layout,
                    )
                    cute.copy(tiled_copy, src_k, copy_k[None, row, col])
                    cute.copy(tiled_copy, src_v, copy_v[None, row, col])
                else:
                    # Zero both operands: zero probabilities cannot mask NaN V.
                    for elem in cutlass.range_constexpr(8):
                        sK[local_token, dim + elem] = cutlass.BFloat16(0.0)
                        sV[local_token, dim + elem] = cutlass.BFloat16(0.0)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def combine(self, partials: cute.Tensor, out: cute.Tensor):
        # Merge unnormalized (max, sum, weighted V) states from the partitions.
        beam_tile, head, batch = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        beam = beam_tile * 4 + thread // 32
        lane = thread % 32
        lane_split = lane % self.combine_group_size
        if beam < out.shape[1]:
            # Each lane group cooperates on the split normalization state.
            split_max = -cutlass.Float32.inf
            split_sum = cutlass.Float32(0.0)
            if lane_split < self.num_splits:
                split_max = partials[batch, beam, head, lane_split, 0]
                split_sum = partials[batch, beam, head, lane_split, 1]
            row_max = split_max
            for i in cutlass.range_constexpr(self.combine_group_bits):
                row_max = cute.arch.fmax(
                    row_max, cute.arch.shuffle_sync_bfly(row_max, offset=1 << i)
                )
            safe_max = row_max
            if row_max == -cutlass.Float32.inf:
                safe_max = cutlass.Float32(0.0)
            weight = cute.math.exp2((split_max - safe_max) * _LOG2_E)
            denominator = weight * split_sum
            for i in cutlass.range_constexpr(self.combine_group_bits):
                denominator += cute.arch.shuffle_sync_bfly(denominator, offset=1 << i)

            numerators = cute.make_rmem_tensor(
                (cute.ceil_div(self.head_dim, 32),), cutlass.Float32
            )
            numerators.fill(0.0)
            for split in cutlass.range_constexpr(self.num_splits):
                split_weight = cute.arch.shuffle_sync(
                    weight,
                    split,
                    mask_and_clamp=((32 - self.combine_group_size) << 8) | 31,
                )
                for i in cutlass.range_constexpr(cute.ceil_div(self.head_dim, 32)):
                    dim = lane + i * 32
                    if dim < self.head_dim:
                        numerators[i] += (
                            split_weight * partials[batch, beam, head, split, dim + 2]
                        )
            for i in cutlass.range_constexpr(cute.ceil_div(self.head_dim, 32)):
                dim = lane + i * 32
                if dim < self.head_dim:
                    out[batch, beam, head, dim] = cutlass.BFloat16(
                        numerators[i] / denominator
                    )
