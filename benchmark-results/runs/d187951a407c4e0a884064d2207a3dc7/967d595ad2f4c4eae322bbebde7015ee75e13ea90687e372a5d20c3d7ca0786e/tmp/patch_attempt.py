from pathlib import Path
import re
p=Path('/attempts/attempt_01.py')
s=p.read_text()
insert_after = "empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p\n\n"
kernel = """
# Fused small batched matvec for query_step @ recurrent state.
# Owns one (batch, head, output-column) vector per x lane and reduces over 64 rows.
# This replaces cuBLAS bmm on shape (8,1,64) x (8,64,64) and avoids materialized
# query clone kernels; arg1 has fixed asserted strides [8192,128,64,1].
triton_red_fused_query_state_bmm = async_compile.triton('triton_red_fused_query_state_bmm', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'timestep': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_query_state_bmm', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 32768, 'r0_': 8192}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_query_state_bmm(in_ptr0, in_ptr1, out_ptr0, timestep, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0 = tl.arange(0, R0_BLOCK)[None, :]
    col = xindex % 64
    head = (xindex // 64) % 4
    batch = xindex // 256
    q = tl.load(in_ptr0 + (batch * 8192 + timestep * 128 + (head // 2) * 64 + r0), xmask, eviction_policy='evict_last', other=0.0)
    st = tl.load(in_ptr1 + (batch * 16384 + head * 4096 + r0 * 64 + col), xmask, eviction_policy='evict_first', other=0.0)
    acc = tl.sum((q * st) * 0.125, 1)[:, None]
    tl.store(out_ptr0 + xindex, acc, xmask)
''', device_str='cuda')

"""
if "triton_red_fused_query_state_bmm = async_compile.triton" not in s:
    s=s.replace(insert_after, insert_after+kernel)
# remove early delete of arg1
s=s.replace("              del arg1_1\n              assert_size_stride(arg4_1", "              assert_size_stride(arg4_1")
# Remove precompute query pointwise run lines only in initial section (all 64). Leave allocations/comments.
s=re.sub(r"\n\s*triton_poi_fused_clone_expand_mul_select_unsqueeze_view_\d+\.run\(arg1_1, (buf\d+), 512, stream=raw_stream0\)", "", s)
# Replace bmm lines sequentially.
pat=re.compile(r"extern_kernels\.bmm\(reinterpret_tensor\((buf\d+), \(8, 1, 64\), \(64, 0, 1\), 0\), reinterpret_tensor\((buf\d+), \(8, 64, 64\), \(4096, 64, 1\), 0\), out=(buf\d+)\)")
def repl(m):
    t=repl.count
    repl.count += 1
    state,out=m.group(2),m.group(3)
    return f"triton_red_fused_query_state_bmm.run(arg1_1, {state}, {out}, {t}, 512, 64, stream=raw_stream0)"
repl.count=0
s=pat.sub(repl, s)
print('replaced bmm', repl.count)
last="triton_red_fused_query_state_bmm.run(arg1_1, buf269, buf270, 63, 512, 64, stream=raw_stream0)\n              del buf268"
if last in s and "del arg1_1\n              del buf268" not in s:
    s=s.replace(last, "triton_red_fused_query_state_bmm.run(arg1_1, buf269, buf270, 63, 512, 64, stream=raw_stream0)\n              del arg1_1\n              del buf268")
p.write_text(s)
