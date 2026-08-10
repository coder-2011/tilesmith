# --- TILESMITH PYTORCH FX REFERENCE (NOT EXECUTED) ---
# Captured from graph_module.code for this compiled graph.
#
#
#
# def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1):
#     convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
#     relu = torch.ops.aten.relu.default(convolution);  convolution = None
#     convolution_1 = torch.ops.aten.convolution.default(relu, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu = arg3_1 = arg4_1 = None
#     relu_1 = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
#     _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_1 = None
#     getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
#     convolution_2 = torch.ops.aten.convolution.default(getitem, arg5_1, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem = arg5_1 = arg6_1 = None
#     relu_2 = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
#     _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False);  relu_2 = None
#     getitem_2 = _low_memory_max_pool_with_offsets_1[0];  _low_memory_max_pool_with_offsets_1 = None
#     view = torch.ops.aten.view.default(getitem_2, [256, 6272]);  getitem_2 = None
#     permute = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
#     addmm = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
#     relu_3 = torch.ops.aten.relu.default(addmm);  addmm = None
#     permute_1 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
#     addmm_1 = torch.ops.aten.addmm.default(arg10_1, relu_3, permute_1);  arg10_1 = relu_3 = permute_1 = None
#     return (addmm_1,)
#     
# --- END TILESMITH PYTORCH FX REFERENCE ---

from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/wr/cwrpagqsyzadcoh5fv2ooffdwhkjgwytwso2qa6iqpmhd2k4mrsg.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %buf0 : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_poi_fused_convolution_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 51380224, 'x': 25690240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + (y0 + 32*x2 + 25088*y1), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/cx/ccxcxwsigtfujnx42njihrvd6oxlw7xehv3bgxngbgxvjy5a3fyo.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
# Graph fragment:
#   %arg3_1 : Tensor "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf2
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 147456, 'x': 73728}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6h/c6hgrxgvlqxqgy3ys47kh3ipyljf4uaba6mafp5jsnvzf64gte3p.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
# Graph fragment:
#   %buf3 : Tensor "f32[256, 64, 28, 28][50176, 1, 1792, 64]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 154140928}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/vb/cvbjhhoztvqz7jprp6jrdg7jqvzblnlfck3u7zzkhd5qcwrtngjf.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 1, 1792, 64]cuda:0" = PlaceHolder[target=relu_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 77070336}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 14)
    x2 = xindex // 896
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x1 + 3584*x2), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 3584*x2), None)
    tmp3 = tl.load(in_ptr0 + (1792 + x0 + 128*x1 + 3584*x2), None)
    tmp5 = tl.load(in_ptr0 + (1856 + x0 + 128*x1 + 3584*x2), None)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ez/cezbx2j4jv2lphn7ferbkzaaqtq52jno5fb57lmfxdopiuktwex4.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
# Graph fragment:
#   %arg5_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[256, 128, 14, 14][25088, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf6
triton_poi_fused_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 589824, 'x': 294912}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6i/c6icqv45qljeieyuc3dwpdlfs534bvupyw4set2oalos2dlgn3zq.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
# Graph fragment:
#   %buf7 : Tensor "f32[256, 128, 14, 14][25088, 1, 1792, 128]cuda:0" = PlaceHolder[target=buf7]
#   %arg6_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[256, 128, 14, 14][25088, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[256, 128, 14, 14][25088, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   return %relu_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 77070848}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/zn/czn6sfyxolaec637xkvklh5seesnuu5xe2omlu444fxx6casn7rh.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
#   input_3 => convolution_1
#   input_4 => relu_1
#   input_5 => _low_memory_max_pool_with_offsets
#   input_6 => convolution_2
#   input_7 => relu_2
#   input_8 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %relu_2 : Tensor "f32[256, 128, 14, 14][25088, 1, 1792, 128]cuda:0" = PlaceHolder[target=relu_2]
#   %convolution : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[256, 32, 28, 28][25088, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %convolution_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[256, 64, 28, 28][50176, 784, 28, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %convolution_2 : Tensor "f32[256, 128, 14, 14][25088, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[256, 128, 14, 14][25088, 196, 14, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 12845056, 'x': 25690112}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 7)
    y4 = yindex // 7
    y2 = yindex // 49
    y5 = (yindex % 49)
    tmp0 = tl.load(in_ptr0 + (x3 + 256*y0 + 3584*y4), xmask & ymask)
    tmp1 = tl.load(in_ptr0 + (128 + x3 + 256*y0 + 3584*y4), xmask & ymask)
    tmp3 = tl.load(in_ptr0 + (1792 + x3 + 256*y0 + 3584*y4), xmask & ymask)
    tmp5 = tl.load(in_ptr0 + (1920 + x3 + 256*y0 + 3584*y4), xmask & ymask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y5 + 49*x3 + 6272*y2), tmp6, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/us/cusz4cb2xoosl4fbmby4xky5dt3726qcocybu2q4aqgaogwfjknq.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_tensor
#   input_11 => relu_3
# Graph fragment:
#   %arg8_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %mm_default : Tensor "f32[256, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, %mm_default), kwargs = {})
#   %relu_3 : Tensor "f32[256, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   return %relu_3
triton_poi_fused_addmm_relu_7 = async_compile.triton('triton_poi_fused_addmm_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1574912}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
        args.clear()
        assert_size_stride(arg2_1, (256, 1, 28, 28), (784, 784, 28, 1), 'input')
        assert_size_stride(arg0_1, (32, 1, 3, 3), (9, 9, 3, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            raw_stream0 = get_raw_stream(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            assert_size_stride(arg1_1, (32, ), (1, ), 'input')
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=arg1_1)
            assert_size_stride(buf0, (256, 32, 28, 28), (25088, 784, 28, 1), 'torch.ops.aten.convolution.default')
            del arg0_1
            del arg2_1
            buf1 = empty_strided_cuda((256, 32, 28, 28), (25088, 1, 896, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
            triton_poi_fused_convolution_relu_0.run(buf0, arg1_1, buf1, 8192, 784, stream=raw_stream0)
            del arg1_1
            del buf0
            assert_size_stride(arg3_1, (64, 32, 3, 3), (288, 9, 3, 1), 'input')
            buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            triton_poi_fused_convolution_relu_1.run(arg3_1, buf2, 2048, 9, stream=raw_stream0)
            del arg3_1
            assert_size_stride(arg4_1, (64, ), (1, ), 'input')
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.relu]
            buf3 = extern_kernels.convolution(buf1, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=arg4_1)
            assert_size_stride(buf3, (256, 64, 28, 28), (50176, 1, 1792, 64), 'torch.ops.aten.convolution.default')
            del buf1
            del buf2
            buf4 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4], Original ATen: [aten.convolution, aten.relu]
            triton_poi_fused_convolution_relu_2.run(buf4, arg4_1, 12845056, stream=raw_stream0)
            del arg4_1
            buf5 = empty_strided_cuda((256, 64, 14, 14), (12544, 1, 896, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf4, buf5, 3211264, stream=raw_stream0)
            del buf4
            assert_size_stride(arg5_1, (128, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf6 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_4.run(arg5_1, buf6, 8192, 9, stream=raw_stream0)
            del arg5_1
            assert_size_stride(arg6_1, (128, ), (1, ), 'input')
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            buf7 = extern_kernels.convolution(buf5, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=arg6_1)
            assert_size_stride(buf7, (256, 128, 14, 14), (25088, 1, 1792, 128), 'torch.ops.aten.convolution.default')
            del buf5
            del buf6
            buf8 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(buf8, arg6_1, 6422528, stream=raw_stream0)
            del arg6_1
            buf9 = empty_strided_cuda((256, 128, 7, 7), (6272, 49, 7, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
            triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(buf8, buf9, 12544, 128, stream=raw_stream0)
            del buf8
            assert_size_stride(arg7_1, (512, 6272), (6272, 1), 'input')
            buf10 = empty_strided_cuda((256, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf9, (256, 6272), (6272, 1), 0), reinterpret_tensor(arg7_1, (6272, 512), (1, 6272), 0), out=buf10)
            del arg7_1
            del buf9
            assert_size_stride(arg8_1, (512, ), (1, ), 'input')
            buf11 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten.addmm, aten.relu]
            triton_poi_fused_addmm_relu_7.run(buf11, arg8_1, 131072, stream=raw_stream0)
            del arg8_1
            assert_size_stride(arg10_1, (10, ), (1, ), 'input')
            assert_size_stride(arg9_1, (10, 512), (512, 1), 'input')
            buf12 = empty_strided_cuda((256, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.addmm, aten.relu, aten.t]
            extern_kernels.addmm(arg10_1, buf11, reinterpret_tensor(arg9_1, (512, 10), (1, 512), 0), alpha=1, beta=1, out=buf12)
            del arg10_1
            del arg9_1
            del buf11
        return (buf12, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 1, 28, 28), (784, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 6272), (6272, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
