# --- TILESMITH PYTORCH FX REFERENCE (NOT EXECUTED) ---
# Captured from graph_module.code for this compiled graph.
#
#
#
# def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1):
#     embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
#     view = torch.ops.aten.view.default(embedding, [2048, 256]);  embedding = None
#     permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
#     addmm = torch.ops.aten.addmm.default(arg3_1, view, permute);  arg3_1 = view = permute = None
#     view_1 = torch.ops.aten.view.default(addmm, [64, 32, 256]);  addmm = None
#     full_default = torch.ops.aten.full.default([64, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
#     select_1 = torch.ops.aten.select.int(view_1, 1, 0)
#     permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm = torch.ops.aten.mm.default(full_default, permute_1);  full_default = permute_1 = None
#     add = torch.ops.aten.add.Tensor(select_1, mm);  select_1 = mm = None
#     tanh = torch.ops.aten.tanh.default(add);  add = None
#     select_2 = torch.ops.aten.select.int(view_1, 1, 1)
#     permute_2 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_1 = torch.ops.aten.mm.default(tanh, permute_2);  permute_2 = None
#     add_1 = torch.ops.aten.add.Tensor(select_2, mm_1);  select_2 = mm_1 = None
#     tanh_1 = torch.ops.aten.tanh.default(add_1);  add_1 = None
#     select_3 = torch.ops.aten.select.int(view_1, 1, 2)
#     permute_3 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_2 = torch.ops.aten.mm.default(tanh_1, permute_3);  permute_3 = None
#     add_2 = torch.ops.aten.add.Tensor(select_3, mm_2);  select_3 = mm_2 = None
#     tanh_2 = torch.ops.aten.tanh.default(add_2);  add_2 = None
#     select_4 = torch.ops.aten.select.int(view_1, 1, 3)
#     permute_4 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_3 = torch.ops.aten.mm.default(tanh_2, permute_4);  permute_4 = None
#     add_3 = torch.ops.aten.add.Tensor(select_4, mm_3);  select_4 = mm_3 = None
#     tanh_3 = torch.ops.aten.tanh.default(add_3);  add_3 = None
#     select_5 = torch.ops.aten.select.int(view_1, 1, 4)
#     permute_5 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_4 = torch.ops.aten.mm.default(tanh_3, permute_5);  permute_5 = None
#     add_4 = torch.ops.aten.add.Tensor(select_5, mm_4);  select_5 = mm_4 = None
#     tanh_4 = torch.ops.aten.tanh.default(add_4);  add_4 = None
#     select_6 = torch.ops.aten.select.int(view_1, 1, 5)
#     permute_6 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_5 = torch.ops.aten.mm.default(tanh_4, permute_6);  permute_6 = None
#     add_5 = torch.ops.aten.add.Tensor(select_6, mm_5);  select_6 = mm_5 = None
#     tanh_5 = torch.ops.aten.tanh.default(add_5);  add_5 = None
#     select_7 = torch.ops.aten.select.int(view_1, 1, 6)
#     permute_7 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_6 = torch.ops.aten.mm.default(tanh_5, permute_7);  permute_7 = None
#     add_6 = torch.ops.aten.add.Tensor(select_7, mm_6);  select_7 = mm_6 = None
#     tanh_6 = torch.ops.aten.tanh.default(add_6);  add_6 = None
#     select_8 = torch.ops.aten.select.int(view_1, 1, 7)
#     permute_8 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_7 = torch.ops.aten.mm.default(tanh_6, permute_8);  permute_8 = None
#     add_7 = torch.ops.aten.add.Tensor(select_8, mm_7);  select_8 = mm_7 = None
#     tanh_7 = torch.ops.aten.tanh.default(add_7);  add_7 = None
#     select_9 = torch.ops.aten.select.int(view_1, 1, 8)
#     permute_9 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_8 = torch.ops.aten.mm.default(tanh_7, permute_9);  permute_9 = None
#     add_8 = torch.ops.aten.add.Tensor(select_9, mm_8);  select_9 = mm_8 = None
#     tanh_8 = torch.ops.aten.tanh.default(add_8);  add_8 = None
#     select_10 = torch.ops.aten.select.int(view_1, 1, 9)
#     permute_10 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_9 = torch.ops.aten.mm.default(tanh_8, permute_10);  permute_10 = None
#     add_9 = torch.ops.aten.add.Tensor(select_10, mm_9);  select_10 = mm_9 = None
#     tanh_9 = torch.ops.aten.tanh.default(add_9);  add_9 = None
#     select_11 = torch.ops.aten.select.int(view_1, 1, 10)
#     permute_11 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_10 = torch.ops.aten.mm.default(tanh_9, permute_11);  permute_11 = None
#     add_10 = torch.ops.aten.add.Tensor(select_11, mm_10);  select_11 = mm_10 = None
#     tanh_10 = torch.ops.aten.tanh.default(add_10);  add_10 = None
#     select_12 = torch.ops.aten.select.int(view_1, 1, 11)
#     permute_12 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_11 = torch.ops.aten.mm.default(tanh_10, permute_12);  permute_12 = None
#     add_11 = torch.ops.aten.add.Tensor(select_12, mm_11);  select_12 = mm_11 = None
#     tanh_11 = torch.ops.aten.tanh.default(add_11);  add_11 = None
#     select_13 = torch.ops.aten.select.int(view_1, 1, 12)
#     permute_13 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_12 = torch.ops.aten.mm.default(tanh_11, permute_13);  permute_13 = None
#     add_12 = torch.ops.aten.add.Tensor(select_13, mm_12);  select_13 = mm_12 = None
#     tanh_12 = torch.ops.aten.tanh.default(add_12);  add_12 = None
#     select_14 = torch.ops.aten.select.int(view_1, 1, 13)
#     permute_14 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_13 = torch.ops.aten.mm.default(tanh_12, permute_14);  permute_14 = None
#     add_13 = torch.ops.aten.add.Tensor(select_14, mm_13);  select_14 = mm_13 = None
#     tanh_13 = torch.ops.aten.tanh.default(add_13);  add_13 = None
#     select_15 = torch.ops.aten.select.int(view_1, 1, 14)
#     permute_15 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_14 = torch.ops.aten.mm.default(tanh_13, permute_15);  permute_15 = None
#     add_14 = torch.ops.aten.add.Tensor(select_15, mm_14);  select_15 = mm_14 = None
#     tanh_14 = torch.ops.aten.tanh.default(add_14);  add_14 = None
#     select_16 = torch.ops.aten.select.int(view_1, 1, 15)
#     permute_16 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_15 = torch.ops.aten.mm.default(tanh_14, permute_16);  permute_16 = None
#     add_15 = torch.ops.aten.add.Tensor(select_16, mm_15);  select_16 = mm_15 = None
#     tanh_15 = torch.ops.aten.tanh.default(add_15);  add_15 = None
#     select_17 = torch.ops.aten.select.int(view_1, 1, 16)
#     permute_17 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_16 = torch.ops.aten.mm.default(tanh_15, permute_17);  permute_17 = None
#     add_16 = torch.ops.aten.add.Tensor(select_17, mm_16);  select_17 = mm_16 = None
#     tanh_16 = torch.ops.aten.tanh.default(add_16);  add_16 = None
#     select_18 = torch.ops.aten.select.int(view_1, 1, 17)
#     permute_18 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_17 = torch.ops.aten.mm.default(tanh_16, permute_18);  permute_18 = None
#     add_17 = torch.ops.aten.add.Tensor(select_18, mm_17);  select_18 = mm_17 = None
#     tanh_17 = torch.ops.aten.tanh.default(add_17);  add_17 = None
#     select_19 = torch.ops.aten.select.int(view_1, 1, 18)
#     permute_19 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_18 = torch.ops.aten.mm.default(tanh_17, permute_19);  permute_19 = None
#     add_18 = torch.ops.aten.add.Tensor(select_19, mm_18);  select_19 = mm_18 = None
#     tanh_18 = torch.ops.aten.tanh.default(add_18);  add_18 = None
#     select_20 = torch.ops.aten.select.int(view_1, 1, 19)
#     permute_20 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_19 = torch.ops.aten.mm.default(tanh_18, permute_20);  permute_20 = None
#     add_19 = torch.ops.aten.add.Tensor(select_20, mm_19);  select_20 = mm_19 = None
#     tanh_19 = torch.ops.aten.tanh.default(add_19);  add_19 = None
#     select_21 = torch.ops.aten.select.int(view_1, 1, 20)
#     permute_21 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_20 = torch.ops.aten.mm.default(tanh_19, permute_21);  permute_21 = None
#     add_20 = torch.ops.aten.add.Tensor(select_21, mm_20);  select_21 = mm_20 = None
#     tanh_20 = torch.ops.aten.tanh.default(add_20);  add_20 = None
#     select_22 = torch.ops.aten.select.int(view_1, 1, 21)
#     permute_22 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_21 = torch.ops.aten.mm.default(tanh_20, permute_22);  permute_22 = None
#     add_21 = torch.ops.aten.add.Tensor(select_22, mm_21);  select_22 = mm_21 = None
#     tanh_21 = torch.ops.aten.tanh.default(add_21);  add_21 = None
#     select_23 = torch.ops.aten.select.int(view_1, 1, 22)
#     permute_23 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_22 = torch.ops.aten.mm.default(tanh_21, permute_23);  permute_23 = None
#     add_22 = torch.ops.aten.add.Tensor(select_23, mm_22);  select_23 = mm_22 = None
#     tanh_22 = torch.ops.aten.tanh.default(add_22);  add_22 = None
#     select_24 = torch.ops.aten.select.int(view_1, 1, 23)
#     permute_24 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_23 = torch.ops.aten.mm.default(tanh_22, permute_24);  permute_24 = None
#     add_23 = torch.ops.aten.add.Tensor(select_24, mm_23);  select_24 = mm_23 = None
#     tanh_23 = torch.ops.aten.tanh.default(add_23);  add_23 = None
#     select_25 = torch.ops.aten.select.int(view_1, 1, 24)
#     permute_25 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_24 = torch.ops.aten.mm.default(tanh_23, permute_25);  permute_25 = None
#     add_24 = torch.ops.aten.add.Tensor(select_25, mm_24);  select_25 = mm_24 = None
#     tanh_24 = torch.ops.aten.tanh.default(add_24);  add_24 = None
#     select_26 = torch.ops.aten.select.int(view_1, 1, 25)
#     permute_26 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_25 = torch.ops.aten.mm.default(tanh_24, permute_26);  permute_26 = None
#     add_25 = torch.ops.aten.add.Tensor(select_26, mm_25);  select_26 = mm_25 = None
#     tanh_25 = torch.ops.aten.tanh.default(add_25);  add_25 = None
#     select_27 = torch.ops.aten.select.int(view_1, 1, 26)
#     permute_27 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_26 = torch.ops.aten.mm.default(tanh_25, permute_27);  permute_27 = None
#     add_26 = torch.ops.aten.add.Tensor(select_27, mm_26);  select_27 = mm_26 = None
#     tanh_26 = torch.ops.aten.tanh.default(add_26);  add_26 = None
#     select_28 = torch.ops.aten.select.int(view_1, 1, 27)
#     permute_28 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_27 = torch.ops.aten.mm.default(tanh_26, permute_28);  permute_28 = None
#     add_27 = torch.ops.aten.add.Tensor(select_28, mm_27);  select_28 = mm_27 = None
#     tanh_27 = torch.ops.aten.tanh.default(add_27);  add_27 = None
#     select_29 = torch.ops.aten.select.int(view_1, 1, 28)
#     permute_29 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_28 = torch.ops.aten.mm.default(tanh_27, permute_29);  permute_29 = None
#     add_28 = torch.ops.aten.add.Tensor(select_29, mm_28);  select_29 = mm_28 = None
#     tanh_28 = torch.ops.aten.tanh.default(add_28);  add_28 = None
#     select_30 = torch.ops.aten.select.int(view_1, 1, 29)
#     permute_30 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_29 = torch.ops.aten.mm.default(tanh_28, permute_30);  permute_30 = None
#     add_29 = torch.ops.aten.add.Tensor(select_30, mm_29);  select_30 = mm_29 = None
#     tanh_29 = torch.ops.aten.tanh.default(add_29);  add_29 = None
#     select_31 = torch.ops.aten.select.int(view_1, 1, 30)
#     permute_31 = torch.ops.aten.permute.default(arg4_1, [1, 0])
#     mm_30 = torch.ops.aten.mm.default(tanh_29, permute_31);  permute_31 = None
#     add_30 = torch.ops.aten.add.Tensor(select_31, mm_30);  select_31 = mm_30 = None
#     tanh_30 = torch.ops.aten.tanh.default(add_30);  add_30 = None
#     select_32 = torch.ops.aten.select.int(view_1, 1, 31);  view_1 = None
#     permute_32 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
#     mm_31 = torch.ops.aten.mm.default(tanh_30, permute_32);  permute_32 = None
#     add_31 = torch.ops.aten.add.Tensor(select_32, mm_31);  select_32 = mm_31 = None
#     tanh_31 = torch.ops.aten.tanh.default(add_31);  add_31 = None
#     cat = torch.ops.aten.cat.default([tanh, tanh_1, tanh_2, tanh_3, tanh_4, tanh_5, tanh_6, tanh_7, tanh_8, tanh_9, tanh_10, tanh_11, tanh_12, tanh_13, tanh_14, tanh_15, tanh_16, tanh_17, tanh_18, tanh_19, tanh_20, tanh_21, tanh_22, tanh_23, tanh_24, tanh_25, tanh_26, tanh_27, tanh_28, tanh_29, tanh_30, tanh_31], 1);  tanh = tanh_1 = tanh_2 = tanh_3 = tanh_4 = tanh_5 = tanh_6 = tanh_7 = tanh_8 = tanh_9 = tanh_10 = tanh_11 = tanh_12 = tanh_13 = tanh_14 = tanh_15 = tanh_16 = tanh_17 = tanh_18 = tanh_19 = tanh_20 = tanh_21 = tanh_22 = tanh_23 = tanh_24 = tanh_25 = tanh_26 = tanh_27 = tanh_28 = tanh_29 = tanh_30 = tanh_31 = None
#     view_2 = torch.ops.aten.view.default(cat, [64, 32, 256]);  cat = None
#     view_3 = torch.ops.aten.view.default(view_2, [2048, 256])
#     permute_33 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
#     addmm_1 = torch.ops.aten.addmm.default(arg6_1, view_3, permute_33);  arg6_1 = view_3 = permute_33 = None
#     view_4 = torch.ops.aten.view.default(addmm_1, [64, 32, 128]);  addmm_1 = None
#     tanh_32 = torch.ops.aten.tanh.default(view_4);  view_4 = None
#     view_5 = torch.ops.aten.view.default(tanh_32, [2048, 128]);  tanh_32 = None
#     permute_34 = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
#     addmm_2 = torch.ops.aten.addmm.default(arg8_1, view_5, permute_34);  arg8_1 = view_5 = permute_34 = None
#     view_6 = torch.ops.aten.view.default(addmm_2, [64, 32, 1]);  addmm_2 = None
#     amax = torch.ops.aten.amax.default(view_6, [1], True)
#     sub = torch.ops.aten.sub.Tensor(view_6, amax);  view_6 = amax = None
#     exp = torch.ops.aten.exp.default(sub);  sub = None
#     sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True)
#     div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
#     mul = torch.ops.aten.mul.Tensor(view_2, div);  view_2 = div = None
#     sum_2 = torch.ops.aten.sum.dim_IntList(mul, [1]);  mul = None
#     permute_35 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
#     addmm_3 = torch.ops.aten.addmm.default(arg10_1, sum_2, permute_35);  arg10_1 = sum_2 = permute_35 = None
#     return (addmm_3,)
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


# kernel path: /tmp/torchinductor_root/6q/c6qv4sqb2c7t7knv75k6e4wzc5ngqyub72j7sdwxatby6tl3ep6k.py
# Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
# Graph fragment:
#   %arg1_1 : Tensor "i64[64, 32][32, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[8192, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %embedding : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   return %embedding
triton_poi_fused_embedding_0 = async_compile.triton('triton_poi_fused_embedding_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 4210688}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 256
    x0 = (xindex % 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 8192), "index out of bounds: 0 <= tmp0 < 8192")
    tmp2 = tl.load(in_ptr1 + (x0 + 256*tmp0), None)
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/c3/cc3sm25y6vw6gb5a7ystxei5oqsf2lnox3bow26egurlyljpxk44.py
# Topologically Sorted Source Nodes: [hidden], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   hidden => full_default
# Graph fragment:
#   %full_default : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([64, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_zeros_like_1 = async_compile.triton('triton_poi_fused_zeros_like_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 131072}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_like_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/vk/cvkhlweih5n5drildf7wupvhzjg6m6b6lpp6gkfr66kr2zntvi4u.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_1, add, hidden_1, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add => add
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_1 => select_1
#   getitem_32 => select_32
#   hidden_1 => tanh
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %tanh : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_1 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 0), kwargs = {})
#   %add : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_1, %mm), kwargs = {})
#   %tanh : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh,%buf66
triton_poi_fused_add_addmm_select_stack_tanh_view_2 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 8192*x1), None)
    tmp2 = tmp0 + tmp1
    tmp5 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bi/cbix3xynugu7qckytbuym6nnb2wwhs7y4u7segtdo66aovphwho2.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_2, add_1, hidden_2, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_2 => select_2
#   getitem_32 => select_32
#   hidden_2 => tanh_1
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_1 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %tanh_1 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_1]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_2 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 1), kwargs = {})
#   %add_1 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %mm_1), kwargs = {})
#   %tanh_1 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_1,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_1,%buf67
triton_poi_fused_add_addmm_select_stack_tanh_view_3 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/sb/csbbwzdtmh52pex7fuhkoxjrk5r4uno7tmpzysa3rhr63ocbw7yr.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_3, add_2, hidden_3, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_3 => select_3
#   getitem_32 => select_32
#   hidden_3 => tanh_2
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_2 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %tanh_2 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_2]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_3 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 2), kwargs = {})
#   %add_2 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_3, %mm_2), kwargs = {})
#   %tanh_2 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_2,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_2,%buf68
triton_poi_fused_add_addmm_select_stack_tanh_view_4 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/yo/cyoasb5frbbzg6urk7ajnu4wjxrgv7vxjm3hd7ykulfy4hhur5tb.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_4, add_3, hidden_4, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_3 => add_3
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_4 => select_4
#   hidden_32 => tanh_31
#   hidden_4 => tanh_3
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_3 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %tanh_3 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_3]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_4 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 3), kwargs = {})
#   %add_3 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_4, %mm_3), kwargs = {})
#   %tanh_3 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_3,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_3,%buf69
triton_poi_fused_add_addmm_select_stack_tanh_view_5 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_5(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (768 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rg/crgfgu26mnqwaesg3kgse3oxjqmgjx75c4nz6k5dzi6swvopwvyb.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_5, add_4, hidden_5, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_4 => add_4
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_5 => select_5
#   hidden_32 => tanh_31
#   hidden_5 => tanh_4
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_4 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %tanh_4 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_4]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_5 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 4), kwargs = {})
#   %add_4 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_5, %mm_4), kwargs = {})
#   %tanh_4 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_4,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_4,%buf70
triton_poi_fused_add_addmm_select_stack_tanh_view_6 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_6(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1024 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/r3/cr3drz72lzljuk2lfloyahwel2lbfqj7ucz5twybycs2g33kdo6n.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_6, add_5, hidden_6, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_5 => add_5
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_6 => select_6
#   hidden_32 => tanh_31
#   hidden_6 => tanh_5
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_5 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %tanh_5 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_5]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_6 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 5), kwargs = {})
#   %add_5 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_6, %mm_5), kwargs = {})
#   %tanh_5 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_5,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_5,%buf71
triton_poi_fused_add_addmm_select_stack_tanh_view_7 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1280 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/da/cdabpv6iezlmrh6jlf5ghbt5o7w3g3iszfxa5nwsbrxn7flsubjd.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_7, add_6, hidden_7, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_6 => add_6
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_7 => select_7
#   hidden_32 => tanh_31
#   hidden_7 => tanh_6
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_6 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %tanh_6 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_6]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_7 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 6), kwargs = {})
#   %add_6 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_7, %mm_6), kwargs = {})
#   %tanh_6 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_6,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_6,%buf72
triton_poi_fused_add_addmm_select_stack_tanh_view_8 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1536 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mp/cmpsha7ipf3jacxqgvir5r5cehpiw6ea6c3r4zzlfot2vlwaklhs.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_8, add_7, hidden_8, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_7 => add_7
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_8 => select_8
#   hidden_32 => tanh_31
#   hidden_8 => tanh_7
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_7 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %tanh_7 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_7]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_8 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 7), kwargs = {})
#   %add_7 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_8, %mm_7), kwargs = {})
#   %tanh_7 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_7,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_7,%buf73
triton_poi_fused_add_addmm_select_stack_tanh_view_9 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (1792 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xv/cxvv2va5ojb4jqckz2yqorhkofrfj6m2ghx3bbquymzprnlw5bmp.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_9, add_8, hidden_9, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_8 => add_8
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   getitem_9 => select_9
#   hidden_32 => tanh_31
#   hidden_9 => tanh_8
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_8 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %tanh_8 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_8]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_9 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 8), kwargs = {})
#   %add_8 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_9, %mm_8), kwargs = {})
#   %tanh_8 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_8,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_8,%buf74
triton_poi_fused_add_addmm_select_stack_tanh_view_10 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_10(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2048 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ki/ckixe6aojfh7lyfb3roztzikkd6rpkevet6hzsf232ps7cjbwbze.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_10, add_9, hidden_10, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_31 => add_31
#   add_9 => add_9
#   embedded => add_tensor_1, view_1
#   getitem_10 => select_10
#   getitem_32 => select_32
#   hidden_10 => tanh_9
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_9 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %tanh_9 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_9]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_10 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 9), kwargs = {})
#   %add_9 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_10, %mm_9), kwargs = {})
#   %tanh_9 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_9,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_9,%buf75
triton_poi_fused_add_addmm_select_stack_tanh_view_11 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2304 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ox/coxdza2cadtdjeudbirivyxrwdoj5d5wvad7pylmdged4pehpmjs.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_11, add_10, hidden_11, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_11 => select_11
#   getitem_32 => select_32
#   hidden_11 => tanh_10
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_10 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %tanh_10 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_10]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_11 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 10), kwargs = {})
#   %add_10 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_11, %mm_10), kwargs = {})
#   %tanh_10 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_10,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_10,%buf76
triton_poi_fused_add_addmm_select_stack_tanh_view_12 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_12(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2560 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lo/clo6w2wc7biytb5gtdtmbbwci2cd7pzce52bub3kru2pwh26xkua.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_12, add_11, hidden_12, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_11 => add_11
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_12 => select_12
#   getitem_32 => select_32
#   hidden_12 => tanh_11
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_11 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %tanh_11 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_11]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_12 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 11), kwargs = {})
#   %add_11 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_12, %mm_11), kwargs = {})
#   %tanh_11 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_11,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_11,%buf77
triton_poi_fused_add_addmm_select_stack_tanh_view_13 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_13(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2816 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/uk/cukp7e6chcezjsolopvr4z3x5ra6pob6ao3bb4clautj3lmyq42p.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_13, add_12, hidden_13, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_12 => add_12
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_13 => select_13
#   getitem_32 => select_32
#   hidden_13 => tanh_12
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_12 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %tanh_12 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_12]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_13 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 12), kwargs = {})
#   %add_12 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_13, %mm_12), kwargs = {})
#   %tanh_12 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_12,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_12,%buf78
triton_poi_fused_add_addmm_select_stack_tanh_view_14 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_14(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (3072 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rh/crhuycyzhrxszfloacj3bix7ay65m726qbd6ewsaic2tzuelrizd.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_14, add_13, hidden_14, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_13 => add_13
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_14 => select_14
#   getitem_32 => select_32
#   hidden_14 => tanh_13
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_13 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %tanh_13 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_13]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_14 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 13), kwargs = {})
#   %add_13 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_14, %mm_13), kwargs = {})
#   %tanh_13 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_13,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_13,%buf79
triton_poi_fused_add_addmm_select_stack_tanh_view_15 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_15(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (3328 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bl/cblqixznltny4pcyjpnlevar5bvc34wsilxp4jukuca3gwbwus3r.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_15, add_14, hidden_15, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_14 => add_14
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_15 => select_15
#   getitem_32 => select_32
#   hidden_15 => tanh_14
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_14 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_14]
#   %tanh_14 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_14]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_15 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 14), kwargs = {})
#   %add_14 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_15, %mm_14), kwargs = {})
#   %tanh_14 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_14,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_14,%buf80
triton_poi_fused_add_addmm_select_stack_tanh_view_16 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_16(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (3584 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/cq/ccqo5zoq5slmfsoq45wuwvemeh6bime6n22hhfnerqnl2pglolcg.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_16, add_15, hidden_16, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_15 => add_15
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_16 => select_16
#   getitem_32 => select_32
#   hidden_16 => tanh_15
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_15 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_15]
#   %tanh_15 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_15]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_16 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 15), kwargs = {})
#   %add_15 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_16, %mm_15), kwargs = {})
#   %tanh_15 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_15,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_15,%buf81
triton_poi_fused_add_addmm_select_stack_tanh_view_17 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_17(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (3840 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/no/cnoulvq2k342nqnkrngr5vvis2ivjp7qzyzggcukom3n5z4wd43e.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_17, add_16, hidden_17, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_16 => add_16
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_17 => select_17
#   getitem_32 => select_32
#   hidden_17 => tanh_16
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_16 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_16]
#   %tanh_16 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_16]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_17 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 16), kwargs = {})
#   %add_16 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_17, %mm_16), kwargs = {})
#   %tanh_16 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_16,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_16,%buf82
triton_poi_fused_add_addmm_select_stack_tanh_view_18 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4096 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mq/cmqebxg23mvm7si3sqmh2p63eg7jv6slrh6xdkelphioulaolbid.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_18, add_17, hidden_18, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_17 => add_17
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_18 => select_18
#   getitem_32 => select_32
#   hidden_18 => tanh_17
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_17 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %tanh_17 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_17]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_18 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 17), kwargs = {})
#   %add_17 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_18, %mm_17), kwargs = {})
#   %tanh_17 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_17,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_17,%buf83
triton_poi_fused_add_addmm_select_stack_tanh_view_19 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_19(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4352 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/nu/cnuqx2f2ogfodj2h27xxjvay4akah5lipa4ytopdk5jb326eed4s.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_19, add_18, hidden_19, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_18 => add_18
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_19 => select_19
#   getitem_32 => select_32
#   hidden_19 => tanh_18
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_18 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_18]
#   %tanh_18 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_18]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_19 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 18), kwargs = {})
#   %add_18 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_19, %mm_18), kwargs = {})
#   %tanh_18 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_18,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_18,%buf84
triton_poi_fused_add_addmm_select_stack_tanh_view_20 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_20(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4608 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dp/cdpbm42o5l2gybbglmyvm2km7dcmxbt6eaisu633kchg4au4a4ch.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_20, add_19, hidden_20, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_19 => add_19
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_20 => select_20
#   getitem_32 => select_32
#   hidden_20 => tanh_19
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_19 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_19]
#   %tanh_19 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_19]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_20 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 19), kwargs = {})
#   %add_19 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_20, %mm_19), kwargs = {})
#   %tanh_19 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_19,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_19,%buf85
triton_poi_fused_add_addmm_select_stack_tanh_view_21 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_21(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4864 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6r/c6ruippanqxsvcftwh3f44qi5cxdohrvt2cyalx635vrxlomonap.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_21, add_20, hidden_21, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_20 => add_20
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_21 => select_21
#   getitem_32 => select_32
#   hidden_21 => tanh_20
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_20 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %tanh_20 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_20]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_21 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 20), kwargs = {})
#   %add_20 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_21, %mm_20), kwargs = {})
#   %tanh_20 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_20,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_20,%buf86
triton_poi_fused_add_addmm_select_stack_tanh_view_22 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_22(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5120 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/uh/cuhxmc77pnmaswzyk53tnvjat4epmxbsksewd567q6jmzvdn3y6t.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_22, add_21, hidden_22, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_21 => add_21
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_22 => select_22
#   getitem_32 => select_32
#   hidden_22 => tanh_21
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_21 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_21]
#   %tanh_21 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_21]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_22 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 21), kwargs = {})
#   %add_21 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_22, %mm_21), kwargs = {})
#   %tanh_21 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_21,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_21,%buf87
triton_poi_fused_add_addmm_select_stack_tanh_view_23 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_23(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5376 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ug/cugkwv36dztqzmal55xukdlwkkczi565ja4iasuemwfyaygqczdm.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_23, add_22, hidden_23, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_22 => add_22
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_23 => select_23
#   getitem_32 => select_32
#   hidden_23 => tanh_22
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_22 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_22]
#   %tanh_22 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_22]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_23 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 22), kwargs = {})
#   %add_22 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_23, %mm_22), kwargs = {})
#   %tanh_22 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_22,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_22,%buf88
triton_poi_fused_add_addmm_select_stack_tanh_view_24 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_24(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5632 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/4g/c4gbymnsvqm3ihcws3tzcc3cgwqpufshsphjdl2ho3uhfnc3ss7p.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_24, add_23, hidden_24, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_23 => add_23
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_24 => select_24
#   getitem_32 => select_32
#   hidden_24 => tanh_23
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_23 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_23]
#   %tanh_23 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_23]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_24 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 23), kwargs = {})
#   %add_23 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_24, %mm_23), kwargs = {})
#   %tanh_23 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_23,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_23,%buf89
triton_poi_fused_add_addmm_select_stack_tanh_view_25 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_25(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5888 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/h5/ch5ki4wfjqdge54czuo6sy2czxzta7336oezchofo2mo4254edee.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_25, add_24, hidden_25, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_24 => add_24
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_25 => select_25
#   getitem_32 => select_32
#   hidden_25 => tanh_24
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_24 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_24]
#   %tanh_24 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_24]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_25 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 24), kwargs = {})
#   %add_24 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_25, %mm_24), kwargs = {})
#   %tanh_24 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_24,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_24,%buf90
triton_poi_fused_add_addmm_select_stack_tanh_view_26 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_26(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (6144 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/t5/ct5uhel6cxcbyltovaza7znzehyurfblynkw4jmdlxt4owx3nr6n.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_26, add_25, hidden_26, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_25 => add_25
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_26 => select_26
#   getitem_32 => select_32
#   hidden_26 => tanh_25
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_25 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_25]
#   %tanh_25 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_25]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_26 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 25), kwargs = {})
#   %add_25 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_26, %mm_25), kwargs = {})
#   %tanh_25 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_25,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_25,%buf91
triton_poi_fused_add_addmm_select_stack_tanh_view_27 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_27(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (6400 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/6u/c6ua4lk4qfxi3c7g3abdajdpwlkn4y7box2f7cj3mwyueb6el3vd.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_27, add_26, hidden_27, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_26 => add_26
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_27 => select_27
#   getitem_32 => select_32
#   hidden_27 => tanh_26
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_26 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_26]
#   %tanh_26 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_26]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_27 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 26), kwargs = {})
#   %add_26 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_27, %mm_26), kwargs = {})
#   %tanh_26 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_26,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_26,%buf92
triton_poi_fused_add_addmm_select_stack_tanh_view_28 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_28(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (6656 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/pn/cpnalaj6zsy6skexiw2hjqpg5xehpql5w4coabtigm7fs4zlphsh.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_28, add_27, hidden_28, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_27 => add_27
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_28 => select_28
#   getitem_32 => select_32
#   hidden_28 => tanh_27
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_27 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_27]
#   %tanh_27 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_27]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_28 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 27), kwargs = {})
#   %add_27 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_28, %mm_27), kwargs = {})
#   %tanh_27 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_27,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_27,%buf93
triton_poi_fused_add_addmm_select_stack_tanh_view_29 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_29(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (6912 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/se/csenqoikix7d6lmpjtmv5jssb6vjvfkrxagqexmcf3fql5qxmetg.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_29, add_28, hidden_29, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_28 => add_28
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_29 => select_29
#   getitem_32 => select_32
#   hidden_29 => tanh_28
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_28 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_28]
#   %tanh_28 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_28]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_29 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 28), kwargs = {})
#   %add_28 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_29, %mm_28), kwargs = {})
#   %tanh_28 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_28,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_28,%buf94
triton_poi_fused_add_addmm_select_stack_tanh_view_30 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_30(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7168 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ac/cacpazj4qrhom7kp7dnfuqc2d7aijj5z23wgteryrnifwkg3f47f.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_30, add_29, hidden_30, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_29 => add_29
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_30 => select_30
#   getitem_32 => select_32
#   hidden_30 => tanh_29
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_29 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_29]
#   %tanh_29 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_29]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_30 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 29), kwargs = {})
#   %add_29 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_30, %mm_29), kwargs = {})
#   %tanh_29 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_29,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_29,%buf95
triton_poi_fused_add_addmm_select_stack_tanh_view_31 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_31(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7424 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/st/cstrmxifwulyxem3jbqmmjrr6qgzbluequn76n6q432gvrp4azt6.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_31, add_30, hidden_31, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
# Source node to ATen node mapping:
#   add_30 => add_30
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_31 => select_31
#   getitem_32 => select_32
#   hidden_31 => tanh_30
#   hidden_32 => tanh_31
#   states => cat
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_30 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_30]
#   %tanh_30 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=tanh_30]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %select_31 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 30), kwargs = {})
#   %add_30 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_31, %mm_30), kwargs = {})
#   %tanh_30 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_30,), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tanh, %tanh_1, %tanh_2, %tanh_3, %tanh_4, %tanh_5, %tanh_6, %tanh_7, %tanh_8, %tanh_9, %tanh_10, %tanh_11, %tanh_12, %tanh_13, %tanh_14, %tanh_15, %tanh_16, %tanh_17, %tanh_18, %tanh_19, %tanh_20, %tanh_21, %tanh_22, %tanh_23, %tanh_24, %tanh_25, %tanh_26, %tanh_27, %tanh_28, %tanh_29, %tanh_30, %tanh_31], 1), kwargs = {})
#   return %tanh_30,%buf96
triton_poi_fused_add_addmm_select_stack_tanh_view_32 = async_compile.triton('triton_poi_fused_add_addmm_select_stack_tanh_view_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_stack_tanh_view_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_stack_tanh_view_32(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7680 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/o3/co3sgfwmpejz3kkuzkbo2t6vzf4jompczhjutv5sg5w5ouuy6txc.py
# Topologically Sorted Source Nodes: [embedded, getitem_32, add_31, hidden_32], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_31 => add_31
#   embedded => add_tensor_1, view_1
#   getitem_32 => select_32
#   hidden_32 => tanh_31
# Graph fragment:
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm_default_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %mm_31 : Tensor "f32[64, 256][256, 1]cuda:0" = PlaceHolder[target=mm_31]
#   %add_tensor_1 : Tensor "f32[2048, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=32] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [64, 32, 256]), kwargs = {})
#   %select_32 : Tensor "f32[64, 256][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%view_1, 1, 31), kwargs = {})
#   %add_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mm_31), kwargs = {})
#   %tanh_31 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add_31,), kwargs = {})
#   return %tanh_31
triton_poi_fused_add_addmm_select_tanh_view_33 = async_compile.triton('triton_poi_fused_add_addmm_select_tanh_view_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_select_tanh_view_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 263168}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_select_tanh_view_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (7936 + x0 + 8192*x1), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(out_ptr0 + (x0 + 8192*x1), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/lv/clvueyq5nnky4ukupit36o3bi4c45k33m2lozxrk4y4nsjkjmzvw.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.view, aten.tanh]
# Source node to ATen node mapping:
#   input_1 => add_tensor, view_4
#   input_2 => tanh_32
# Graph fragment:
#   %arg6_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %mm_default : Tensor "f32[2048, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_tensor : Tensor "f32[2048, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, %mm_default), kwargs = {})
#   %view_4 : Tensor "f32[64, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [64, 32, 128]), kwargs = {})
#   %tanh_32 : Tensor "f32[64, 32, 128][4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%view_4,), kwargs = {})
#   return %tanh_32
triton_poi_fused_addmm_tanh_view_34 = async_compile.triton('triton_poi_fused_addmm_tanh_view_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_tanh_view_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3146240}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_tanh_view_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tg/ctg3uzse2nwefah7ifz6svrc5ruz7c3eh5mhaszqdbf37czmnzpj.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.view, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   input_3 => view_6
# Graph fragment:
#   %addmm_2 : Tensor "f32[2048, 1][1, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %view_6 : Tensor "f32[64, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [64, 32, 1]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_6, 1), kwargs = {})
#   return %getitem,%getitem_1
triton_per_fused_prepare_softmax_online_view_35 = async_compile.triton('triton_per_fused_prepare_softmax_online_view_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_prepare_softmax_online_view_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 1024, 'r0_': 8192}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_prepare_softmax_online_view_35(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tmp1 - tmp6
    tmp8 = libdevice.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/po/cpoerqjbgymkbcqlbmwvijjztdeftfi3z3dyvjcz3sxjmhyxdd6d.py
# Topologically Sorted Source Nodes: [states, input_3, weights, mul, sum_1], Original ATen: [aten.stack, aten.view, aten._softmax, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   input_3 => view_6
#   mul => mul
#   states => view_2
#   sum_1 => sum_2
#   weights => div, exp_default, sub_tensor
# Graph fragment:
#   %cat : Tensor "f32[64, 8192][8192, 1]cuda:0" = PlaceHolder[target=cat]
#   %addmm_2 : Tensor "f32[2048, 1][1, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %getitem : Tensor "f32[64, 1, 1][1, 64, 64]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[64, 1, 1][1, 64, 64]cuda:0" = PlaceHolder[target=getitem_1]
#   %view_2 : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [64, 32, 256]), kwargs = {})
#   %view_6 : Tensor "f32[64, 32, 1][32, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [64, 32, 1]), kwargs = {})
#   %sub_tensor : Tensor "f32[64, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[64, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[64, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[64, 32, 256][8192, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %div), kwargs = {})
#   %sum_2 : Tensor "f32[64, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   return %sum_2
triton_per_fused__softmax_mul_stack_sum_view_36 = async_compile.triton('triton_per_fused__softmax_mul_stack_sum_view_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax_mul_stack_sum_view_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 2228224, 'r0_': 8192}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__softmax_mul_stack_sum_view_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 8192*x1), None, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + (r0_2 + 32*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.max2(tmp1, 1)[:, None].to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp10, None)
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
        assert_size_stride(arg1_1, (64, 32), (32, 1), 'input')
        assert_size_stride(arg0_1, (8192, 256), (256, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf0 = empty_strided_cuda((64, 32, 256), (8192, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embedding], Original ATen: [aten.embedding]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_0.run(arg1_1, arg0_1, buf0, 524288, stream=raw_stream0)
            del arg0_1
            del arg1_1
            assert_size_stride(arg2_1, (256, 256), (256, 1), 'input')
            buf1 = empty_strided_cuda((2048, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [embedding, embedded], Original ATen: [aten.embedding, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf0, (2048, 256), (256, 1), 0), reinterpret_tensor(arg2_1, (256, 256), (1, 256), 0), out=buf1)
            del arg2_1
            del buf0
            assert_size_stride(arg4_1, (256, 256), (256, 1), 'input')
            assert_size_stride(arg3_1, (256, ), (1, ), 'input')
            buf4 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
            buf98 = empty_strided_cuda((64, 8192), (8192, 1), torch.float32)
            buf66 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_1, add, hidden_1, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_2.run(buf4, arg3_1, buf1, buf66, 16384, stream=raw_stream0)
            buf5 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf4, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf5)
            buf6 = buf5; del buf5  # reuse
            buf67 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 256)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_2, add_1, hidden_2, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_3.run(buf6, arg3_1, buf1, buf67, 16384, stream=raw_stream0)
            buf7 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf6, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf7)
            buf8 = buf7; del buf7  # reuse
            buf68 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 512)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_3, add_2, hidden_3, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_4.run(buf8, arg3_1, buf1, buf68, 16384, stream=raw_stream0)
            buf9 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf8, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf9)
            buf10 = buf9; del buf9  # reuse
            buf69 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 768)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_4, add_3, hidden_4, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_5.run(buf10, arg3_1, buf1, buf69, 16384, stream=raw_stream0)
            buf11 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf10, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf11)
            buf12 = buf11; del buf11  # reuse
            buf70 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 1024)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_5, add_4, hidden_5, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_6.run(buf12, arg3_1, buf1, buf70, 16384, stream=raw_stream0)
            buf13 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf12, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf13)
            buf14 = buf13; del buf13  # reuse
            buf71 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 1280)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_6, add_5, hidden_6, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_7.run(buf14, arg3_1, buf1, buf71, 16384, stream=raw_stream0)
            buf15 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf14, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf15)
            buf16 = buf15; del buf15  # reuse
            buf72 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 1536)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_7, add_6, hidden_7, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_8.run(buf16, arg3_1, buf1, buf72, 16384, stream=raw_stream0)
            buf17 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf16, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf17)
            buf18 = buf17; del buf17  # reuse
            buf73 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 1792)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_8, add_7, hidden_8, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_9.run(buf18, arg3_1, buf1, buf73, 16384, stream=raw_stream0)
            buf19 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf18, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf19)
            buf20 = buf19; del buf19  # reuse
            buf74 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 2048)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_9, add_8, hidden_9, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_10.run(buf20, arg3_1, buf1, buf74, 16384, stream=raw_stream0)
            buf21 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf20, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf21)
            buf22 = buf21; del buf21  # reuse
            buf75 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 2304)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_10, add_9, hidden_10, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_11.run(buf22, arg3_1, buf1, buf75, 16384, stream=raw_stream0)
            buf23 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf22, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf23)
            buf24 = buf23; del buf23  # reuse
            buf76 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 2560)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_11, add_10, hidden_11, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_12.run(buf24, arg3_1, buf1, buf76, 16384, stream=raw_stream0)
            buf25 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf24, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf25)
            buf26 = buf25; del buf25  # reuse
            buf77 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 2816)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_12, add_11, hidden_12, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_13.run(buf26, arg3_1, buf1, buf77, 16384, stream=raw_stream0)
            buf27 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf26, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf27)
            buf28 = buf27; del buf27  # reuse
            buf78 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 3072)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_13, add_12, hidden_13, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_14.run(buf28, arg3_1, buf1, buf78, 16384, stream=raw_stream0)
            buf29 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf28, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf29)
            buf30 = buf29; del buf29  # reuse
            buf79 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 3328)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_14, add_13, hidden_14, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_15.run(buf30, arg3_1, buf1, buf79, 16384, stream=raw_stream0)
            buf31 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf30, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf31)
            buf32 = buf31; del buf31  # reuse
            buf80 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 3584)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_15, add_14, hidden_15, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_16.run(buf32, arg3_1, buf1, buf80, 16384, stream=raw_stream0)
            buf33 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf32, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf33)
            buf34 = buf33; del buf33  # reuse
            buf81 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 3840)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_16, add_15, hidden_16, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_17.run(buf34, arg3_1, buf1, buf81, 16384, stream=raw_stream0)
            buf35 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf34, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf35)
            buf36 = buf35; del buf35  # reuse
            buf82 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 4096)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_17, add_16, hidden_17, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_18.run(buf36, arg3_1, buf1, buf82, 16384, stream=raw_stream0)
            buf37 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf36, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf37)
            buf38 = buf37; del buf37  # reuse
            buf83 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 4352)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_18, add_17, hidden_18, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_19.run(buf38, arg3_1, buf1, buf83, 16384, stream=raw_stream0)
            buf39 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf38, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf39)
            buf40 = buf39; del buf39  # reuse
            buf84 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 4608)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_19, add_18, hidden_19, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_20.run(buf40, arg3_1, buf1, buf84, 16384, stream=raw_stream0)
            buf41 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf40, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf41)
            buf42 = buf41; del buf41  # reuse
            buf85 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 4864)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_20, add_19, hidden_20, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_21.run(buf42, arg3_1, buf1, buf85, 16384, stream=raw_stream0)
            buf43 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf42, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf43)
            buf44 = buf43; del buf43  # reuse
            buf86 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 5120)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_21, add_20, hidden_21, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_22.run(buf44, arg3_1, buf1, buf86, 16384, stream=raw_stream0)
            buf45 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf44, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf45)
            buf46 = buf45; del buf45  # reuse
            buf87 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 5376)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_22, add_21, hidden_22, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_23.run(buf46, arg3_1, buf1, buf87, 16384, stream=raw_stream0)
            buf47 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf46, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf47)
            buf48 = buf47; del buf47  # reuse
            buf88 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 5632)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_23, add_22, hidden_23, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_24.run(buf48, arg3_1, buf1, buf88, 16384, stream=raw_stream0)
            buf49 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf48, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf49)
            buf50 = buf49; del buf49  # reuse
            buf89 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 5888)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_24, add_23, hidden_24, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_25.run(buf50, arg3_1, buf1, buf89, 16384, stream=raw_stream0)
            buf51 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf50, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf51)
            buf52 = buf51; del buf51  # reuse
            buf90 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 6144)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_25, add_24, hidden_25, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_26.run(buf52, arg3_1, buf1, buf90, 16384, stream=raw_stream0)
            buf53 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf52, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf53)
            buf54 = buf53; del buf53  # reuse
            buf91 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 6400)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_26, add_25, hidden_26, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_27.run(buf54, arg3_1, buf1, buf91, 16384, stream=raw_stream0)
            buf55 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf54, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf55)
            buf56 = buf55; del buf55  # reuse
            buf92 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 6656)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_27, add_26, hidden_27, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_28.run(buf56, arg3_1, buf1, buf92, 16384, stream=raw_stream0)
            buf57 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf56, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf57)
            buf58 = buf57; del buf57  # reuse
            buf93 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 6912)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_28, add_27, hidden_28, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_29.run(buf58, arg3_1, buf1, buf93, 16384, stream=raw_stream0)
            buf59 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf58, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf59)
            buf60 = buf59; del buf59  # reuse
            buf94 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 7168)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_29, add_28, hidden_29, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_30.run(buf60, arg3_1, buf1, buf94, 16384, stream=raw_stream0)
            buf61 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf60, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf61)
            buf62 = buf61; del buf61  # reuse
            buf95 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 7424)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_30, add_29, hidden_30, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_31.run(buf62, arg3_1, buf1, buf95, 16384, stream=raw_stream0)
            buf63 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf62, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf63)
            del buf62
            buf64 = buf63; del buf63  # reuse
            buf96 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 7680)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, getitem_31, add_30, hidden_31, add_31, hidden_32, states], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh, aten.stack]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_stack_tanh_view_32.run(buf64, arg3_1, buf1, buf96, 16384, stream=raw_stream0)
            buf65 = empty_strided_cuda((64, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf64, reinterpret_tensor(arg4_1, (256, 256), (1, 256), 0), out=buf65)
            del arg4_1
            del buf64
            buf97 = reinterpret_tensor(buf98, (64, 256), (8192, 1), 7936)  # alias
            # Topologically Sorted Source Nodes: [embedded, getitem_32, add_31, hidden_32], Original ATen: [aten.addmm, aten.view, aten.select, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_select_tanh_view_33.run(arg3_1, buf1, buf65, buf97, 16384, stream=raw_stream0)
            del arg3_1
            del buf1
            assert_size_stride(arg5_1, (128, 256), (256, 1), 'input')
            del buf66
            del buf67
            del buf68
            del buf69
            del buf70
            del buf71
            del buf72
            del buf73
            del buf74
            del buf75
            del buf76
            del buf77
            del buf78
            del buf79
            del buf80
            del buf81
            del buf82
            del buf83
            del buf84
            del buf85
            del buf86
            del buf87
            del buf88
            del buf89
            del buf90
            del buf91
            del buf92
            del buf93
            del buf94
            del buf95
            del buf96
            del buf97
            buf99 = empty_strided_cuda((2048, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [states, input_1], Original ATen: [aten.stack, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf98, (2048, 256), (256, 1), 0), reinterpret_tensor(arg5_1, (256, 128), (1, 256), 0), out=buf99)
            del arg5_1
            assert_size_stride(arg6_1, (128, ), (1, ), 'input')
            buf100 = reinterpret_tensor(buf99, (64, 32, 128), (4096, 128, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.addmm, aten.view, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_tanh_view_34.run(buf100, arg6_1, 262144, stream=raw_stream0)
            del arg6_1
            assert_size_stride(arg8_1, (1, ), (1, ), 'input')
            assert_size_stride(arg7_1, (1, 128), (128, 1), 'input')
            buf102 = empty_strided_cuda((2048, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.addmm, aten.view, aten.tanh, aten.t]
            extern_kernels.addmm(arg8_1, reinterpret_tensor(buf100, (2048, 128), (128, 1), 0), reinterpret_tensor(arg7_1, (128, 1), (1, 128), 0), alpha=1, beta=1, out=buf102)
            del arg7_1
            del arg8_1
            del buf100
            buf105 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [states, input_3, weights, mul, sum_1], Original ATen: [aten.stack, aten.view, aten._softmax, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax_mul_stack_sum_view_36.run(buf98, buf102, buf102, buf102, buf105, 16384, 32, stream=raw_stream0)
            del buf102
            del buf98
            assert_size_stride(arg10_1, (4, ), (1, ), 'input')
            assert_size_stride(arg9_1, (4, 256), (256, 1), 'input')
            buf106 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.t, aten.addmm]
            extern_kernels.addmm(arg10_1, buf105, reinterpret_tensor(arg9_1, (256, 4), (1, 256), 0), alpha=1, beta=1, out=buf106)
            del arg10_1
            del arg9_1
            del buf105
        return (buf106, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 32), (32, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
