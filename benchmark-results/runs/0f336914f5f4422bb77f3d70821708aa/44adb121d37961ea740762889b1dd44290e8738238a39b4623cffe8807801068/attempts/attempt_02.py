# --- TILESMITH PYTORCH FX REFERENCE (NOT EXECUTED) ---
# Captured from graph_module.code for this compiled graph.
#
#
#
# def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1):
#     embedding = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
#     var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
#     getitem = var_mean[0]
#     getitem_1 = var_mean[1];  var_mean = None
#     add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
#     rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
#     sub = torch.ops.aten.sub.Tensor(embedding, getitem_1);  getitem_1 = None
#     mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
#     mul_1 = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
#     add_1 = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
#     permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
#     view = torch.ops.aten.view.default(add_1, [1024, 256]);  add_1 = None
#     mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
#     view_1 = torch.ops.aten.view.default(mm, [8, 128, 768]);  mm = None
#     view_2 = torch.ops.aten.view.default(view_1, [8, 128, 3, 8, 32]);  view_1 = None
#     unbind = torch.ops.aten.unbind.int(view_2, 2);  view_2 = None
#     getitem_2 = unbind[0]
#     getitem_3 = unbind[1]
#     getitem_4 = unbind[2];  unbind = None
#     permute_1 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
#     permute_2 = torch.ops.aten.permute.default(getitem_3, [0, 2, 1, 3]);  getitem_3 = None
#     permute_3 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
#     permute_4 = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
#     expand = torch.ops.aten.expand.default(permute_1, [8, 8, 128, 32]);  permute_1 = None
#     clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
#     view_3 = torch.ops.aten.view.default(clone, [64, 128, 32]);  clone = None
#     expand_1 = torch.ops.aten.expand.default(permute_4, [8, 8, 32, 128]);  permute_4 = None
#     clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
#     view_4 = torch.ops.aten.view.default(clone_1, [64, 32, 128]);  clone_1 = None
#     bmm = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
#     view_5 = torch.ops.aten.view.default(bmm, [8, 8, 128, 128]);  bmm = None
#     div = torch.ops.aten.div.Tensor(view_5, 5.656854249492381);  view_5 = None
#     full_default = torch.ops.aten.full.default([128, 128], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
#     iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
#     unsqueeze = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
#     iota_1 = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
#     unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
#     sub_1 = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
#     le = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
#     logical_and = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None
#     bitwise_not = torch.ops.aten.bitwise_not.default(logical_and);  logical_and = None
#     full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
#     where = torch.ops.aten.where.self(bitwise_not, full_default_1, div);  bitwise_not = full_default_1 = div = None
#     amax = torch.ops.aten.amax.default(where, [-1], True)
#     sub_2 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
#     exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
#     sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
#     div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
#     expand_2 = torch.ops.aten.expand.default(div_1, [8, 8, 128, 128]);  div_1 = None
#     view_6 = torch.ops.aten.view.default(expand_2, [64, 128, 128]);  expand_2 = None
#     expand_3 = torch.ops.aten.expand.default(permute_3, [8, 8, 128, 32]);  permute_3 = None
#     clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
#     view_7 = torch.ops.aten.view.default(clone_2, [64, 128, 32]);  clone_2 = None
#     bmm_1 = torch.ops.aten.bmm.default(view_6, view_7);  view_6 = view_7 = None
#     view_8 = torch.ops.aten.view.default(bmm_1, [8, 8, 128, 32]);  bmm_1 = None
#     permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
#     clone_3 = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
#     view_9 = torch.ops.aten.view.default(clone_3, [8, 128, 256]);  clone_3 = None
#     permute_6 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
#     view_10 = torch.ops.aten.view.default(view_9, [1024, 256]);  view_9 = None
#     mm_1 = torch.ops.aten.mm.default(view_10, permute_6);  view_10 = permute_6 = None
#     view_11 = torch.ops.aten.view.default(mm_1, [8, 128, 256]);  mm_1 = None
#     add_2 = torch.ops.aten.add.Tensor(embedding, view_11);  embedding = view_11 = None
#     var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
#     getitem_5 = var_mean_1[0]
#     getitem_6 = var_mean_1[1];  var_mean_1 = None
#     add_3 = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
#     rsqrt_1 = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
#     sub_3 = torch.ops.aten.sub.Tensor(add_2, getitem_6);  getitem_6 = None
#     mul_2 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
#     mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
#     add_4 = torch.ops.aten.add.Tensor(mul_3, arg7_1);  mul_3 = arg7_1 = None
#     permute_7 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
#     view_12 = torch.ops.aten.view.default(add_4, [1024, 256])
#     mm_2 = torch.ops.aten.mm.default(view_12, permute_7);  view_12 = permute_7 = None
#     view_13 = torch.ops.aten.view.default(mm_2, [8, 128, 4]);  mm_2 = None
#     amax_1 = torch.ops.aten.amax.default(view_13, [-1], True)
#     sub_4 = torch.ops.aten.sub.Tensor(view_13, amax_1);  view_13 = amax_1 = None
#     exp_1 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
#     sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
#     div_2 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
#     topk = torch.ops.aten.topk.default(div_2, 2);  div_2 = None
#     getitem_7 = topk[0]
#     getitem_8 = topk[1];  topk = None
#     ge = torch.ops.aten.ge.Scalar(getitem_8, 0)
#     logical_not = torch.ops.aten.logical_not.default(ge);  ge = None
#     any_1 = torch.ops.aten.any.dims(logical_not);  logical_not = None
#     logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
#     _assert_async = torch.ops.aten._assert_async.msg(logical_not_1, 'one_hot: Class values must be non-negative.');  logical_not_1 = _assert_async = None
#     lt = torch.ops.aten.lt.Scalar(getitem_8, 4)
#     logical_not_2 = torch.ops.aten.logical_not.default(lt);  lt = None
#     any_2 = torch.ops.aten.any.dims(logical_not_2);  logical_not_2 = None
#     logical_not_3 = torch.ops.aten.logical_not.default(any_2);  any_2 = None
#     _assert_async_1 = torch.ops.aten._assert_async.msg(logical_not_3, 'one_hot: Class values must be smaller than num_classes.');  logical_not_3 = _assert_async_1 = None
#     unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_8, -1);  getitem_8 = None
#     iota_2 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
#     eq = torch.ops.aten.eq.Tensor(unsqueeze_2, iota_2);  unsqueeze_2 = iota_2 = None
#     convert_element_type = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
#     unsqueeze_3 = torch.ops.aten.unsqueeze.default(getitem_7, -1);  getitem_7 = None
#     mul_4 = torch.ops.aten.mul.Tensor(convert_element_type, unsqueeze_3);  convert_element_type = unsqueeze_3 = None
#     sum_3 = torch.ops.aten.sum.dim_IntList(mul_4, [-2]);  mul_4 = None
#     permute_8 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
#     view_14 = torch.ops.aten.view.default(add_4, [1024, 256])
#     mm_3 = torch.ops.aten.mm.default(view_14, permute_8);  view_14 = permute_8 = None
#     view_15 = torch.ops.aten.view.default(mm_3, [8, 128, 512]);  mm_3 = None
#     mul_5 = torch.ops.aten.mul.Tensor(view_15, 0.5)
#     mul_6 = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
#     erf = torch.ops.aten.erf.default(mul_6);  mul_6 = None
#     add_5 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
#     mul_7 = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
#     permute_9 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
#     view_16 = torch.ops.aten.view.default(mul_7, [1024, 512]);  mul_7 = None
#     mm_4 = torch.ops.aten.mm.default(view_16, permute_9);  view_16 = permute_9 = None
#     view_17 = torch.ops.aten.view.default(mm_4, [8, 128, 256]);  mm_4 = None
#     permute_10 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
#     view_18 = torch.ops.aten.view.default(add_4, [1024, 256])
#     mm_5 = torch.ops.aten.mm.default(view_18, permute_10);  view_18 = permute_10 = None
#     view_19 = torch.ops.aten.view.default(mm_5, [8, 128, 512]);  mm_5 = None
#     mul_8 = torch.ops.aten.mul.Tensor(view_19, 0.5)
#     mul_9 = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
#     erf_1 = torch.ops.aten.erf.default(mul_9);  mul_9 = None
#     add_6 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
#     mul_10 = torch.ops.aten.mul.Tensor(mul_8, add_6);  mul_8 = add_6 = None
#     permute_11 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
#     view_20 = torch.ops.aten.view.default(mul_10, [1024, 512]);  mul_10 = None
#     mm_6 = torch.ops.aten.mm.default(view_20, permute_11);  view_20 = permute_11 = None
#     view_21 = torch.ops.aten.view.default(mm_6, [8, 128, 256]);  mm_6 = None
#     permute_12 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
#     view_22 = torch.ops.aten.view.default(add_4, [1024, 256])
#     mm_7 = torch.ops.aten.mm.default(view_22, permute_12);  view_22 = permute_12 = None
#     view_23 = torch.ops.aten.view.default(mm_7, [8, 128, 512]);  mm_7 = None
#     mul_11 = torch.ops.aten.mul.Tensor(view_23, 0.5)
#     mul_12 = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476);  view_23 = None
#     erf_2 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
#     add_7 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
#     mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_7);  mul_11 = add_7 = None
#     permute_13 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
#     view_24 = torch.ops.aten.view.default(mul_13, [1024, 512]);  mul_13 = None
#     mm_8 = torch.ops.aten.mm.default(view_24, permute_13);  view_24 = permute_13 = None
#     view_25 = torch.ops.aten.view.default(mm_8, [8, 128, 256]);  mm_8 = None
#     permute_14 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
#     view_26 = torch.ops.aten.view.default(add_4, [1024, 256]);  add_4 = None
#     mm_9 = torch.ops.aten.mm.default(view_26, permute_14);  view_26 = permute_14 = None
#     view_27 = torch.ops.aten.view.default(mm_9, [8, 128, 512]);  mm_9 = None
#     mul_14 = torch.ops.aten.mul.Tensor(view_27, 0.5)
#     mul_15 = torch.ops.aten.mul.Tensor(view_27, 0.7071067811865476);  view_27 = None
#     erf_3 = torch.ops.aten.erf.default(mul_15);  mul_15 = None
#     add_8 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
#     mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_8);  mul_14 = add_8 = None
#     permute_15 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
#     view_28 = torch.ops.aten.view.default(mul_16, [1024, 512]);  mul_16 = None
#     mm_10 = torch.ops.aten.mm.default(view_28, permute_15);  view_28 = permute_15 = None
#     view_29 = torch.ops.aten.view.default(mm_10, [8, 128, 256]);  mm_10 = None
#     cat = torch.ops.aten.cat.default([view_17, view_21, view_25, view_29], 2);  view_17 = view_21 = view_25 = view_29 = None
#     view_30 = torch.ops.aten.view.default(cat, [8, 128, 4, 256]);  cat = None
#     unsqueeze_4 = torch.ops.aten.unsqueeze.default(sum_3, -1);  sum_3 = None
#     mul_17 = torch.ops.aten.mul.Tensor(view_30, unsqueeze_4);  view_30 = unsqueeze_4 = None
#     sum_4 = torch.ops.aten.sum.dim_IntList(mul_17, [-2]);  mul_17 = None
#     add_9 = torch.ops.aten.add.Tensor(add_2, sum_4);  add_2 = sum_4 = None
#     var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
#     getitem_9 = var_mean_2[0]
#     getitem_10 = var_mean_2[1];  var_mean_2 = None
#     add_10 = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
#     rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
#     sub_5 = torch.ops.aten.sub.Tensor(add_9, getitem_10);  add_9 = getitem_10 = None
#     mul_18 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
#     mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg17_1);  mul_18 = arg17_1 = None
#     add_11 = torch.ops.aten.add.Tensor(mul_19, arg18_1);  mul_19 = arg18_1 = None
#     permute_16 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
#     view_31 = torch.ops.aten.view.default(add_11, [1024, 256]);  add_11 = None
#     mm_11 = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
#     view_32 = torch.ops.aten.view.default(mm_11, [8, 128, 2048]);  mm_11 = None
#     return (view_32,)
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


# kernel path: /tmp/torchinductor_root/jq/cjqkqmjf4blwrzcew4ejznjuvwanp6s3nit3bj7yrafz6bzsr53l.py
# Topologically Sorted Source Nodes: [hidden, layer_norm], Original ATen: [aten.embedding, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden => embedding
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %getitem_1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf1]
#   %arg2_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %embedding : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%embedding, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %getitem_1), kwargs = {})
#   %add : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg2_1), kwargs = {})
#   %add_1 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg3_1), kwargs = {})
#   return %getitem_1,%buf1,%add_1
triton_per_fused_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 8192, 'r0_': 2099200}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 2048)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 2048")
    tmp2 = tl.load(in_ptr1 + (r0_1 + 256*tmp0), xmask, eviction_policy='evict_first', other=0.0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = (tl.full([1, 1], 256, tl.int32)).to(tl.float32)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tmp3 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tmp2 - tmp11
    tmp19 = tl.full([1, 1], 256.0, tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp21 = tl.full([1, 1], 1e-05, tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rk/crk7aqsmiay724wxxoo6mo7keqybqpsy4iefjf66wgp7ix5ecqh3.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, query_1, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul => clone
#   qkv => view_1
#   qkv_1 => view_2
#   query_1 => permute_1
#   unbind => unbind
# Graph fragment:
#   %mm : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [8, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[8, 128, 3, 8, 32][98304, 768, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [8, 128, 3, 8, 32]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%view_2, 2), kwargs = {})
#   %permute_1 : Tensor "f32[8, 8, 128, 32][98304, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f32[8, 8, 128, 32][32768, 4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused__unsafe_view_clone_transpose_unbind_view_1 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_unbind_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_unbind_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3145728}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_unbind_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 128)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 768*x1 + 98304*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/3l/c3lr22wbesjfcuxdtxwmbqp5pumnyzexvxlnugnu4237aqyt647j.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, key_1, transpose_3, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   key_1 => permute_2
#   matmul => clone_1
#   qkv => view_1
#   qkv_1 => view_2
#   transpose_3 => permute_4
#   unbind => unbind
# Graph fragment:
#   %mm : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [8, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[8, 128, 3, 8, 32][98304, 768, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [8, 128, 3, 8, 32]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%view_2, 2), kwargs = {})
#   %permute_2 : Tensor "f32[8, 8, 128, 32][98304, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 2, 1, 3]), kwargs = {})
#   %permute_4 : Tensor "f32[8, 8, 32, 128][98304, 32, 1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_2, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[8, 8, 32, 128][32768, 4096, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_clone_transpose_unbind_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_unbind_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_unbind_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 1048576, 'x': 2097152}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_unbind_view_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (256 + y0 + 768*x2 + 98304*y1), xmask)
    tl.store(out_ptr0 + (x2 + 128*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ju/cju4xagakngbbi7l3fboghog5f757kxzisvanwi5gzs7co2xu2iw.py
# Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, value_1, matmul_1], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_2
#   qkv => view_1
#   qkv_1 => view_2
#   unbind => unbind
#   value_1 => permute_3
# Graph fragment:
#   %mm : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm]
#   %view_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [8, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[8, 128, 3, 8, 32][98304, 768, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [8, 128, 3, 8, 32]), kwargs = {})
#   %unbind : [num_users=3] = call_function[target=torch.ops.aten.unbind.int](args = (%view_2, 2), kwargs = {})
#   %permute_3 : Tensor "f32[8, 8, 128, 32][98304, 32, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_4, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f32[8, 8, 128, 32][32768, 4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__unsafe_view_clone_transpose_unbind_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_unbind_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_unbind_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3145728}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_unbind_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 128)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + 32*x2 + 768*x1 + 98304*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rv/crvdvz3c5w7sgscwz2jy7vcinv4j42chspavdl6ev5qs76hu3ctc.py
# Topologically Sorted Source Nodes: [causal, ones, invert, masked_fill, matmul, scores, softmax], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.view, aten.div, prims.prepare_softmax_online, aten._softmax]
# Source node to ATen node mapping:
#   causal => iota, iota_1, le, logical_and, sub_1, unsqueeze, unsqueeze_1
#   invert => bitwise_not
#   masked_fill => full_default_1, where
#   matmul => view_5
#   ones => full_default
#   scores => div
#   softmax => div_1, exp_default_1, sub_tensor_1
# Graph fragment:
#   %bmm : Tensor "f32[64, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm]
#   %getitem_13 : Tensor "f32[8, 8, 128, 1][1024, 128, 1, 8192]cuda:0" = PlaceHolder[target=getitem_13]
#   %getitem_14 : Tensor "f32[8, 8, 128, 1][1024, 128, 1, 8192]cuda:0" = PlaceHolder[target=getitem_14]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, -2), kwargs = {})
#   %iota_1 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, -1), kwargs = {})
#   %sub_1 : Tensor "i64[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %le : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_1, 0), kwargs = {})
#   %full_default : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_not : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_5 : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [8, 8, 128, 128]), kwargs = {})
#   %div : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_5, 5.656854249492381), kwargs = {})
#   %where : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default_1, %div), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%where, -1), kwargs = {})
#   %sub_tensor_1 : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %getitem_13), kwargs = {})
#   %exp_default_1 : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_1 : Tensor "f32[8, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_14), kwargs = {})
#   return %getitem_13,%getitem_14,%expand_2
triton_per_fused__softmax_bitwise_not_div_masked_fill_ones_prepare_softmax_online_tril_view_4 = async_compile.triton('triton_per_fused__softmax_bitwise_not_div_masked_fill_ones_prepare_softmax_online_tril_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax_bitwise_not_div_masked_fill_ones_prepare_softmax_online_tril_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 12582912}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__softmax_bitwise_not_div_masked_fill_ones_prepare_softmax_online_tril_view_4(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    x0 = (xindex % 128)
    x3 = xindex
    tmp7 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), None, eviction_policy='evict_first')
    tmp0 = (r0_2 + ((-1)*x0)).to(tl.int64)
    tmp1 = (tmp0).to(tl.int64)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1, 1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp8 = tl.full([1, 1], 0.17677669529663687, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp11 = tl.where(tmp6, tmp10, tmp9)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp16 = triton_helpers.max2(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = tmp12 - tmp16
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp22 = tmp11 - tmp16
    tmp23 = libdevice.exp(tmp22)
    tmp24 = (tmp23 / tmp21)
    tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ar/carsd3nz2jtyznvdbgf5fjkcfx2rb4dzjroifw5hy4mcftxrabb2.py
# Topologically Sorted Source Nodes: [matmul_1, transpose_4, attended], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   attended => clone_3
#   matmul_1 => view_8
#   transpose_4 => permute_5
# Graph fragment:
#   %bmm_1 : Tensor "f32[64, 128, 32][4096, 32, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_8 : Tensor "f32[8, 8, 128, 32][32768, 4096, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [8, 8, 128, 32]), kwargs = {})
#   %permute_5 : Tensor "f32[8, 128, 8, 32][32768, 32, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f32[8, 128, 8, 32][32768, 256, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_transpose_view_5 = async_compile.triton('triton_poi_fused_clone_transpose_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_transpose_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3145728}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 8)
    x2 = ((xindex // 256) % 128)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 4096*x1 + 32768*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xs/cxseogqyf6nam6ruguncu3vf2yuurb3tvgvk36xi3uktexygxczb.py
# Topologically Sorted Source Nodes: [hidden, linear_1, hidden_1, layer_norm_1], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden => embedding
#   hidden_1 => add_2
#   layer_norm_1 => add_3, add_4, mul_2, mul_3, rsqrt_1, sub_3, var_mean_1
#   linear_1 => view_11
# Graph fragment:
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mm_1 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %getitem_6 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_6]
#   %buf16 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf16]
#   %arg6_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %embedding : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %view_11 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [8, 128, 256]), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_11), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_6), kwargs = {})
#   %add_3 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_5, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   %add_4 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg7_1), kwargs = {})
#   return %getitem_6,%buf16,%add_4
triton_per_fused__unsafe_view_add_embedding_native_layer_norm_6 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_native_layer_norm_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 8192, 'r0_': 3147776}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, eviction_policy='evict_first', other=0.0)
    tmp27 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 2048)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 2048")
    tmp2 = tl.load(in_ptr1 + (r0_1 + 256*tmp0), xmask, eviction_policy='evict_first', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = (tl.full([1, 1], 256, tl.int32)).to(tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tmp5 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tmp4 - tmp13
    tmp21 = tl.full([1, 1], 256.0, tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tl.full([1, 1], 1e-05, tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr2 + (r0_1 + 256*x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ei/ceiuweya2o5hk4caa64ejqjonjagaisptzvdrntj4u37j5r4s7i5.py
# Topologically Sorted Source Nodes: [linear_2, routing], Original ATen: [aten._unsafe_view, prims.prepare_softmax_online, aten._softmax]
# Source node to ATen node mapping:
#   linear_2 => view_13
#   routing => div_2, exp_default, sub_tensor
# Graph fragment:
#   %mm_2 : Tensor "f32[1024, 4][4, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_13 : Tensor "f32[8, 128, 4][512, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [8, 128, 4]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_13, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[8, 128, 4][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_13, %getitem_11), kwargs = {})
#   %exp_default : Tensor "f32[8, 128, 4][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_2 : Tensor "f32[8, 128, 4][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_12), kwargs = {})
#   return %div_2
triton_poi_fused__softmax__unsafe_view_prepare_softmax_online_7 = async_compile.triton('triton_poi_fused__softmax__unsafe_view_prepare_softmax_online_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__softmax__unsafe_view_prepare_softmax_online_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 49152}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__unsafe_view_prepare_softmax_online_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (4*x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tmp1 - tmp7
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tmp2 - tmp7
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp4 - tmp7
    tmp16 = libdevice.exp(tmp15)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp6 - tmp7
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = (tmp9 / tmp20)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/h2/ch23nxu55g7ndishmmdn7slrnmux63jnfanc42jdj4eotxedgfl4.py
# Topologically Sorted Source Nodes: [selected], Original ATen: [aten.ge, aten.all, aten._assert_async, aten.lt]
# Source node to ATen node mapping:
#   selected => _assert_async, _assert_async_1, any_1, any_2, ge, logical_not, logical_not_1, logical_not_2, logical_not_3, lt
# Graph fragment:
#   %getitem_8 : Tensor "i64[8, 128, 2][256, 2, 1]cuda:0" = PlaceHolder[target=getitem_8]
#   %any_1 : Tensor "b8[][]cuda:0" = PlaceHolder[target=any_1]
#   %logical_not_1 : Tensor "b8[][]cuda:0" = PlaceHolder[target=logical_not_1]
#   %any_2 : Tensor "b8[][]cuda:0" = PlaceHolder[target=any_2]
#   %logical_not_3 : Tensor "b8[][]cuda:0" = PlaceHolder[target=logical_not_3]
#   %ge : Tensor "b8[8, 128, 2][256, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%getitem_8, 0), kwargs = {})
#   %logical_not : Tensor "b8[8, 128, 2][256, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%ge,), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not,), kwargs = {})
#   %logical_not_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %_assert_async : [num_users=0] = call_function[target=torch.ops.aten._assert_async.msg](args = (%logical_not_1, one_hot: Class values must be non-negative.), kwargs = {})
#   %lt : Tensor "b8[8, 128, 2][256, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%getitem_8, 4), kwargs = {})
#   %logical_not_2 : Tensor "b8[8, 128, 2][256, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%lt,), kwargs = {})
#   %any_2 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.dims](args = (%logical_not_2,), kwargs = {})
#   %logical_not_3 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %_assert_async_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_async.msg](args = (%logical_not_3, one_hot: Class values must be smaller than num_classes.), kwargs = {})
#   return %any_1,%any_2,%logical_not_1,%_assert_async,%logical_not_3,%_assert_async_1
triton_red_fused__assert_async_all_ge_lt_8 = async_compile.triton('triton_red_fused__assert_async_all_ge_lt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__assert_async_all_ge_lt_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': -2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'r0_': 16384}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__assert_async_all_ge_lt_8(in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tmp2 == 0
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 | tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp8 = tl.full([1, 1], 4, tl.int64)
        tmp9 = tmp0 < tmp8
        tmp10 = tmp9 == 0
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 | tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
    tmp7 = _tmp5.to(tl.int8)
    tmp5 = triton_helpers.any(tmp7, 1)[:, None]
    tmp14 = _tmp12.to(tl.int8)
    tmp12 = triton_helpers.any(tmp14, 1)[:, None]
    tmp15 = tmp5 == 0
    tl.device_assert(tmp15, 'one_hot: Class values must be non-negative.')
    tmp16 = tmp12 == 0
    tl.device_assert(tmp16, 'one_hot: Class values must be smaller than num_classes.')
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rd/crdhvbathavo77whrftjshroqsqjk6sje7chx6germnvak2iw5mg.py
# Topologically Sorted Source Nodes: [linear_3, gelu], Original ATen: [aten._unsafe_view, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_5, erf, mul_5, mul_6, mul_7
#   linear_3 => view_15
# Graph fragment:
#   %mm_3 : Tensor "f32[1024, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %view_15 : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [8, 128, 512]), kwargs = {})
#   %mul_5 : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.5), kwargs = {})
#   %mul_6 : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : Tensor "f32[8, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
#   return %mul_7
triton_poi_fused__unsafe_view_gelu_9 = async_compile.triton('triton_poi_fused__unsafe_view_gelu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_gelu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 6291456}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_gelu_9(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/qt/cqtydaybi6mkz3ssj2dspzy4h3evk5pjndunexgqrm67eblkax5g.py
# Topologically Sorted Source Nodes: [linear_4, linear_6, linear_8, linear_10, outputs, selected, unsqueeze, mul, weights, unsqueeze_1, mul_1], Original ATen: [aten._unsafe_view, aten.stack, aten.unsqueeze, aten.arange, aten.eq, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   linear_10 => view_29
#   linear_4 => view_17
#   linear_6 => view_21
#   linear_8 => view_25
#   mul => mul_4
#   mul_1 => mul_17
#   outputs => cat, view_30
#   selected => convert_element_type, eq, iota_2, unsqueeze_2
#   unsqueeze => unsqueeze_3
#   unsqueeze_1 => unsqueeze_4
#   weights => sum_3
# Graph fragment:
#   %mm_4 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mm_6 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %mm_8 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_10 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %getitem_8 : Tensor "i64[8, 128, 2][256, 2, 1]cuda:0" = PlaceHolder[target=getitem_8]
#   %getitem_7 : Tensor "f32[8, 128, 2][256, 2, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %view_17 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [8, 128, 256]), kwargs = {})
#   %view_21 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [8, 128, 256]), kwargs = {})
#   %view_25 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [8, 128, 256]), kwargs = {})
#   %view_29 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [8, 128, 256]), kwargs = {})
#   %cat : Tensor "f32[8, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_17, %view_21, %view_25, %view_29], 2), kwargs = {})
#   %view_30 : Tensor "f32[8, 128, 4, 256][131072, 1024, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [8, 128, 4, 256]), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[8, 128, 2, 1][256, 2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_8, -1), kwargs = {})
#   %iota_2 : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : Tensor "b8[8, 128, 2, 4][1024, 8, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_2, %iota_2), kwargs = {})
#   %convert_element_type : Tensor "i64[8, 128, 2, 4][1024, 8, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.int64), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[8, 128, 2, 1][256, 2, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%getitem_7, -1), kwargs = {})
#   %mul_4 : Tensor "f32[8, 128, 2, 4][1024, 8, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %unsqueeze_3), kwargs = {})
#   %sum_3 : Tensor "f32[8, 128, 4][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [-2]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[8, 128, 4, 1][512, 4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%sum_3, -1), kwargs = {})
#   %mul_17 : Tensor "f32[8, 128, 4, 256][131072, 1024, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_30, %unsqueeze_4), kwargs = {})
#   return %mul_17
triton_poi_fused__to_copy__unsafe_view_arange_eq_mul_stack_sum_unsqueeze_10 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_arange_eq_mul_stack_sum_unsqueeze_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_arange_eq_mul_stack_sum_unsqueeze_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 12582912}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_arange_eq_mul_stack_sum_unsqueeze_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = (xindex % 1024)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 4)
    x2 = xindex // 1024
    x4 = xindex
    tmp25 = tl.load(in_ptr4 + (2*x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (2*x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (1 + 2*x2), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr5 + (1 + 2*x2), None, eviction_policy='evict_last')
    tmp0 = (x3).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x3).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (256*x2 + (x0 + 256*x1)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp4 >= tmp5
    tmp9 = tl.full([1], 512, tl.int64)
    tmp10 = tmp4 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + (256*x2 + ((-256) + x0 + 256*x1)), tmp11, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp4 >= tmp9
    tmp14 = tl.full([1], 768, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr2 + (256*x2 + ((-512) + x0 + 256*x1)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp0 >= tmp14
    tmp19 = tl.full([1], 1024, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr3 + (256*x2 + ((-768) + x0 + 256*x1)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp16, tmp17, tmp21)
    tmp23 = tl.where(tmp11, tmp12, tmp22)
    tmp24 = tl.where(tmp6, tmp7, tmp23)
    tmp26 = (x1).to(tl.int64)
    tmp27 = (tmp26).to(tl.int64)
    tmp28 = tmp25 == tmp27
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp34 = tmp33 == tmp27
    tmp35 = tmp34.to(tl.int64)
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 + tmp38
    tmp40 = tmp24 * tmp39
    tl.store(out_ptr0 + (x4), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/z5/cz52e2n3vmgiqkcpdajkpwxyay5brpjxhympk7xxovq4zhijayka.py
# Topologically Sorted Source Nodes: [hidden, linear_1, hidden_1, sum_2, hidden_2, layer_norm_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.sum, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden => embedding
#   hidden_1 => add_2
#   hidden_2 => add_9
#   layer_norm_2 => add_10, add_11, mul_18, mul_19, rsqrt_2, sub_5, var_mean_2
#   linear_1 => view_11
#   sum_2 => sum_4
# Graph fragment:
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "f32[2048, 256][256, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %mm_1 : Tensor "f32[1024, 256][256, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mul_17 : Tensor "f32[8, 128, 4, 256][131072, 1024, 256, 1]cuda:0" = PlaceHolder[target=mul_17]
#   %add_9 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0" = PlaceHolder[target=add_9]
#   %getitem_10 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf45 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf45]
#   %arg17_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %embedding : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg0_1, %arg1_1), kwargs = {})
#   %view_11 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [8, 128, 256]), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_11), kwargs = {})
#   %sum_4 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [-2]), kwargs = {})
#   %add_9 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %sum_4), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_10), kwargs = {})
#   %add_10 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_18 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_2), kwargs = {})
#   %mul_19 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %arg17_1), kwargs = {})
#   %add_11 : Tensor "f32[8, 128, 256][32768, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg18_1), kwargs = {})
#   return %add_9,%getitem_10,%buf45,%add_11
triton_per_fused__unsafe_view_add_embedding_native_layer_norm_sum_11 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_native_layer_norm_sum_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_native_layer_norm_sum_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 8192, 'r0_': 7342080}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_native_layer_norm_sum_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (r0_1 + 256*x0), xmask, eviction_policy='evict_first', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 256*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tl.device_assert(((0 <= tmp0) & (tmp0 < 2048)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 2048")
    tmp2 = tl.load(in_ptr1 + (r0_1 + 256*tmp0), xmask, eviction_policy='evict_first', other=0.0)
    tmp4 = tmp2 + tmp3
    tmp12 = tmp4 + tmp5
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = (tl.full([1, 1], 256, tl.int32)).to(tl.float32)
    tmp21 = (tmp19 / tmp20)
    tmp22 = tmp13 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp28 = tmp12 - tmp21
    tmp29 = tl.full([1, 1], 256.0, tl.float32)
    tmp30 = (tmp27 / tmp29)
    tmp31 = tl.full([1, 1], 1e-05, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tl.store(in_out_ptr0 + (r0_1 + 256*x0), tmp38, xmask)
''', device_str='cuda')


triton_poi_fused_weighted_expert_sum_12 = async_compile.triton('triton_poi_fused_weighted_expert_sum_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_weighted_expert_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3145728}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_weighted_expert_sum_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < 262144
    feat = xindex % 256
    tok = xindex // 256
    idx0 = tl.load(in_ptr4 + (2*tok), xmask, eviction_policy='evict_last', other=0)
    idx1 = tl.load(in_ptr4 + (1 + 2*tok), xmask, eviction_policy='evict_last', other=0)
    wt0 = tl.load(in_ptr5 + (2*tok), xmask, eviction_policy='evict_last', other=0.0)
    wt1 = tl.load(in_ptr5 + (1 + 2*tok), xmask, eviction_policy='evict_last', other=0.0)
    base = 256*tok + feat
    v00 = tl.load(in_ptr0 + base, xmask & (idx0 == 0), eviction_policy='evict_last', other=0.0)
    v01 = tl.load(in_ptr1 + base, xmask & (idx0 == 1), eviction_policy='evict_last', other=0.0)
    v02 = tl.load(in_ptr2 + base, xmask & (idx0 == 2), eviction_policy='evict_last', other=0.0)
    v03 = tl.load(in_ptr3 + base, xmask & (idx0 == 3), eviction_policy='evict_last', other=0.0)
    v10 = tl.load(in_ptr0 + base, xmask & (idx1 == 0), eviction_policy='evict_last', other=0.0)
    v11 = tl.load(in_ptr1 + base, xmask & (idx1 == 1), eviction_policy='evict_last', other=0.0)
    v12 = tl.load(in_ptr2 + base, xmask & (idx1 == 2), eviction_policy='evict_last', other=0.0)
    v13 = tl.load(in_ptr3 + base, xmask & (idx1 == 3), eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + xindex, wt0 * (v00 + v01 + v02 + v03) + wt1 * (v10 + v11 + v12 + v13), xmask)
''', device_str='cuda')


triton_poi_fused_softmax_top2_12 = async_compile.triton('triton_poi_fused_softmax_top2_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_softmax_top2_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 4, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 12288}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_softmax_top2_12(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < 1024
    a = tl.load(in_ptr0 + (4*xindex), xmask, other=-float('inf'))
    b = tl.load(in_ptr0 + (1 + 4*xindex), xmask, other=-float('inf'))
    c = tl.load(in_ptr0 + (2 + 4*xindex), xmask, other=-float('inf'))
    d = tl.load(in_ptr0 + (3 + 4*xindex), xmask, other=-float('inf'))
    m01 = triton_helpers.maximum(a, b)
    m23 = triton_helpers.maximum(c, d)
    m = triton_helpers.maximum(m01, m23)
    ea = libdevice.exp(a - m)
    eb = libdevice.exp(b - m)
    ec = libdevice.exp(c - m)
    ed = libdevice.exp(d - m)
    s = ea + eb + ec + ed
    pa = ea / s
    pb = eb / s
    pc = ec / s
    pd = ed / s

    idx2 = tl.full([XBLOCK], 2, tl.int64)
    idx3 = tl.full([XBLOCK], 3, tl.int64)
    top1_v = a
    top1_i = tl.full([XBLOCK], 0, tl.int64)
    top2_v = b
    top2_i = tl.full([XBLOCK], 1, tl.int64)
    swap01 = top2_v > top1_v
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_v = tl.where(swap01, top2_v, top1_v)
    top1_i = tl.where(swap01, top2_i, top1_i)
    top2_v = tl.where(swap01, old_top1_v, top2_v)
    top2_i = tl.where(swap01, old_top1_i, top2_i)

    take1c = c > top1_v
    take2c = (c > top2_v) & ~take1c
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_v = tl.where(take1c, c, top1_v)
    top1_i = tl.where(take1c, idx2, top1_i)
    top2_v = tl.where(take1c, old_top1_v, tl.where(take2c, c, top2_v))
    top2_i = tl.where(take1c, old_top1_i, tl.where(take2c, idx2, top2_i))

    take1d = d > top1_v
    take2d = (d > top2_v) & ~take1d
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_v = tl.where(take1d, d, top1_v)
    top1_i = tl.where(take1d, idx3, top1_i)
    top2_v = tl.where(take1d, old_top1_v, tl.where(take2d, d, top2_v))
    top2_i = tl.where(take1d, old_top1_i, tl.where(take2d, idx3, top2_i))

    p1 = tl.where(top1_i == 0, pa, tl.where(top1_i == 1, pb, tl.where(top1_i == 2, pc, pd)))
    p2 = tl.where(top2_i == 0, pa, tl.where(top2_i == 1, pb, tl.where(top2_i == 2, pc, pd)))
    tl.store(out_ptr0 + (2*xindex), p1, xmask)
    tl.store(out_ptr0 + (1 + 2*xindex), p2, xmask)
    tl.store(out_ptr1 + (2*xindex), top1_i, xmask)
    tl.store(out_ptr1 + (1 + 2*xindex), top2_i, xmask)
''', device_str='cuda')


triton_per_fused_router_mm_softmax_top2_13 = async_compile.triton('triton_per_fused_router_mm_softmax_top2_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=58, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_router_mm_softmax_top2_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 8192, 'r0_': 2099200}, 'backend_hash': '87A67B323EE5F6A0843017BE833358B05B0B7AA0445E3BAB9A504DA1E12BD5E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_router_mm_softmax_top2_13(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    rows = xoffset + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, R0_BLOCK)[None, :]
    xmask = rows < xnumel
    xv = tl.load(in_ptr0 + (256*rows + r), xmask, other=0.0)
    w0 = tl.load(in_ptr1 + r, None, eviction_policy='evict_last')
    w1 = tl.load(in_ptr1 + (256 + r), None, eviction_policy='evict_last')
    w2 = tl.load(in_ptr1 + (512 + r), None, eviction_policy='evict_last')
    w3 = tl.load(in_ptr1 + (768 + r), None, eviction_policy='evict_last')
    a = tl.sum(xv * w0, 1)[:, None]
    b = tl.sum(xv * w1, 1)[:, None]
    c = tl.sum(xv * w2, 1)[:, None]
    d = tl.sum(xv * w3, 1)[:, None]

    m01 = triton_helpers.maximum(a, b)
    m23 = triton_helpers.maximum(c, d)
    m = triton_helpers.maximum(m01, m23)
    ea = libdevice.exp(a - m)
    eb = libdevice.exp(b - m)
    ec = libdevice.exp(c - m)
    ed = libdevice.exp(d - m)
    s = ea + eb + ec + ed
    pa = ea / s
    pb = eb / s
    pc = ec / s
    pd = ed / s

    idx2 = tl.full([XBLOCK, 1], 2, tl.int64)
    idx3 = tl.full([XBLOCK, 1], 3, tl.int64)
    top1_v = a
    top1_i = tl.full([XBLOCK, 1], 0, tl.int64)
    top2_v = b
    top2_i = tl.full([XBLOCK, 1], 1, tl.int64)
    swap01 = top2_v > top1_v
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_v = tl.where(swap01, top2_v, top1_v)
    top1_i = tl.where(swap01, top2_i, top1_i)
    top2_v = tl.where(swap01, old_top1_v, top2_v)
    top2_i = tl.where(swap01, old_top1_i, top2_i)

    take1c = c > top1_v
    take2c = (c > top2_v) & ~take1c
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_v = tl.where(take1c, c, top1_v)
    top1_i = tl.where(take1c, idx2, top1_i)
    top2_v = tl.where(take1c, old_top1_v, tl.where(take2c, c, top2_v))
    top2_i = tl.where(take1c, old_top1_i, tl.where(take2c, idx2, top2_i))

    take1d = d > top1_v
    take2d = (d > top2_v) & ~take1d
    old_top1_v = top1_v
    old_top1_i = top1_i
    top1_i = tl.where(take1d, idx3, top1_i)
    top2_i = tl.where(take1d, old_top1_i, tl.where(take2d, idx3, top2_i))

    p1 = tl.where(top1_i == 0, pa, tl.where(top1_i == 1, pb, tl.where(top1_i == 2, pc, pd)))
    p2 = tl.where(top2_i == 0, pa, tl.where(top2_i == 1, pb, tl.where(top2_i == 2, pc, pd)))
    tl.store(out_ptr0 + (2*rows), p1, xmask)
    tl.store(out_ptr0 + (1 + 2*rows), p2, xmask)
    tl.store(out_ptr1 + (2*rows), top1_i, xmask)
    tl.store(out_ptr1 + (1 + 2*rows), top2_i, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
        args.clear()
        assert_size_stride(arg1_1, (8, 128), (128, 1), 'input')
        assert_size_stride(arg0_1, (2048, 256), (256, 1), 'input')
        assert_size_stride(arg2_1, (256, ), (1, ), 'input')
        assert_size_stride(arg3_1, (256, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf3 = empty_strided_cuda((8, 128, 256), (32768, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden, layer_norm], Original ATen: [aten.embedding, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_embedding_native_layer_norm_0.run(arg1_1, arg0_1, arg2_1, arg3_1, buf3, 1024, 256, stream=raw_stream0)
            del arg2_1
            del arg3_1
            assert_size_stride(arg4_1, (768, 256), (256, 1), 'input')
            buf4 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden, layer_norm, qkv], Original ATen: [aten.embedding, aten.native_layer_norm, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (1024, 256), (256, 1), 0), reinterpret_tensor(arg4_1, (256, 768), (1, 256), 0), out=buf4)
            del arg4_1
            buf5 = reinterpret_tensor(buf3, (8, 8, 128, 32), (32768, 4096, 32, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, query_1, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_unbind_view_1.run(buf4, buf5, 262144, stream=raw_stream0)
            buf6 = empty_strided_cuda((8, 8, 32, 128), (32768, 4096, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, key_1, transpose_3, matmul], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_unbind_view_2.run(buf4, buf6, 2048, 128, stream=raw_stream0)
            buf11 = empty_strided_cuda((8, 8, 128, 32), (32768, 4096, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, value_1, matmul_1], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_unbind_view_3.run(buf4, buf11, 262144, stream=raw_stream0)
            del buf4
            buf7 = empty_strided_cuda((64, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, query_1, matmul, key_1, transpose_3], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf5, (64, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf6, (64, 32, 128), (4096, 128, 1), 0), out=buf7)
            del buf5
            del buf6
            buf10 = reinterpret_tensor(buf7, (8, 8, 128, 128), (131072, 16384, 128, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [causal, ones, invert, masked_fill, matmul, scores, softmax], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.view, aten.div, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax_bitwise_not_div_masked_fill_ones_prepare_softmax_online_tril_view_4.run(buf10, 8192, 128, stream=raw_stream0)
            buf12 = empty_strided_cuda((64, 128, 32), (4096, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [qkv, qkv_1, unbind, causal, ones, invert, masked_fill, matmul, scores, softmax, matmul_1, value_1], Original ATen: [aten._unsafe_view, aten.view, aten.unbind, aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.div, aten._softmax, aten.transpose, aten.clone, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf10, (64, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf11, (64, 128, 32), (4096, 32, 1), 0), out=buf12)
            del buf10
            buf13 = reinterpret_tensor(buf11, (8, 128, 8, 32), (32768, 256, 32, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_4, attended], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_5.run(buf12, buf13, 262144, stream=raw_stream0)
            assert_size_stride(arg5_1, (256, 256), (256, 1), 'input')
            buf14 = reinterpret_tensor(buf12, (1024, 256), (256, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [matmul_1, transpose_4, attended, linear_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf13, (1024, 256), (256, 1), 0), reinterpret_tensor(arg5_1, (256, 256), (1, 256), 0), out=buf14)
            del arg5_1
            assert_size_stride(arg6_1, (256, ), (1, ), 'input')
            assert_size_stride(arg7_1, (256, ), (1, ), 'input')
            buf18 = reinterpret_tensor(buf13, (8, 128, 256), (32768, 256, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [hidden, linear_1, hidden_1, layer_norm_1], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_native_layer_norm_6.run(arg1_1, arg0_1, buf14, arg6_1, arg7_1, buf18, 1024, 256, stream=raw_stream0)
            del arg6_1
            del arg7_1
            assert_size_stride(arg8_1, (4, 256), (256, 1), 'input')
            buf22 = empty_strided_cuda((8, 128, 2), (256, 2, 1), torch.float32)
            buf23 = empty_strided_cuda((8, 128, 2), (256, 2, 1), torch.int64)
            # Fused router projection, four-way softmax, and sorted top-2 selection.
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_router_mm_softmax_top2_13.run(buf18, arg8_1, buf22, buf23, 1024, 256, stream=raw_stream0)
            del arg8_1
            assert_size_stride(arg9_1, (512, 256), (256, 1), 'input')
            buf30 = empty_strided_cuda((1024, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 256), (256, 1), 0), reinterpret_tensor(arg9_1, (256, 512), (1, 256), 0), out=buf30)
            del arg9_1
            buf31 = reinterpret_tensor(buf30, (8, 128, 512), (65536, 512, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [linear_3, gelu], Original ATen: [aten._unsafe_view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_gelu_9.run(buf31, 524288, stream=raw_stream0)
            assert_size_stride(arg10_1, (256, 512), (512, 1), 'input')
            buf32 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, gelu, linear_4], Original ATen: [aten._unsafe_view, aten.gelu, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf31, (1024, 512), (512, 1), 0), reinterpret_tensor(arg10_1, (512, 256), (1, 512), 0), out=buf32)
            del arg10_1
            assert_size_stride(arg11_1, (512, 256), (256, 1), 'input')
            buf33 = reinterpret_tensor(buf31, (1024, 512), (512, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 256), (256, 1), 0), reinterpret_tensor(arg11_1, (256, 512), (1, 256), 0), out=buf33)
            del arg11_1
            buf34 = reinterpret_tensor(buf33, (8, 128, 512), (65536, 512, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [linear_5, gelu_1], Original ATen: [aten._unsafe_view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_gelu_9.run(buf34, 524288, stream=raw_stream0)
            assert_size_stride(arg12_1, (256, 512), (512, 1), 'input')
            buf35 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, gelu_1, linear_6], Original ATen: [aten._unsafe_view, aten.gelu, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (1024, 512), (512, 1), 0), reinterpret_tensor(arg12_1, (512, 256), (1, 512), 0), out=buf35)
            del arg12_1
            assert_size_stride(arg13_1, (512, 256), (256, 1), 'input')
            buf36 = reinterpret_tensor(buf34, (1024, 512), (512, 1), 0); del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 256), (256, 1), 0), reinterpret_tensor(arg13_1, (256, 512), (1, 256), 0), out=buf36)
            del arg13_1
            buf37 = reinterpret_tensor(buf36, (8, 128, 512), (65536, 512, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [linear_7, gelu_2], Original ATen: [aten._unsafe_view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_gelu_9.run(buf37, 524288, stream=raw_stream0)
            assert_size_stride(arg14_1, (256, 512), (512, 1), 'input')
            buf38 = empty_strided_cuda((1024, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, gelu_2, linear_8], Original ATen: [aten._unsafe_view, aten.gelu, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (1024, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 256), (1, 512), 0), out=buf38)
            del arg14_1
            assert_size_stride(arg15_1, (512, 256), (256, 1), 'input')
            buf39 = reinterpret_tensor(buf37, (1024, 512), (512, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 256), (256, 1), 0), reinterpret_tensor(arg15_1, (256, 512), (1, 256), 0), out=buf39)
            del arg15_1
            buf40 = reinterpret_tensor(buf39, (8, 128, 512), (65536, 512, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_9, gelu_3], Original ATen: [aten._unsafe_view, aten.gelu]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_gelu_9.run(buf40, 524288, stream=raw_stream0)
            assert_size_stride(arg16_1, (256, 512), (512, 1), 'input')
            buf41 = reinterpret_tensor(buf18, (1024, 256), (256, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_9, gelu_3, linear_10], Original ATen: [aten._unsafe_view, aten.gelu, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (1024, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 256), (1, 512), 0), out=buf41)
            del arg16_1
            del buf40
            buf42 = empty_strided_cuda((8, 128, 256), (32768, 256, 1), torch.float32)
            # Weighted sum of the two selected expert outputs.
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_weighted_expert_sum_12.run(buf32, buf35, buf38, buf41, buf23, buf22, buf42, 262144, stream=raw_stream0)
            del buf22
            del buf23
            del buf32
            del buf35
            del buf38
            del buf41
            assert_size_stride(arg17_1, (256, ), (1, ), 'input')
            assert_size_stride(arg18_1, (256, ), (1, ), 'input')
            buf43 = reinterpret_tensor(buf14, (8, 128, 256), (32768, 256, 1), 0); del buf14  # reuse
            buf47 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [hidden, linear_1, hidden_1, sum_2, hidden_2, layer_norm_2], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.sum, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_native_layer_norm_sum_11.run(buf47, arg1_1, arg0_1, buf42, arg17_1, arg18_1, 1024, 256, stream=raw_stream0)
            del arg0_1
            del arg17_1
            del arg18_1
            del arg1_1
            del buf42
            assert_size_stride(arg19_1, (2048, 256), (256, 1), 'input')
            buf48 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_2, linear_11], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf47, (1024, 256), (256, 1), 0), reinterpret_tensor(arg19_1, (256, 2048), (1, 256), 0), out=buf48)
            del arg19_1
            del buf47
        return (reinterpret_tensor(buf48, (8, 128, 2048), (262144, 2048, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
