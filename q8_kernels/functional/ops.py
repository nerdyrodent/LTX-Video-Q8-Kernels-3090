import math
from typing import Optional, Tuple

import torch
from q8_kernels_cuda.ops._C import (
    dequant_fast_hadamard_transform_12N_cvt,
    dequant_fast_hadamard_transform_cvt,
    fast_hadamard_transform,
    gelu_fast_hadamard_transform_cvt,
    norm_fma_hadamard_cvt,
    quant_fast_hadamard_transform_12N_cvt,
    quant_fast_hadamard_transform_cvt,
)
from q8_kernels_cuda.ops._C import rms_norm_rope as rms_norm_rope_cuda


###CURRENTLY ONLY BF16, FP8 and INT8 supported
@torch.library.custom_op("q8_kernels_ops::rms_norm_rope", mutates_args=())
def _rms_norm_rope_cuda(
    x: torch.Tensor,
    weights: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    out_16bit: bool,
) -> torch.Tensor:
    return rms_norm_rope_cuda(x, weights, cos_freqs, sin_freqs, out_16bit)


@torch.library.register_fake("q8_kernels_ops::rms_norm_rope")
def _rms_norm_rope_cuda_fake(
    x: torch.Tensor,
    weights: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    out_16bit: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    if out_16bit:
        return out.to(torch.bfloat16)
    else:
        return out.to(torch.float8_e4m3fn)


@torch.library.custom_op("q8_kernels_ops::fast_hadamard_transform", mutates_args=())
def _fast_hadamard_transform(
    x: torch.Tensor, scale: float, out_type: torch.dtype
) -> torch.Tensor:
    return fast_hadamard_transform(x, scale, out_type)


@torch.library.custom_op(
    "q8_kernels_ops::gelu_fast_hadamard_transform_cvt", mutates_args=()
)
def _gelu_fast_hadamard_transform_cvt(
    x: torch.Tensor, scale: float, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return gelu_fast_hadamard_transform_cvt(x, scale, out_dtype)


@torch.library.custom_op("q8_kernels_ops::norm_fma_hadamard_cvt", mutates_args=())
def _norm_fma_hadamard_cvt(
    x: torch.Tensor,
    weights: Optional[torch.Tensor],
    y: torch.Tensor,
    z: torch.Tensor,
    scale: float,
    add_one: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return norm_fma_hadamard_cvt(x, weights, y, z, scale, add_one, out_dtype)


@torch.library.custom_op(
    "q8_kernels_ops::quant_fast_hadamard_transform_cvt", mutates_args=()
)
def _quant_fast_hadamard_transform_cvt(
    x: torch.Tensor, scale: float, out_type: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return quant_fast_hadamard_transform_cvt(x, scale, out_type)


@torch.library.custom_op(
    "q8_kernels_ops::dequant_fast_hadamard_transform", mutates_args=()
)
def _dequant_fast_hadamard_transform(
    x: torch.Tensor, row_scales: torch.Tensor, scale: float
) -> torch.Tensor:
    return dequant_fast_hadamard_transform_cvt(x, row_scales, scale)


@torch.library.custom_op(
    "q8_kernels_ops::quant_fast_hadamard_transform_12N_cvt", mutates_args=()
)
def _quant_fast_hadamard_transform_12N_cvt(
    x: torch.Tensor, scale: float, out_type: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    return quant_fast_hadamard_transform_12N_cvt(x, scale, out_type)


@torch.library.custom_op(
    "q8_kernels_ops::dequant_fast_hadamard_transform_12N", mutates_args=()
)
def _dequant_fast_hadamard_transform_12N(
    x: torch.Tensor, row_scales: torch.Tensor, scale: float
) -> torch.Tensor:
    return dequant_fast_hadamard_transform_12N_cvt(x, row_scales, scale)


###FAKE KERNELS
@torch.library.register_fake("q8_kernels_ops::dequant_fast_hadamard_transform")
def _dequant_fast_hadamard_transform_fake(
    x: torch.Tensor, row_scales: torch.Tensor, scale: float
) -> torch.Tensor:
    out = torch.empty_like(x)
    return out.to(torch.bfloat16)


@torch.library.register_fake("q8_kernels_ops::dequant_fast_hadamard_transform_12N")
def _dequant_fast_hadamard_transform_fake(
    x: torch.Tensor, row_scales: torch.Tensor, scale: float
) -> torch.Tensor:
    out = torch.empty_like(x)
    return out.to(torch.bfloat16)


@torch.library.register_fake("q8_kernels_ops::fast_hadamard_transform")
def _fast_hadamard_transform_fake(
    x: torch.Tensor, scale: float, out_type: torch.dtype
) -> torch.Tensor:
    out = torch.empty_like(x)
    if out_type == torch.bfloat16:
        return out.to(torch.bfloat16)
    else:
        return out.to(torch.float8_e4m3fn)


@torch.library.register_fake("q8_kernels_ops::norm_fma_hadamard_cvt")
def _norm_fma_hadamard_cvt_fake(
    x: torch.Tensor,
    weights: Optional[torch.Tensor],
    y: torch.Tensor,
    z: torch.Tensor,
    scale: float,
    add_one: bool,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_scales = torch.empty(*x.shape[:-1], device=out.device, dtype=torch.float32)
    return out.to(out_dtype), out_scales


@torch.library.register_fake("q8_kernels_ops::quant_fast_hadamard_transform_cvt")
def _quant_fast_hadamard_transform_cvt_fake(
    x: torch.Tensor, scale: float, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_scales = torch.empty(*x.shape[:-1], device=out.device, dtype=torch.float32)
    return out.to(out_dtype), out_scales


@torch.library.register_fake("q8_kernels_ops::quant_fast_hadamard_transform_12N_cvt")
def _quant_fast_hadamard_transform_cvt_fake(
    x: torch.Tensor, scale: float, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_scales = torch.empty(*x.shape[:-1], device=out.device, dtype=torch.float32)
    return out.to(out_dtype), out_scales


@torch.library.register_fake("q8_kernels_ops::gelu_fast_hadamard_transform_cvt")
def _gelu_fast_hadamard_transform_cvt_fake(
    x: torch.Tensor, scale: float, out_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    out_scales = torch.empty(*x.shape[:-1], device=out.device, dtype=torch.float32)
    return out.to(out_dtype), out_scales


class NormFMAHadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weights,
        y_scale,
        z_shift,
        scale=1.0,
        add_one=False,
        out_dtype=torch.int8,
    ):
        return torch.ops.q8_kernels_ops.norm_fma_hadamard_cvt(
            x, weights, y_scale, z_shift, scale, add_one, out_dtype
        )


class GeluHadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0, out_dtype=torch.int8):
        return torch.ops.q8_kernels_ops.gelu_fast_hadamard_transform_cvt(
            x, scale, out_dtype
        )


class RMSNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, cos_freqs, sin_freqs, out_16bit):
        return torch.ops.q8_kernels_ops.rms_norm_rope(
            x, weights, cos_freqs, sin_freqs, out_16bit
        )


class QuantFastHadamardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, out_dtype):
        if x.shape[-1] % 12 == 0:
            return torch.ops.q8_kernels_ops.quant_fast_hadamard_transform_12N_cvt(
                x, scale, out_dtype
            )
        else:
            return torch.ops.q8_kernels_ops.quant_fast_hadamard_transform_cvt(
                x, scale, out_dtype
            )


class DequantFastHadamardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, row_scales, scale):
        if x.shape[-1] % 12 == 0:
            return torch.ops.q8_kernels_ops.dequant_fast_hadamard_transform_12N(
                x, row_scales, scale
            )
        else:
            return torch.ops.q8_kernels_ops.dequant_fast_hadamard_transform(
                x, row_scales, scale
            )


class HadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0, out_type=None):
        ctx._hadamard_transform_scale = scale
        return torch.ops.q8_kernels_ops.fast_hadamard_transform(x, scale, out_type)

    @staticmethod
    def backward(ctx, dout):
        return fast_hadamard_transform(dout, ctx._hadamard_transform_scale), None


def rms_norm_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    out_16bit=True,
) -> torch.Tensor:
    return RMSNormRope.apply(x, weights, cos_freqs, sin_freqs, out_16bit)


def norm_scale_shift_hadamard_transform(
    x: torch.Tensor,
    rms_weights: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hd_scale = 1 / math.sqrt(x.shape[-1])
    out, out_scales = NormFMAHadamardTransformFn.apply(
        x, rms_weights, scale_msa, shift_msa, hd_scale, True, out_dtype
    )
    if out_dtype == torch.int8:
        return out, out_scales
    else:
        return out, None


def gelu_hadamard_transform(
    x: torch.Tensor, scale: Optional[float] = None, out_dtype: torch.dtype = torch.int8
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = 1 / math.sqrt(x.shape[-1])
    out, out_scales = GeluHadamardTransformFn.apply(x, scale, out_dtype)
    if out_dtype == torch.int8:
        return out, out_scales
    else:
        return out, None


def quantize_hadamard(x, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = 1 / math.sqrt(x.shape[-1])
    out, out_scales = QuantFastHadamardFn.apply(x, scale, out_dtype)
    if out_dtype == torch.int8:
        return out, out_scales
    else:
        return out, None


def hadamard_transform(
    x: torch.Tensor,
    scale: Optional[float] = None,
    out_type: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1 / math.sqrt(x.shape[-1])
    return HadamardTransformFn.apply(x, scale, out_type)


def dequant_hadamard_transform(
    x: torch.Tensor, row_scales, scale: Optional[float] = None
) -> torch.Tensor:
    out_type = torch.bfloat16
    if scale is None:
        scale = 1 / math.sqrt(x.shape[-1])
    if x.dtype == torch.int8:
        return DequantFastHadamardFn.apply(x, row_scales, scale)
    else:
        return hadamard_transform(x, scale, out_type)
