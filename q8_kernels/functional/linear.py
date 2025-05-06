from typing import Optional

import torch

try:
    from q8_kernels.triton_kernels import q8_mm_bias_triton, q8_mm_triton

    IS_TRITON_AVAILABLE = True
except ImportError:
    IS_TRITON_AVAILABLE = False
from .ops import dequant_hadamard_transform, quantize_hadamard

try:
    from q8_kernels_cuda.gemm._C import fp8_gemm

    IS_FP8_FAST_ACC_AVAILABLE = True
except ImportError:
    IS_FP8_FAST_ACC_AVAILABLE = False

if IS_FP8_FAST_ACC_AVAILABLE:

    @torch.library.custom_op("fp8_gemm::fp8_mm", mutates_args=())
    def fp8_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        return fp8_gemm(a, b, bias, use_fp16)

    @fp8_mm.register_fake
    def fp8_mm_fake(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        o = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=torch.bfloat16)
        return o


def is_16bit(x) -> bool:
    if hasattr(x, "dtype"):
        return x.dtype == torch.float16 or x.dtype == torch.bfloat16
    else:
        return x == torch.float16 or x == torch.bfloat16


class Q8LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor],
        scale_a: Optional[torch.Tensor],
        scale_b: Optional[torch.Tensor],
        fuse_gelu: bool,
        use_hadamard: bool,
        out_dtype: Optional[torch.dtype],
        b_transposed: Optional[torch.Tensor],
        b_transposed_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert (
            (a.dtype == torch.float8_e4m3fn or is_16bit(a)) and scale_a is None
        ) or (
            a.dtype == torch.int8 and scale_a is not None
        ), "Q8LinearFunc: a dtype missmatch"
        assert (
            (b.dtype == torch.float8_e4m3fn or is_16bit(b)) and scale_b is None
        ) or (
            b.dtype == torch.int8 and scale_b is not None
        ), "Q8LinearFunc: b dtype missmatch"
        assert a.shape[-1] == b.shape[-1], "Q8LinearFunc: mnk missmatch"
        assert (
            bias is None or bias.dtype == torch.float
        ), "Q8LinearFunc: bias must be in fp32"

        if is_16bit(a):
            a, scale_a = quantize_hadamard(a, torch.int8)
        assert (
            IS_TRITON_AVAILABLE
        ), "Triton is not available. Please install Triton to use Q8LinearFunc."
        ctx.fuse_gelu = fuse_gelu
        ctx.out_dtype = out_dtype

        mm_func = q8_mm_bias_triton if bias is not None else q8_mm_triton
        mm_args = (
            (a, b, bias, scale_a, scale_b, fuse_gelu, out_dtype)
            if bias is not None
            else (a, b, scale_a, scale_b, fuse_gelu, out_dtype)
        )

        if b_transposed is not None:
            ctx.save_for_backward(b_transposed, b_transposed_scales)
            ctx.is_b_transposed = True
        else:
            ctx.save_for_backward(b, scale_b)
            ctx.is_b_transposed = False
        return mm_func(*mm_args)


class FP8LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
        use_hadamard: bool,
    ) -> torch.Tensor:
        a_orig_shape = a.shape
        new_shape = a_orig_shape[:-1] + (b.shape[0],)
        is_16bit_a = is_16bit(a)
        if is_16bit_a:
            a, _ = quantize_hadamard(a, torch.float8_e4m3fn)
        else:
            a = a.to(torch.float8_e4m3fn)

        a = a.view(-1, a_orig_shape[-1])
        if IS_FP8_FAST_ACC_AVAILABLE:
            o = fp8_mm(a, b, bias, True)
        else:
            b = b.t()
            scale = torch.tensor([1.0], device=a.device)
            o = torch._scaled_mm(
                a, b, scale, scale, bias, out_dtype=out_dtype, use_fast_accum=True
            )

        return o.view(new_shape)


def fp8_linear_func(a, b, bias=None, out_dtype=torch.bfloat16, use_hadamard=False):
    if out_dtype is None:
        out_dtype = a.dtype
    return FP8LinearFunc.apply(a, b, bias, out_dtype, use_hadamard)


def q8_linear(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    fuse_gelu: bool = False,
    use_hadamard=True,
    out_dtype: Optional[torch.dtype] = None,
    b_transposed: Optional[torch.Tensor] = None,
    b_transposed_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out_dtype is None:
        if is_16bit(a):
            out_dtype = a.dtype
        else:
            out_dtype = torch.float8_e4m3fn
    return Q8LinearFunc.apply(
        a,
        b,
        bias,
        scale_a,
        scale_b,
        fuse_gelu,
        use_hadamard,
        out_dtype,
        b_transposed,
        b_transposed_scales,
    )
