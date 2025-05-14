from typing import *

import torch
import torch.nn as nn

from ..functional.linear import fp8_linear_func, q8_linear
from ..functional.ops import quantize_hadamard

try:
    from q8_kernels_cuda.gemm._C import fp8_gemm

    IS_FP8_FAST_ACC_AVAILABLE = True
except ImportError:
    IS_FP8_FAST_ACC_AVAILABLE = False


def is_16bit(x) -> bool:
    return x.dtype == torch.float16 or x.dtype == torch.bfloat16


class Q8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_hadamard=True,
        device=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_hadamard = use_hadamard

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.int8),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "scales", torch.empty(out_features, device=device, dtype=torch.float)
        )

    def forward(self, x, x_scales=None, out_dtype=None):
        return q8_linear(
            x,
            self.weight.data,
            self.bias.data if self.bias is not None else None,
            x_scales,
            self.scales,
            False,
            True,
            out_dtype,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_hadamard=True):
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            use_hadamard,
            linear.weight.device,
        )
        if linear.weight.data.dtype == torch.int8:
            w_quant, w_scale = linear.weight.data, linear.scales.data
        else:
            w_quant, w_scale = quantize_hadamard(linear.weight.data.cuda(), torch.int8)
        layer.weight.data = w_quant
        layer.scales.data = w_scale
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.float()
        return layer


class FP8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        use_hadamard: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features, device=device, dtype=torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        self.use_hadamard = use_hadamard
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x, x_scales=None, out_dtype=None):
        if out_dtype is None:
            out_dtype = x.dtype
        return fp8_linear_func(
            x,
            self.weight.data,
            self.bias.data if self.bias is not None else None,
            out_dtype,
            self.use_hadamard,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_hadamard=False):
        layer = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            use_hadamard=use_hadamard,
            device=linear.weight.device,
        )
        if use_hadamard:
            w_fp8, _ = quantize_hadamard(
                linear.weight.data.cuda().to(torch.bfloat16), torch.torch.float8_e4m3fn
            )
        else:
            w_fp8 = linear.weight.data.to(torch.float8_e4m3fn)
        layer.weight.data = w_fp8
        if linear.bias is not None:
            if IS_FP8_FAST_ACC_AVAILABLE:
                layer.bias.data = linear.bias.data.to(torch.torch.float32)
            else:
                layer.bias.data = linear.bias.data.to(torch.bfloat16)
        return layer
