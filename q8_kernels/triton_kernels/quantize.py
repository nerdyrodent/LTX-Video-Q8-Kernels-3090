from typing import *

import torch
from triton import jit
from triton import language as tl
from triton import next_power_of_2


@jit
def _kernel(X, OUT, SCALES, HDIM, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * HDIM
    out_ptr = OUT + row_idx * HDIM

    h_offset = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + h_offset, mask=h_offset < HDIM).to(tl.float32)
    x_scale = 127.0 / tl.max(tl.abs(x))
    x_scaled = x * x_scale
    x_scaled += (0.5 * tl.where(x_scaled >= 0, 1, -1)).to(tl.int8)

    tl.store(out_ptr + h_offset, x_scaled, mask=h_offset < HDIM)
    tl.store(SCALES + row_idx, 1 / x_scale)


# @torch.compile(fullgraph=True)
# @torch.compiler.disable()
def run_quantize_kernel(x: torch.Tensor, out_dtype: Optional[torch.dtype] = None):
    x_shape_orig = x.shape
    x = x.view(-1, x_shape_orig[-1])
    out = torch.empty(x_shape_orig, dtype=torch.int8, device=x.device)
    scales = torch.empty(x.shape[0], dtype=torch.float, device=x.device)

    BLOCK_SIZE = next_power_of_2(x_shape_orig[-1])
    grid = (x.shape[0],)
    _kernel[grid](x, out, scales, x_shape_orig[-1], BLOCK_SIZE, num_warps=4)

    return out.view(x_shape_orig), scales.view(x_shape_orig[:-1])


quantize_triton = run_quantize_kernel
