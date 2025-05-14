import types

import torch

from .utils import get_linear_cls


def tree_convert_bf16(module: torch.nn.Module, linear_cls):
    children = list(module.children())
    if len(children) == 0 and not (
        isinstance(module, torch.nn.Linear) or isinstance(module, linear_cls)
    ):
        module = module.to(torch.bfloat16)
    for child in children:
        tree_convert_bf16(child, linear_cls)


def patch_comfyui_native_transformer(
    transformer, use_fp8_attention=False, transform_weights=True
):

    from .comfyui_native import create_forwards, cross_attn_forward

    linear_cls = get_linear_cls()
    fused_forward, gelu_forward, ff_forward = create_forwards()
    attn_forward = cross_attn_forward(use_fp8_attention)

    for block in transformer.transformer_blocks:
        tree_convert_bf16(block, linear_cls)
    is_patched = getattr(transformer, "is_patched", False)
    if not is_patched:
        for name, module in transformer.named_modules():
            if "transformer_block" in name and isinstance(module, torch.nn.Linear):
                if isinstance(module, linear_cls):
                    continue

                *parent_path, child_name = name.split(".")
                parent = transformer
                for part in parent_path:
                    parent = getattr(parent, part)

                setattr(
                    parent,
                    child_name,
                    linear_cls.from_linear(module, transform_weights),
                )
                del module.weight
                del module.bias
                torch.cuda.empty_cache()

    for block in transformer.transformer_blocks:
        block.forward = types.MethodType(fused_forward, block)

    for block in transformer.transformer_blocks:
        block.attn1.forward = types.MethodType(attn_forward, block.attn1)
        block.attn2.forward = types.MethodType(attn_forward, block.attn2)

    for block in transformer.transformer_blocks:
        block.ff.net[0].forward = types.MethodType(gelu_forward, block.ff.net[0])

    for block in transformer.transformer_blocks:
        block.ff.forward = types.MethodType(ff_forward, block.ff)

    setattr(transformer, "is_patched", True)


def patch_comfyui_transformer(transformer, use_fp8_attention=False):
    from .comfyui import attn_forward, create_attn_processor, create_forwards

    linear_cls = get_linear_cls()
    attn_processor_cls = create_attn_processor()
    fused_forward, gelu_forward, ff_forward = create_forwards()

    for name, module in transformer.named_modules():
        if "transformer_block" in name and isinstance(module, torch.nn.Linear):
            *parent_path, child_name = name.split(".")

            parent = transformer
            for part in parent_path:
                parent = getattr(parent, part)

            setattr(parent, child_name, linear_cls.from_linear(module, True))

    attn_processor = attn_processor_cls()
    for name, module in transformer.named_modules():
        if hasattr(module, "set_processor"):
            module.set_processor(attn_processor)

    for block in transformer.transformer_blocks:
        block.attn1.forward = types.MethodType(attn_forward, block.attn1)
        block.attn2.forward = types.MethodType(attn_forward, block.attn2)

    for block in transformer.transformer_blocks:
        block.forward = types.MethodType(fused_forward, block)

    for block in transformer.transformer_blocks:
        block.ff.net[0].forward = types.MethodType(gelu_forward, block.ff.net[0])

    for block in transformer.transformer_blocks:
        block.ff.forward = types.MethodType(ff_forward, block.ff)


def patch_diffusers_transformer(transformer):
    from .diffusers import attn_forward, create_attn_processor, create_forwards

    linear_cls = get_linear_cls()
    attn_processor_cls = create_attn_processor()
    fused_forward, gelu_forward, ff_forward = create_forwards()

    for name, module in transformer.named_modules():
        if "transformer_block" in name and isinstance(module, torch.nn.Linear):
            *parent_path, child_name = name.split(".")

            parent = transformer
            for part in parent_path:
                parent = getattr(parent, part)

            setattr(parent, child_name, linear_cls.from_linear(module, True))
        else:
            module = module.to(torch.bfloat16)

    attn_processor = attn_processor_cls()
    for name, module in transformer.named_modules():
        if hasattr(module, "set_processor"):
            module.set_processor(attn_processor)

    for block in transformer.transformer_blocks:
        block.attn1.forward = types.MethodType(attn_forward, block.attn1)
        block.attn2.forward = types.MethodType(attn_forward, block.attn2)

    for block in transformer.transformer_blocks:
        block.forward = types.MethodType(fused_forward, block)

    for block in transformer.transformer_blocks:
        block.ff.net[0].forward = types.MethodType(gelu_forward, block.ff.net[0])

    for block in transformer.transformer_blocks:
        block.ff.forward = types.MethodType(ff_forward, block.ff)
