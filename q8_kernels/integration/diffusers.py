import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention import _chunked_feed_forward
from diffusers.utils import deprecate
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from ..functional.ops import (
    dequant_hadamard_transform,
    gelu_hadamard_transform,
    norm_scale_shift_hadamard_transform,
    rms_norm_rope,
)
from .utils import get_attention_func, get_compute_dtype


def attn_forward(
    self,
    hidden_states: torch.FloatTensor,
    hidden_states_scales: Optional[torch.FloatTensor],
    freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    skip_layer_mask: Optional[torch.Tensor] = None,
    skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        skip_layer_mask (`torch.Tensor`, *optional*):
            The skip layer mask to use. If `None`, no mask is applied.
        skip_layer_strategy (`SkipLayerStrategy`, *optional*, defaults to `None`):
            Controls which layers to skip for spatiotemporal guidance.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty

    attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters
    ]
    if len(unused_kwargs) > 0:
        logger.warning(
            f"cross_attention_kwargs {unused_kwargs} are not expected by"
            f" {self.processor.__class__.__name__} and will be ignored."
        )
    cross_attention_kwargs = {
        k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
    }

    return self.processor(
        self,
        hidden_states,
        hidden_states_scales,
        freqs_cis=freqs_cis,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        skip_layer_mask=skip_layer_mask,
        skip_layer_strategy=skip_layer_strategy,
        **cross_attention_kwargs,
    )


def create_attn_processor():
    self_attn_props, cross_attn_props, out_tuple = get_attention_func()
    compute_dtype = get_compute_dtype()

    class AttnProcessor3_0:
        r"""
        Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
        """

        def __init__(self):
            pass

        def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            hidden_states_scales: Optional[torch.FloatTensor],
            freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            skip_layer_mask: Optional[torch.FloatTensor] = None,
            skip_layer_strategy: Optional[SkipLayerStrategy] = None,
            *args,
            **kwargs,
        ) -> torch.FloatTensor:
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)

            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if skip_layer_mask is not None:
                skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1)

            if (attention_mask is not None) and (not attn.use_tpu_flash_attention):
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, attn.heads, -1, attention_mask.shape[-1]
                )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            if encoder_hidden_states is not None:
                is_self_attention = False

                query = attn.to_q(hidden_states, None, True)
                query = attn.q_norm(query)

                key = attn.to_k(encoder_hidden_states, None, True)
                key = attn.k_norm(key)
                value = attn.to_v(encoder_hidden_states, None, True)
            else:  # if no context provided do self-attention
                is_self_attention = True

                query = attn.to_q(
                    hidden_states, hidden_states_scales, False, torch.bfloat16
                )
                query = rms_norm_rope(
                    query, freqs_cis[0], freqs_cis[1], attn.q_norm.weight
                )

                key = attn.to_k(
                    hidden_states, hidden_states_scales, False, torch.bfloat16
                )
                key = rms_norm_rope(key, freqs_cis[0], freqs_cis[1], attn.k_norm.weight)

                value = attn.to_v(
                    hidden_states, hidden_states_scales, False, torch.bfloat16
                )

            value_for_stg = value

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            if is_self_attention:
                query = self_attn_props[1](query)
                key = self_attn_props[1](key)
                value = self_attn_props[1](value)
            else:
                query = cross_attn_props[1](query)
                key = cross_attn_props[1](key)
                value = cross_attn_props[1](value)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)

            if (
                attn.use_tpu_flash_attention
            ):  # use tpu attention offload 'flash attention'
                q_segment_indexes = None
                if (
                    attention_mask is not None
                ):  # if mask is required need to tune both segmenIds fields
                    # attention_mask = torch.squeeze(attention_mask).to(torch.float32)
                    attention_mask = attention_mask.to(torch.float32)
                    q_segment_indexes = torch.ones(
                        batch_size,
                        query.shape[2],
                        device=query.device,
                        dtype=torch.float32,
                    )
                    assert (
                        attention_mask.shape[1] == key.shape[2]
                    ), f"ERROR: KEY SHAPE must be same as attention mask [{key.shape[2]}, {attention_mask.shape[1]}]"

                assert (
                    query.shape[2] % 128 == 0
                ), f"ERROR: QUERY SHAPE must be divisible by 128 (TPU limitation) [{query.shape[2]}]"
                assert (
                    key.shape[2] % 128 == 0
                ), f"ERROR: KEY SHAPE must be divisible by 128 (TPU limitation) [{key.shape[2]}]"

                # run the TPU kernel implemented in jax with pallas
                hidden_states_a = flash_attention(
                    q=query,
                    k=key,
                    v=value,
                    q_segment_ids=q_segment_indexes,
                    kv_segment_ids=attention_mask,
                    sm_scale=attn.scale,
                )
            else:
                if is_self_attention:
                    if out_tuple:
                        hidden_states_a, _ = self_attn_props[0](query, key, value)
                    else:
                        hidden_states_a = self_attn_props[0](query, key, value)
                else:
                    hidden_states_a = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
            if is_self_attention:
                hidden_states_a = self_attn_props[1](hidden_states_a).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
            else:
                hidden_states_a = cross_attn_props[1](hidden_states_a).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
            hidden_states_a = hidden_states_a.to(query.dtype)

            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.AttentionSkip
            ):
                if is_self_attention:
                    hidden_states = (
                        hidden_states_a * skip_layer_mask
                        + dequant_hadamard_transform(
                            hidden_states, hidden_states_scales
                        )
                        * (1.0 - skip_layer_mask)
                    )
                else:
                    hidden_states = (
                        hidden_states_a * skip_layer_mask
                        + hidden_states * (1.0 - skip_layer_mask)
                    )
                if compute_dtype == torch.float8_e4m3fn:
                    hidden_states = hidden_states.to(torch.bfloat16)
            elif (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.AttentionValues
            ):
                hidden_states = hidden_states_a * skip_layer_mask + value_for_stg * (
                    1.0 - skip_layer_mask
                )
            else:
                hidden_states = hidden_states_a

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, None, True)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
                if (
                    skip_layer_mask is not None
                    and skip_layer_strategy == SkipLayerStrategy.Residual
                ):
                    skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1, 1)

            if attn.residual_connection:
                if (
                    skip_layer_mask is not None
                    and skip_layer_strategy == SkipLayerStrategy.Residual
                ):
                    hidden_states = hidden_states + residual * skip_layer_mask
                else:
                    hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    return AttnProcessor3_0


def create_forwards():
    compute_dtype = get_compute_dtype()

    def fused_forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        sharding_mesh=None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored."
                )
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # norm_hidden_states = self.norm1(hidden_states)

        # Apply ada_norm_single
        if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
            assert timestep.ndim == 3  # [batch, 1 or num_tokens, embedding_dim]
            num_ada_params = self.scale_shift_table.shape[0]
            ada_values = self.scale_shift_table[None, None] + timestep.reshape(
                batch_size, timestep.shape[1], num_ada_params, -1
            )
            if self.adaptive_norm == "single_scale_shift":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    ada_values.unbind(dim=2)
                )
                norm_hidden_states, norm_hidden_states_scales = (
                    norm_scale_shift_hadamard_transform(
                        hidden_states,
                        self.norm1.weight,
                        scale_msa,
                        shift_msa,
                        compute_dtype,
                    )
                )
                # norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                scale_msa, gate_msa, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        elif self.adaptive_norm == "none":
            scale_msa, gate_msa, scale_mlp, gate_mlp = None, None, None, None
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        norm_hidden_states = norm_hidden_states.squeeze(
            1
        )  # TODO: Check if this is needed

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states_scales,
            freqs_cis=freqs_cis,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )
        if gate_msa is not None:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.adaptive_norm == "none":
                attn_input = self.attn2_norm(hidden_states)
            else:
                attn_input = hidden_states
            attn_output = self.attn2(
                attn_input,
                None,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # norm_hidden_states = self.norm2(hidden_states)
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states, norm_hidden_states_scales = (
                norm_scale_shift_hadamard_transform(
                    hidden_states,
                    self.norm2.weight,
                    scale_mlp,
                    shift_mlp,
                    compute_dtype,
                )
            )
            # norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)
        elif self.adaptive_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states, norm_hidden_states_scales)
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    def gelu_forward(self, hidden_states, hidden_states_scales):
        hidden_states = self.proj(
            hidden_states, hidden_states_scales, False, torch.bfloat16
        )
        hidden_states, hidden_states_scales = gelu_hadamard_transform(
            hidden_states, out_dtype=compute_dtype
        )
        return hidden_states, hidden_states_scales

    def ff_forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scales: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states, hidden_states_scales = self.net[0](
            hidden_states, hidden_states_scales
        )
        hidden_states = self.net[2](
            hidden_states, hidden_states_scales, False, out_dtype=torch.bfloat16
        )
        return hidden_states

    return fused_forward, gelu_forward, ff_forward
