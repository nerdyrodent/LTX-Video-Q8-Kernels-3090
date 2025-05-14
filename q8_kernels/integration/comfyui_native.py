import torch
from einops import rearrange

from ..functional.ops import (gelu_hadamard_transform,
                              norm_scale_shift_hadamard_transform,
                              rms_norm_rope)
from .utils import get_attention_func, get_compute_dtype


def _feta_score(query_image, key_image, head_dim, num_frames, enhance_weight):
    scale = head_dim**-0.5
    query_image = query_image * scale
    attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
    attn_temp = attn_temp.to(torch.float32)
    attn_temp = attn_temp.softmax(dim=-1)

    # Reshape to [batch_size * num_tokens, num_frames, num_frames]
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

    # Create a mask for diagonal elements
    diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
    diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

    # Zero out diagonal elements
    attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

    # Calculate mean for each token's attention matrix
    # Number of off-diagonal elements per matrix is n*n - n
    num_off_diag = num_frames * num_frames - num_frames
    mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

    enhance_scores = mean_scores.mean() * (num_frames + enhance_weight)
    enhance_scores = enhance_scores.clamp(min=1)
    return enhance_scores


def get_feta_scores(img_q, img_k, num_heads, transformer_options):
    num_frames = transformer_options["original_shape"][2]
    _, ST, dim = img_q.shape
    head_dim = dim // num_heads
    spatial_dim = ST // num_frames

    query_image = rearrange(
        img_q,
        "B (T S) (N C) -> (B S) N T C",
        T=num_frames,
        S=spatial_dim,
        N=num_heads,
        C=head_dim,
    )
    key_image = rearrange(
        img_k,
        "B (T S) (N C) -> (B S) N T C",
        T=num_frames,
        S=spatial_dim,
        N=num_heads,
        C=head_dim,
    )
    weight = transformer_options.get("feta_weight", 0)
    return _feta_score(query_image, key_image, head_dim, num_frames, weight)


def cross_attn_forward(use_fp8_attention):
    import comfy.ldm.modules.attention

    self_attn, cross_attn, is_out_tuple = get_attention_func(use_fp8_attention)
    self_attn_function, self_attn_memory_layout = self_attn

    def self_attn_fn(q, k, v, heads, *args, **kwargs):
        b, _, dim = q.shape
        q = q.view(b, -1, heads, dim // heads)
        k = k.view(b, -1, heads, dim // heads)
        v = v.view(b, -1, heads, dim // heads)

        q = self_attn_memory_layout(q)
        k = self_attn_memory_layout(k)
        v = self_attn_memory_layout(v)

        out = self_attn_function(q, k, v)
        if is_out_tuple:
            out = out[0]

        out = self_attn_memory_layout(out).view(b, -1, dim)
        return out

    comfy.ldm.modules.attention.optimized_attention = self_attn_fn

    def forward(self, x, x_scales=None, context=None, mask=None, pe=None, **kwargs):

        is_self_attn = context is None

        context = x if is_self_attn else context
        context_v = x if is_self_attn else context

        context_scales = x_scales if is_self_attn else None
        context_v_scales = x_scales if is_self_attn else None

        if "transformer_options" in kwargs:
            transformer_options = kwargs["transformer_options"]
            step = transformer_options.get("step", -1)
            total_steps = transformer_options.get("total_steps", 0)
            attn_bank = transformer_options.get("attn_bank", None)
            sample_mode = transformer_options.get("sample_mode", None)
            if attn_bank is not None and self.idx in attn_bank["block_map"]:
                len_conds = len(transformer_options["cond_or_uncond"])
                pred_order = transformer_options["pred_order"]
                if (
                    sample_mode == "forward"
                    and total_steps - step - 1 < attn_bank["save_steps"]
                ):
                    step_idx = f"{pred_order}_{total_steps-step-1}"
                    attn_bank["block_map"][self.idx][step_idx] = (
                        x.cpu(),
                        x_scales.cpu(),
                    )
                elif sample_mode == "reverse" and step < attn_bank["inject_steps"]:
                    step_idx = f"{pred_order}_{step}"
                    inject_settings = attn_bank.get("inject_settings", {})
                    if len(inject_settings) > 0:
                        inj = attn_bank["block_map"][self.idx][step_idx]
                        inj_vals = inj[0].to(x.device).repeat(len_conds, 1, 1)
                        if inj[1] is not None:
                            inj_scales = inj[1].to(x.device).repeat(len_conds, 1)

                    if "q" in inject_settings:
                        x, x_scales = inj_vals, inj_scales
                    if "k" in inject_settings:
                        context, context_scales = inj_vals, inj_scales
                    if "v" in inject_settings:
                        context_v, context_v_scales = inj_vals, inj_scales
        else:
            transformer_options = {}
        v_dtype = torch.float8_e4m3fn if use_fp8_attention else torch.bfloat16
        if (
            "stg" in comfy.ldm.modules.attention.optimized_attention.__name__
            or not is_self_attn
        ):
            v_dtype = torch.bfloat16

        q = self.to_q(x, x_scales, torch.bfloat16)
        k = self.to_k(context, context_scales, torch.bfloat16)
        v = self.to_v(context_v, context_v_scales, v_dtype)

        if is_self_attn:
            q = rms_norm_rope(
                q, pe[0], pe[1], self.q_norm.weight, not use_fp8_attention
            )
            k = rms_norm_rope(
                k, pe[0], pe[1], self.k_norm.weight, not use_fp8_attention
            )
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        feta_score = None
        if (
            transformer_options.get("feta_weight", 0) > 0
            and self.idx in transformer_options["feta_layers"]["layers"]
        ):
            feta_score = get_feta_scores(q, k, self.heads, transformer_options)

        alt_attn_fn = transformer_options.get("patches_replace", {}).get("layer", {})
        alt_attn_fn = (
            alt_attn_fn.get(("self_attn", self.idx), None)
            if hasattr(self, "idx")
            else None
        )
        if alt_attn_fn is not None:
            out = alt_attn_fn(
                q,
                k,
                v,
                self.heads,
                attn_precision=self.attn_precision,
                transformer_options=transformer_options,
            )
        elif mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(
                q, k, v, self.heads, attn_precision=self.attn_precision
            )
        else:
            out = comfy.ldm.modules.attention.optimized_attention_masked(
                q, k, v, self.heads, mask, attn_precision=self.attn_precision
            )

        if feta_score is not None:
            out *= feta_score
        return self.to_out[0](out, None)

    return forward


def create_forwards():
    compute_dtype = get_compute_dtype()

    def fused_forward(
        self, x, context=None, attention_mask=None, timestep=None, pe=None, **kwargs
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + timestep.reshape(
                x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1
            )
        ).unbind(dim=2)

        x_quant, x_scales = norm_scale_shift_hadamard_transform(
            x, None, scale_msa, shift_msa, compute_dtype
        )

        x = x + self.attn1(x_quant, x_scales, pe=pe, **kwargs) * gate_msa
        x = x + self.attn2(x, context=context, mask=attention_mask)

        y, y_scales = norm_scale_shift_hadamard_transform(
            x, None, scale_mlp, shift_mlp, compute_dtype
        )
        x = x + self.ff(y, y_scales) * gate_mlp

        return x

    def gelu_forward(self, hidden_states, hidden_states_scales):
        hidden_states = self.proj(hidden_states, hidden_states_scales, torch.bfloat16)
        hidden_states, hidden_states_scales = gelu_hadamard_transform(
            hidden_states, out_dtype=compute_dtype
        )
        return hidden_states, hidden_states_scales

    def ff_forward(
        self, hidden_states: torch.Tensor, hidden_states_scales: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, hidden_states_scales = self.net[0](
            hidden_states, hidden_states_scales
        )
        hidden_states = self.net[2](
            hidden_states, hidden_states_scales, out_dtype=torch.bfloat16
        )
        return hidden_states

    return fused_forward, gelu_forward, ff_forward
