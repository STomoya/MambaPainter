"""Mamba + Cross-attention network.

Based on Zigma architecture.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self._attention_mode = 'sdpa' if hasattr(F, 'scaled_dot_product_attention') else 'math'

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, c):
        B, L, C = x.shape

        if c.ndim == 2:
            c = c.unsqueeze(1)

        q = self.to_q(x)
        k = self.to_k(c)
        v = self.to_v(c)

        q, k, v = (rearrange(t, 'B L (H D) -> B H L D', H=self.heads) for t in (q, k, v))  # B H L D
        if self._attention_mode == 'sdpa':
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, 'B H L D -> B L (H D)')
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        return self.to_out(x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        ref_dim: int,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        cross_attn_first: bool = False,
    ):
        super().__init__()
        self.cross_attn_first = cross_attn_first

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, 'RMSNorm import fails'
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), 'Only LayerNorm and RMSNorm are supported for fused_add_norm'

        # self.use_cross_attn = use_cross_attn
        adaln_mul = 3 * 2
        self.adaln_affine = nn.Sequential(nn.SiLU(), nn.Linear(ref_dim, dim * adaln_mul, bias=True))
        self.use_cross_attn = True
        if self.use_cross_attn:
            self.cross_attn = CrossAttention(query_dim=dim, context_dim=ref_dim, heads=8, dim_head=64)
            self.norm_msa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        residual: torch.Tensor | None = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        shift_mba, scale_mba, gate_mba, shift_msa, scale_msa, gate_msa = self.adaln_affine(
            condition.mean(dim=1),
        ).chunk(6, dim=1)

        if not self.cross_attn_first:
            hidden_states = hidden_states + gate_mba.unsqueeze(1) * self.mixer(
                self.modulate(hidden_states, shift=shift_mba, scale=scale_mba),
                inference_params=inference_params,
            )

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.cross_attn(
                self.modulate(self.norm_msa(hidden_states), shift=shift_msa, scale=scale_msa),
                c=condition,
            )

        else:
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * self.cross_attn(
                self.modulate(self.norm_msa(hidden_states), shift=shift_msa, scale=scale_msa),
                c=condition,
            )

            hidden_states = hidden_states + gate_mba.unsqueeze(1) * self.mixer(
                self.modulate(hidden_states, shift=shift_mba, scale=scale_mba),
                inference_params=inference_params,
            )

        # hidden_states = hidden_states + self.mixer(hidden_states, inference_params=inference_params)
        # hidden_states = hidden_states + self.cross_attn(self.norm_msa(hidden_states), c=condition)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_ref,
    ssm_layer='Mamba2',
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    cross_attn_first=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba2 if ssm_layer == 'Mamba2' else Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        dim=d_model,
        ref_dim=d_ref,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        cross_attn_first=cross_attn_first,
    )
    block.layer_idx = layer_idx
    return block


class MambaFormerBackbone(nn.Module):
    def __init__(
        self,
        d_model=768,
        d_ref=768,
        n_layer: int = 64,
        ssm_layer: str = 'Mamba2',
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        cross_attn_first: bool = False,
    ) -> None:
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.depth = n_layer

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_ref,
                    ssm_layer=ssm_layer,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    cross_attn_first=cross_attn_first,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, ref_embeddings, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                ref_embeddings,
                residual,
                inference_params=inference_params,
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ['out_proj.weight', 'fc2.weight']:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)  # noqa: PLW2901


class MambaFormer(nn.Module):
    def __init__(
        self,
        n_params: int = 8,
        n_strokes: int = 50,
        d_model: int = 512,
        d_ref: int = 512,
        n_layer: int = 64,
        ssm_layer: str = 'Mamba2',
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        cross_attn_first: bool = False,
        identical_stroke_token: bool = False,
    ) -> None:
        super().__init__()
        self.identical_stroke_token = identical_stroke_token

        if not identical_stroke_token:
            self.stroke_query = nn.Parameter(torch.randn(1, n_strokes, d_model) * 0.02)
            self.seq_repeat = 1
        else:
            self.stroke_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.seq_repeat = n_strokes

        self.backbone = MambaFormerBackbone(
            d_model=d_model,
            d_ref=d_ref,
            n_layer=n_layer,
            ssm_layer=ssm_layer,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            cross_attn_first=cross_attn_first,
        )
        self.head = nn.Linear(d_model, n_params, bias=False)

        self.init_weights()

    def init_weights(self):
        for block in self.backbone.layers:
            if hasattr(block, 'adaln_affine'):
                nn.init.constant_(block.adaln_affine[-1].weight, 0.0)
                nn.init.constant_(block.adaln_affine[-1].bias, 0.0)
        self.apply(partial(_init_weights, n_layer=self.backbone.depth))
        nn.init.normal_(self.head.weight, std=0.02)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        input,
        stroke_index: int | torch.LongTensor = None,
        inference_params: dict | None = None,
        return_logits: bool = False,
    ):
        strokes = self.stroke_query.repeat(input.size(0), self.seq_repeat, 1)
        if stroke_index is not None:
            strokes = strokes[:, :stroke_index, :].contiguous()

        hidden_states = self.backbone(strokes, input, inference_params=inference_params)
        logits = self.head(hidden_states)

        params = torch.sigmoid(logits)

        if return_logits:
            return params, logits

        return params
