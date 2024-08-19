from __future__ import annotations

import torch
import torch.nn as nn

from mambapainter.models.mambaformer import MambaFormer
from mambapainter.models.vmamba import VSSM


class MambaStrokePredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_strokes: int,
        n_params: int,
        n_layer: int = 16,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        cross_attn_first: bool = True,
        identical_stroke_token: bool = False,
        ssm_layer: str = 'Mamba2',
        ssm_d_state: int = 128,  # This should be adjusted when switching Mamba versions.
        encoder_patch_size: int = 4,
        encoder_image_size: int = 128,
        encoder_in_chans: int = 3,
        encoder_depths: list[int] = (2, 2, 9),
        encoder_dims: list[int] = (64, 128, 256),
    ) -> None:
        super().__init__()
        assert encoder_in_chans in {3, 6}, f'"encoder_in_chans" must be one of [3, 6]. Got {encoder_in_chans}.'

        self.image_size = encoder_image_size
        self.canvas_is_stacked = encoder_in_chans == 6

        self.encoder = VSSM(
            patch_size=encoder_patch_size,
            depths=encoder_depths,
            dims=encoder_dims,
            ssm_d_state=ssm_d_state,
            ssm_init='v2',
            forward_type='m0' if ssm_layer == 'Mamba2' else 'v3',
            imgsize=encoder_image_size,
            in_chans=encoder_in_chans,
        )
        d_ref = self.encoder.num_features

        self.mamba = MambaFormer(
            n_params=n_params,
            n_strokes=n_strokes,
            d_model=d_model,
            d_ref=d_ref,
            n_layer=n_layer,
            ssm_layer=ssm_layer,
            ssm_cfg=dict(d_state=ssm_d_state),
            norm_epsilon=1e-5,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            cross_attn_first=cross_attn_first,
            identical_stroke_token=identical_stroke_token,
        )

    def encode(self, x):
        return self.encoder(x)

    def predict_params(self, x, stroke_index: int | torch.LongTensor = None):
        return self.mamba(x, stroke_index=stroke_index)

    def forward(
        self,
        x,
        stroke_index: int | torch.LongTensor = None,
    ):
        x = self.encode(x)
        x = self.predict_params(x, stroke_index=stroke_index)
        return x
