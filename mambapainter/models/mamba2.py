"""VMamba

MIT License

Copyright (c) 2024 MzeroMiko
"""

import torch
import torch.nn.functional as F
from mamba_ssm.modules.ssd_minimal import ssd_minimal_discrete


def mamba_chunk_scan_combined_torch(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float('inf')),
    return_final_states=False,
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, ngroups, dstate = B.shape
    nheads, headdim = x.shape[2:]

    while seqlen % chunk_size != 0:
        chunk_size = chunk_size >> 1

    if nheads != ngroups:
        assert nheads % ngroups == 0
        B = (
            B.view(batch, seqlen, ngroups, 1, dstate)
            .repeat(1, 1, 1, nheads // ngroups, 1)
            .view(batch, seqlen, nheads, dstate)
        )
        C = (
            C.view(batch, seqlen, ngroups, 1, dstate)
            .repeat(1, 1, 1, nheads // ngroups, 1)
            .view(batch, seqlen, nheads, dstate)
        )

    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    u = x * dt.unsqueeze(-1)
    w = A * dt

    y, state = ssd_minimal_discrete(u, w, B, C, block_len=chunk_size, initial_states=initial_states)
    if D is not None:
        y = y + D.view(y.shape[-2], -1) * x
    if z is not None:
        y = y * (z * torch.sigmoid(z))

    return (y, state) if return_final_states else y


WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton  # noqa: F401
except ImportError:
    WITH_TRITON = False

if WITH_TRITON:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
else:
    mamba_chunk_scan_combined = None


def selective_scan_chunk_fn(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float('inf')),
    return_final_states=False,
    backend=None,
):
    fn = mamba_chunk_scan_combined_torch if backend == 'torch' or (not WITH_TRITON) else mamba_chunk_scan_combined
    return fn(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        seq_idx,
        cu_seqlens,
        dt_softplus,
        dt_limit,
        return_final_states,
    )
