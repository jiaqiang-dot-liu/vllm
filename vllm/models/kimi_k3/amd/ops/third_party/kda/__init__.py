# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# AMD/ROCm vendored copy of the Kimi-K3 KDA triton kernels.
#
# Provenance: mirror of vllm/models/kimi_k3/nvidia/ops/third_party/kda (tracker
# mke-tracker @ 7adebfcf9; FLA vendored per PRs #39/#86). Split per-vendor so
# AMD can carry gfx950-specific kernel changes without touching the NVIDIA copy.
#
# fla-org/flash-linear-attention#869 (the unmerged ROCm fixes our earlier
# amd_fla shim carried against FLA 0.5.0) is covered here by the *newer* vendored
# FLA rather than the literal patch:
#   - transpose-state-layout workaround: N/A (kernels rewritten; no
#     transpose_state_layout path remains),
#   - AMD autotune configs: present in chunk.py (is_amd num_warps/num_stages
#     branches) -- but note this covers the PREFILL kernels only; see the delta
#     below for the decode launch, which carries no autotune on either copy,
#   - OOB-mask correctness fix: present (all tl.load use mask=..., other=0).
# Validated on gfx950: no core-dump, gsm8k 94.1%.
#
# AMD-specific deltas vs the NVIDIA copy:
#   - fused_recurrent.py: fused_recurrent_kda_packed_decode() resolves its
#     (BV, num_warps) launch tile through _kda_packed_decode_tile() instead of
#     the inherited hard-coded BV=min(next_pow2(V), 32) / num_warps=4, and the
#     default BV moves 32 -> 64 on gfx950. This launch is the decode-step hot
#     path for all 69 KDA layers and was the one kernel here with neither an
#     autotune config nor an is_amd branch, so it still carried an NVIDIA-shaped
#     tile. BV partitions only the V axis (all reductions run along axis=1 over
#     K and stay whole within a workgroup, and each V block owns a disjoint
#     slice of state rows), so this changes work decomposition, not numerics.
#     Overridable via VLLM_KDA_DECODE_BV / VLLM_KDA_DECODE_NUM_WARPS;
#     VLLM_KDA_DECODE_BV=32 restores the NVIDIA-identical tile.
# Otherwise keep in sync with the NVIDIA copy on FLA updates; any divergence
# should be an intentional, documented gfx950-specific change (a #869-style
# AMD-only fix).

from .chunk import (
    chunk_kda,
    chunk_kda_fwd,
    chunk_kda_with_fused_gate,
    chunk_kda_with_fused_gate_fwd,
    fused_kda_gate,
    fused_kda_gate_chunk_cumsum,
)
from .fused_recurrent import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)

__all__ = [
    "chunk_kda",
    "chunk_kda_fwd",
    "chunk_kda_with_fused_gate",
    "chunk_kda_with_fused_gate_fwd",
    "fused_kda_gate",
    "fused_kda_gate_chunk_cumsum",
    "fused_recurrent_kda",
    "fused_recurrent_kda_fwd",
    "fused_recurrent_kda_packed_decode",
]
