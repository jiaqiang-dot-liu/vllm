# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA AITER block-sparse attend for MiniMax M3."""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import (
    AttentionLayer,
)


# One slot per rank. Holds the decode sparse block table built by the most
# recent sparse layer that actually ran the lightning indexer, so the
# ``index_topk_freq`` layers that reuse that layer's ``topk_indices_buffer``
# selection can reuse the table derived from it instead of rebuilding it.
#
# Written by every non-skipped layer, and the first sparse layer of a forward
# pass is never skipped, so the slot is refreshed at the start of every pass.
_DECODE_SBT_SLOT: dict = {}


def _decode_sparse_block_table(
    layer: AttentionLayer,
    topk_rows: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    block_page_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the page-16 decode sparse block table, or reuse a shared one.

    A layer with ``skip_index_topk`` set did not run its indexer: it reads the
    selection a preceding layer wrote into the shared ``topk_indices_buffer``
    this same forward pass (see ``_should_skip_index_topk`` and
    ``MiniMaxM3SparseAttention._run_attention``). The sparse block table is a
    pure function of that selection plus ``block_table`` / ``seq_lens`` /
    ``decode_query_len`` / the page stride, all identical across the group, so
    the preceding layer's table is exactly what this layer would rebuild.

    With ``use_index_cache`` off, ``skip_index_topk`` is always False and every
    layer builds its own table, i.e. behaviour is unchanged.
    """
    from vllm.models.minimax_m3.amd.ops.sparse_pa import (
        minimax_m3_build_sparse_block_table_decode,
    )

    total_q = topk_rows.shape[1]
    topk_width = topk_rows.shape[-1]
    key = (
        total_q,
        topk_width,
        decode_query_len,
        block_page_stride,
        block_table.data_ptr(),
        seq_lens.data_ptr(),
    )

    if getattr(layer, "skip_index_topk", False):
        cached = _DECODE_SBT_SLOT.get("decode")
        if cached is not None and cached[2] == key:
            return cached[0], cached[1]
        # Shape or buffer identity moved under us (first call after a resize or
        # a capture/replay boundary): fall through and build, which is always
        # correct, just not free.

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_decode(
        topk_rows,
        block_table,
        seq_lens,
        decode_query_len,
        block_page_stride,
    )
    _DECODE_SBT_SLOT["decode"] = (sparse_bt, sparse_ctx, key)
    return sparse_bt, sparse_ctx


class MiniMaxM3SparseAiterPAImpl(MiniMaxM3SparseImpl):
    """ROCm AITER page-16 SHUFFLE sparse paged attention."""

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            _block_page_stride,
            _run_gluon_decode,
            minimax_m3_sparse_attn_prefill_aiter,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        if self.num_kv_heads != 1:
            raise NotImplementedError(
                "MiniMax-M3 AITER sparse PA currently requires per-rank "
                f"num_kv_heads == 1, got {self.num_kv_heads}"
            )

        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        k_cache, v_cache = layer.get_aiter_sparse_pa_kv_cache()  # type: ignore[attr-defined]
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            sparse_bt, sparse_ctx = _decode_sparse_block_table(
                layer,
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                d.decode_query_len,
                _block_page_stride(k_cache, v_cache),
            )
            _run_gluon_decode(
                q[:nd],
                k_cache,
                v_cache,
                sparse_bt,
                sparse_ctx,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                k_scale,
                v_scale,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            minimax_m3_sparse_attn_prefill_aiter(
                q[nd:],
                k_cache,
                v_cache,
                topk[:, nd:num_tokens, :],
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                k_scale=k_scale,
                v_scale=v_scale,
            )
        return output
