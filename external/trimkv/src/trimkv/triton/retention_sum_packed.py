import math
import torch
from torch.autograd.function import Function
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Forward kernel (masked)
# Computes, for each i: out[i] = sum_{j=0}^i exp((i - j) * x[j]),
# but ONLY when doc_id[i] == doc_id[j] != pad_id
# -----------------------------------------------------------------------------
@triton.jit
def _retention_sum_fwd_kernel_masked(
    x_ptr,           # (B*H, S) float*  (input log-weights)
    out_ptr,         # (B*H, S) float*  (output, float32)
    doc_ptr,         # (B,   S) int32*  (document ids, -1 is padding)
    S: tl.int32,     # sequence length (S_total)
    H: tl.int32,     # number of heads
    PAD_ID: tl.int32,
    BLOCK_M: tl.constexpr,  # rows per program (i indices)
    BLOCK_N: tl.constexpr,  # cols per inner loop (j indices)
):
    pid_bh = tl.program_id(axis=0)   # which (batch*head)
    pid_m  = tl.program_id(axis=1)   # which row block

    # derive batch from (b,h) flattened id
    b = pid_bh // H

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < S

    # Pointers for this (b,h)
    x_bh   = x_ptr   + pid_bh * S
    out_bh = out_ptr + pid_bh * S

    # Pointer for this batch's doc ids
    doc_b  = doc_ptr + b * S

    # Load doc ids for target rows once
    doc_i = tl.load(doc_b + rows, mask=row_mask, other=PAD_ID)  # (BLOCK_M,)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n = 0
    while n < S:
        col_start = n
        cols = col_start + tl.arange(0, BLOCK_N)
        col_mask = cols < S

        # x[j] in float32 for numerical stability, (BLOCK_N,)
        base = tl.load(x_bh + cols, mask=col_mask, other=0.0).to(tl.float32)

        # doc ids for columns (BLOCK_N,)
        doc_j = tl.load(doc_b + cols, mask=col_mask, other=PAD_ID)

        # Broadcast indices
        r = rows[:, None].to(tl.float32)
        c = cols[None, :].to(tl.float32)
        exps = r - c  # (i - j)

        # masks
        tri_mask   = (exps >= 0)
        rows_ok    = row_mask[:, None]
        cols_ok    = col_mask[None, :]
        docs_match = (doc_i[:, None] == doc_j[None, :])
        docs_ok    = (doc_i[:, None] != PAD_ID) & (doc_j[None, :] != PAD_ID)

        mask = tri_mask & rows_ok & cols_ok & docs_match & docs_ok

        vals = tl.exp(base[None, :] * exps)  # exp((i-j)*x[j])
        vals = tl.where(mask, vals, 0.0)

        acc += tl.sum(vals, axis=1)

        n += BLOCK_N

    tl.store(out_bh + rows, acc, mask=row_mask)


# -----------------------------------------------------------------------------
# Backward kernel (masked)
# dL/dx[j] = sum_{i >= j and doc[i]==doc[j]!=pad} dL/dout[i] * (i-j) * exp((i-j) * x[j])
# -----------------------------------------------------------------------------
@triton.jit
def _retention_sum_bwd_kernel_masked(
    x_ptr,           # (B*H, S) float*   (input log-weights, float32)
    do_ptr,          # (B*H, S) float*   (grad wrt output, float32)
    dx_ptr,          # (B*H, S) float*   (grad wrt input,  float32)
    doc_ptr,         # (B,   S) int32*   (document ids)
    S: tl.int32,
    H: tl.int32,
    PAD_ID: tl.int32,
    BLOCK_M: tl.constexpr,  # rows per inner loop (i indices)
    BLOCK_N: tl.constexpr,  # cols per program (j indices)
):
    pid_bh = tl.program_id(axis=0)   # which (batch*head)
    pid_n  = tl.program_id(axis=1)   # which col block (j chunk)

    # derive batch from (b,h) flattened id
    b = pid_bh // H

    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < S

    # Pointers (b,h)
    x_bh  = x_ptr  + pid_bh * S
    do_bh = do_ptr + pid_bh * S
    dx_bh = dx_ptr + pid_bh * S

    # doc ids for this batch
    doc_b = doc_ptr + b * S

    xcols = tl.load(x_bh + cols, mask=col_mask, other=0.0).to(tl.float32)   # (BLOCK_N,)
    doc_j = tl.load(doc_b + cols, mask=col_mask, other=PAD_ID)              # (BLOCK_N,)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    m = 0
    while m < S:
        row_start = m
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < S

        # indices
        r = rows[:, None].to(tl.float32)
        c = cols[None, :].to(tl.float32)
        exps = r - c  # (i - j)

        # doc ids for rows
        doc_i = tl.load(doc_b + rows, mask=row_mask, other=PAD_ID)

        tri_mask   = (exps >= 0)
        rows_ok    = row_mask[:, None]
        cols_ok    = col_mask[None, :]
        docs_match = (doc_i[:, None] == doc_j[None, :])
        docs_ok    = (doc_i[:, None] != PAD_ID) & (doc_j[None, :] != PAD_ID)

        mask = tri_mask & rows_ok & cols_ok & docs_match & docs_ok

        # (i-j) * exp((i-j)*x[j])
        contrib = tl.exp(exps * xcols[None, :]) * exps
        contrib = tl.where(mask, contrib, 0.0)  # (BLOCK_M, BLOCK_N)

        drows = tl.load(do_bh + rows, mask=row_mask, other=0.0).to(tl.float32)  # (BLOCK_M,)
        acc += tl.sum(contrib * drows[:, None], axis=0)

        m += BLOCK_M

    tl.store(dx_bh + cols, acc, mask=col_mask)


# -----------------------------------------------------------------------------
# Autograd wrapper for masked packed inputs
# -----------------------------------------------------------------------------
class RetentionSumTritonPacked(Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,          # (B, H, S_total), float
                doc_mask: torch.Tensor,   # (B, S_total), int32, -1 is padding
                pad_id: int = -1,
                row_block: int = 128,
                col_block: int | None = None):
        assert x.dim() == 3, "x must be (B, H, S_total)"
        assert doc_mask.dim() == 2, "doc_mask must be (B, S_total)"
        B, H, S = x.shape
        assert doc_mask.shape == (B, S)
        assert doc_mask.dtype in (torch.int32, torch.int64)
        assert doc_mask.device == x.device

        BHS = B * H
        BLOCK_M = int(row_block)
        BLOCK_N = int(col_block) if col_block is not None else min(128, S)

        # layouts
        x_2d     = x.contiguous().view(BHS, S)
        out_2d   = torch.empty((BHS, S), device=x.device, dtype=torch.float32)
        doc_2d   = doc_mask.to(torch.int32).contiguous()  # (B, S)

        grid = lambda META: (BHS, triton.cdiv(S, META["BLOCK_M"]))
        _retention_sum_fwd_kernel_masked[grid](
            x_2d, out_2d, doc_2d,
            S, H, int(pad_id),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x, doc_2d)
        ctx.meta = (BLOCK_M, BLOCK_N, S, B, H, int(pad_id))
        return out_2d.view(B, H, S)

    @staticmethod
    def backward(ctx, d_out):
        x, doc_2d = ctx.saved_tensors
        BLOCK_M, BLOCK_N, S, B, H, pad_id = ctx.meta
        BHS = B * H

        # grads in f32, then cast back
        x_f32   = x.to(torch.float32).contiguous().view(BHS, S)
        do_f32  = d_out.to(torch.float32).contiguous().view(BHS, S)
        dx_f32  = torch.zeros_like(x_f32)

        grid = lambda META: (BHS, triton.cdiv(S, META["BLOCK_N"]))
        _retention_sum_bwd_kernel_masked[grid](
            x_f32, do_f32, dx_f32, doc_2d,
            S, H, int(pad_id),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        dx = dx_f32.view(B, H, S).to(x.dtype)
        return dx, None, None, None, None


def retention_sum_packed_triton(x: torch.Tensor,
                                doc_mask: torch.Tensor=None,
                                pad_id: int = -1,
                                row_block: int = 128,
                                col_block: int | None = None):
    """
    Packed/masked Triton version.

    Args:
        x:        (B, H, S_total) log retention weights (log_beta)
        doc_mask: (B, S_total) int32/64 ids; positions with the same id belong
                  to the same document; 'pad_id' marks padding that contributes 0
        pad_id:   sentinel in doc_mask (default -1)
    Returns:
        out: (B, H, S_total) float32
             out[..., i] = sum_{j<=i and doc[i]==doc[j]!=pad} exp((i-j)*x[..., j])
    """
    if doc_mask is None:
        doc_mask = torch.zeros(x.shape[0], x.shape[2], dtype=torch.int32, device=x.device)
    return RetentionSumTritonPacked.apply(x, doc_mask, pad_id, row_block, col_block)
