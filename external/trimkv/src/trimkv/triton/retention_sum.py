import math
import torch
from functools import partial
from torch import nn, einsum
from torch.autograd.function import Function
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Forward kernel
# Computes, for each i: out[i] = sum_{j=0}^i exp((i - j) * x[j])
# where x is the log of the retention factor (e.g., logsigmoid outputs)
# -----------------------------------------------------------------------------
@triton.jit
def _retention_sum_fwd_kernel(
    x_ptr,          # (BHS, S) float*  (input log-weights)
    out_ptr,        # (BHS, S) float*  (output, float32)
    S: tl.int32,    # sequence length
    BLOCK_M: tl.constexpr,  # rows per program (i indices)
    BLOCK_N: tl.constexpr,  # cols per inner loop (j indices)
):
    pid_bh = tl.program_id(axis=0)   # which (batch*head)
    pid_m  = tl.program_id(axis=1)   # which row block

    row_start = pid_m * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < S

    # Pointers to this (b*h) row
    x_bh   = x_ptr   + pid_bh * S
    out_bh = out_ptr + pid_bh * S

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n = 0
    while n < S:
        col_start = n
        cols = col_start + tl.arange(0, BLOCK_N)
        col_mask = cols < S

        # x[j] in float32 for numerical stability
        base = tl.load(x_bh + cols, mask=col_mask, other=0.0).to(tl.float32)  # (BLOCK_N,)

        # Broadcast to (BLOCK_M, BLOCK_N)
        r = rows[:, None].to(tl.float32)
        c = cols[None, :].to(tl.float32)
        exps = r - c  # (i - j)

        tri_mask = (exps >= 0) & row_mask[:, None] & col_mask[None, :]

        vals = tl.exp(base[None, :] * exps)  # exp((i-j)*x[j])
        vals = tl.where(tri_mask, vals, 0.0)

        # Sum over j for each row-i in the block
        acc += tl.sum(vals, axis=1)

        n += BLOCK_N

    tl.store(out_bh + rows, acc, mask=row_mask)


# -----------------------------------------------------------------------------
# Backward kernel
# dL/dx[j] = sum_{i >= j} dL/dout[i] * (i-j) * exp((i-j) * x[j])
# -----------------------------------------------------------------------------
@triton.jit
def _retention_sum_bwd_kernel(
    x_ptr,          # (BHS, S) float*  (input log-weights, float32)
    do_ptr,         # (BHS, S) float*  (grad wrt output, float32)
    dx_ptr,         # (BHS, S) float*  (grad wrt input, float32)
    S: tl.int32,    # sequence length
    BLOCK_M: tl.constexpr,  # rows per inner loop (i indices)
    BLOCK_N: tl.constexpr,  # cols per program (j indices)
):
    pid_bh = tl.program_id(axis=0)   # which (batch*head)
    pid_n  = tl.program_id(axis=1)   # which col block (j chunk)

    col_start = pid_n * BLOCK_N
    cols = col_start + tl.arange(0, BLOCK_N)
    col_mask = cols < S

    # Pointers to this (b*h) row
    x_bh  = x_ptr  + pid_bh * S
    do_bh = do_ptr + pid_bh * S
    dx_bh = dx_ptr + pid_bh * S

    xcols = tl.load(x_bh + cols, mask=col_mask, other=0.0).to(tl.float32)  # (BLOCK_N,)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    m = 0
    while m < S:
        row_start = m
        rows = row_start + tl.arange(0, BLOCK_M)
        row_mask = rows < S

        r = rows[:, None].to(tl.float32)
        c = cols[None, :].to(tl.float32)
        exps = r - c  # (i - j)

        tri_mask = (exps >= 0) & row_mask[:, None] & col_mask[None, :]

        # (i-j) * exp((i-j)*x[j])
        contrib = tl.exp(exps * xcols[None, :]) * exps
        contrib = tl.where(tri_mask, contrib, 0.0)  # (BLOCK_M, BLOCK_N)

        drows = tl.load(do_bh + rows, mask=row_mask, other=0.0).to(tl.float32)  # (BLOCK_M,)
        acc += tl.sum(contrib * drows[:, None], axis=0)

        m += BLOCK_M

    tl.store(dx_bh + cols, acc, mask=col_mask)


# -----------------------------------------------------------------------------
# Autograd wrapper (drop-in)
# -----------------------------------------------------------------------------
class RetentionSumTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, row_block: int = 128, col_block: int | None = None):
        # x: (B, H, S), log-space input (e.g., logsigmoid of retention gate)
        assert x.dim() == 3, "expected input shape (B, H, S)"
        B, H, S = x.shape
        BHS = B * H
        BLOCK_M = int(row_block)
        BLOCK_N = int(col_block) if col_block is not None else min(128, S)

        x_2d = x.contiguous().view(BHS, S)
        out = torch.empty((BHS, S), device=x.device, dtype=torch.float32)

        grid = lambda META: (BHS, triton.cdiv(S, META["BLOCK_M"]))
        _retention_sum_fwd_kernel[grid](
            x_2d, out, S,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(x)  # original dtype
        ctx.meta = (BLOCK_M, BLOCK_N, S, B, H)

        return out.view(B, H, S)

    @staticmethod
    def backward(ctx, d_out):
        (x,) = ctx.saved_tensors
        BLOCK_M, BLOCK_N, S, B, H = ctx.meta
        BHS = B * H

        # We compute grads in f32 for stability, then cast back to x.dtype.
        x_f32   = x.to(torch.float32).contiguous().view(BHS, S)
        do_f32  = d_out.to(torch.float32).contiguous().view(BHS, S)
        dx_f32  = torch.zeros_like(x_f32)

        grid = lambda META: (BHS, triton.cdiv(S, META["BLOCK_N"]))
        _retention_sum_bwd_kernel[grid](
            x_f32, do_f32, dx_f32, S,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        dx = dx_f32.view(B, H, S).to(x.dtype)
        return dx, None, None


def retention_sum_triton(x: torch.Tensor, row_block: int = 128, col_block: int | None = None):
    """
    x: (B, H, S) tensor of log retention weights (e.g., logsigmoid outputs)
    Returns float32 output with shape (B, H, S):
      out[..., i] = sum_{j=0}^i exp((i-j) * x[..., j])
    """
    return RetentionSumTriton.apply(x, row_block, col_block)


def retention_sum(input, row_chunk_size=128, col_chunk_size=None):
    *batch, seqlen = input.shape
    device = input.device
    dtype = torch.float32  # use float32 for numerical stability
    out = torch.zeros(*batch, seqlen, device=device, dtype=dtype)
    col_chunk_size = col_chunk_size or seqlen
    for row_chunk_start in range(0, seqlen, row_chunk_size):
        for col_chunk_start in range(0, seqlen, col_chunk_size):
            row_chunk_end = min(row_chunk_start + row_chunk_size, seqlen)
            col_chunk_end = min(col_chunk_start + col_chunk_size, seqlen)
            if col_chunk_start >= row_chunk_end:
                continue

            rows = torch.arange(row_chunk_start, row_chunk_end, device=input.device)
            cols = torch.arange(col_chunk_start, col_chunk_end, device=input.device)

            exps = (rows.unsqueeze(-1) - cols.unsqueeze(0)).to(dtype)
            mask = (exps >= 0)
            exps = exps.clamp(min=0.0)
            base = input[..., cols].unsqueeze(-2)  # shape: (..., 1, col_chunk_size)
            log_vals = base * exps
            mask = mask.view((1,)*len(batch) + exps.shape).expand_as(log_vals)
            log_vals = log_vals.masked_fill(~mask, -float('inf'))
            vals = torch.exp(log_vals)
            out[..., rows] += vals.sum(dim=-1)
    return out


class RetentionSum(Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, input, row_chunk_size, col_chunk_size=None):
        # retention weights is logsigmoid of the retention gate output
        *batch, seqlen = input.shape
        device = input.device
        dtype = torch.float32  # use float32 for numerical stability
        out = torch.zeros(*batch, seqlen, device=device, dtype=dtype)
        col_chunk_size = col_chunk_size or seqlen
        for row_chunk_start in range(0, seqlen, row_chunk_size):
            for col_chunk_start in range(0, seqlen, col_chunk_size):
                row_chunk_end = min(row_chunk_start + row_chunk_size, seqlen)
                col_chunk_end = min(col_chunk_start + col_chunk_size, seqlen)
                if col_chunk_start >= row_chunk_end:
                    continue

                rows = torch.arange(row_chunk_start, row_chunk_end, device=input.device)
                cols = torch.arange(col_chunk_start, col_chunk_end, device=input.device)

                exps = (rows.unsqueeze(-1) - cols.unsqueeze(0)).to(dtype)
                mask = (exps >= 0)
                exps = exps.clamp(min=0.0)
                base = input[..., cols].unsqueeze(-2)  # shape: (..., 1, col_chunk_size)
                log_vals = base * exps
                mask = mask.view((1,)*len(batch) + exps.shape).expand_as(log_vals)
                log_vals = log_vals.masked_fill(~mask, -float('inf'))
                vals = torch.exp(log_vals)
                out[..., rows] += vals.sum(dim=-1)

        ctx.args = (row_chunk_size, col_chunk_size)
        ctx.save_for_backward(input)

        return out

    @staticmethod
    @torch.no_grad()
    def backward(ctx, d_out):
        row_chunk_size, col_chunk_size = ctx.args
        input, = ctx.saved_tensors

        *batch, seqlen = input.shape
        device, dtype = input.device, input.dtype
        d_input = torch.zeros_like(input, dtype=dtype, device=device)

        for row_chunk_start in range(0, seqlen, row_chunk_size):
            for col_chunk_start in range(0, seqlen, col_chunk_size):
                row_chunk_end = min(row_chunk_start + row_chunk_size, seqlen)
                col_chunk_end = min(col_chunk_start + col_chunk_size, seqlen)
                if col_chunk_start >= row_chunk_end:
                    continue

                rows = torch.arange(row_chunk_start, row_chunk_end, device=device)
                cols = torch.arange(col_chunk_start, col_chunk_end, device=device)

                exps = (rows.unsqueeze(-1) - cols.unsqueeze(0)).clamp(min=0.0).to(torch.float32)
                base = input[..., cols].unsqueeze(-2)  # shape: (..., 1, col_chunk_size)
                vals = torch.exp(exps * base) * exps
                d_vals = d_out[..., rows].unsqueeze(-1)  # shape: (..., row_chunk_size, 1)
                d_input[..., cols] += (d_vals * vals).sum(dim=-2).to(dtype)

        return d_input, None, None
