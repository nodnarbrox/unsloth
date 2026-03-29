# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
M40 Maxwell GPU (compute capability 5.2) fallback implementations.

Triton requires sm_70+ (Volta and newer). This module provides pure PyTorch
implementations of all Triton kernel functions so Unsloth can run on Maxwell
GPUs (e.g. Tesla M40, GTX 9xx) without Triton.
"""

import torch
import math


# ============================================================================
# Cross Entropy Loss
# ============================================================================

class Fast_CrossEntropyLoss_M40(torch.autograd.Function):
    """Pure PyTorch cross entropy loss for Maxwell GPUs."""

    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape
        device = logits.device
        labels = labels.to(device)

        logits_f = logits.float()
        if logit_scaling:
            logits_f = logit_scaling * logits_f
        if logit_softcapping:
            logits_f = logit_softcapping * torch.tanh(logits_f / logit_softcapping)

        loss = torch.nn.functional.cross_entropy(
            logits_f, labels, ignore_index=-100, reduction='none'
        )

        ctx.save_for_backward(logits, labels)
        ctx.logit_softcapping = logit_softcapping
        ctx.logit_scaling = logit_scaling
        return loss

    @staticmethod
    def backward(ctx, dlosses):
        logits, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        logits_f = logits.float()
        if ctx.logit_scaling:
            logits_f = ctx.logit_scaling * logits_f
        if ctx.logit_softcapping:
            logits_f = ctx.logit_softcapping * torch.tanh(logits_f / ctx.logit_softcapping)

        # Compute softmax gradient
        probs = torch.softmax(logits_f, dim=-1)
        mask = labels != -100
        rows = torch.arange(n_rows, device=labels.device)
        probs[rows[mask], labels[mask]] -= 1.0
        probs[~mask] = 0.0
        grad = probs * dlosses.unsqueeze(-1)

        # Chain rule for softcapping
        if ctx.logit_softcapping:
            t = ctx.logit_softcapping
            tanh_val = torch.tanh(logits.float() * (ctx.logit_scaling if ctx.logit_scaling else 1.0) / t)
            grad = grad * (1.0 - tanh_val * tanh_val)
        if ctx.logit_scaling:
            grad = grad * ctx.logit_scaling

        return grad.to(logits.dtype), None, None, None


def fast_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)

    device = logits.device
    loss = Fast_CrossEntropyLoss_M40.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    if torch.is_tensor(n_items):
        n_items = n_items.to(device)
    return loss.sum() / n_items


# ============================================================================
# RMS LayerNorm
# ============================================================================

class Fast_RMS_Layernorm_M40(torch.autograd.Function):
    """Pure PyTorch RMS LayerNorm for Maxwell GPUs."""

    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim)
        n_rows, n_cols = X.shape

        X_f = X.float()
        row_var = (X_f * X_f).mean(dim=-1, keepdim=True)
        inv_var = torch.rsqrt(row_var + eps)

        normed = X_f * inv_var
        if gemma:
            output = normed * (W.float() + 1.0)
        else:
            normed_cast = normed.to(W.dtype)
            output = normed_cast * W

        ctx.save_for_backward(X, W, inv_var.squeeze(-1))
        ctx.eps = eps
        ctx.gemma = gemma
        return output.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)
        X, W, inv_var = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        X_f = X.float()
        dY_f = dY.float()
        W_f = W.float()
        inv_var_2d = inv_var.unsqueeze(-1)  # (n_rows, 1)

        normed = X_f * inv_var_2d

        if ctx.gemma:
            dY_W = dY_f * (W_f + 1.0)
        else:
            dY_W = dY_f * W_f

        rowsum_dY_normed = (dY_W * normed).sum(dim=-1, keepdim=True)
        output = inv_var_2d / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)

        if ctx.gemma:
            dX = output.to(dY.dtype).view(*shape)
        else:
            # Original Triton kernel writes in-place to dY
            dX = output.to(dY.dtype).view(*shape)

        return dX, None, None, None


# ============================================================================
# LayerNorm
# ============================================================================

class Fast_Layernorm_M40(torch.autograd.Function):
    """Pure PyTorch LayerNorm for Maxwell GPUs."""

    @staticmethod
    def forward(ctx, X, W, b, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape

        X_f = X.float()
        W_f = W.float()
        b_f = b.float()

        mean = X_f.mean(dim=-1, keepdim=True)
        centered = X_f - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        inv_var = torch.rsqrt(var + eps)

        output = (centered * inv_var) * W_f + b_f

        ctx.save_for_backward(X, W, b, inv_var.squeeze(-1), mean.squeeze(-1))
        ctx.eps = eps
        return output.to(X.dtype).view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, b, inv_var, mean = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        X_f = X.float()
        dY_f = dY.float()
        W_f = W.float()
        inv_var_2d = inv_var.unsqueeze(-1)
        mean_2d = mean.unsqueeze(-1)

        normed = (X_f - mean_2d) * inv_var_2d
        dY_W = dY_f * W_f
        dX = (
            dY_W
            - dY_W.mean(dim=-1, keepdim=True)
            - normed * (dY_W * normed).mean(dim=-1, keepdim=True)
        ) * inv_var_2d

        dX = dX.to(dY.dtype).view(*shape)
        return dX, None, None, None, None


# ============================================================================
# SwiGLU
# ============================================================================

def swiglu_fg_kernel(e, g):
    """SwiGLU forward: h = SiLU(e) * g"""
    batch, seq_len, hd = e.shape
    e_f = e.float()
    f = e_f * torch.sigmoid(e_f)
    f = f.to(g.dtype)
    h = f * g
    return h


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    """SwiGLU backward kernel.
    Computes h, df, de in-place in the DW, e, g buffers.
    DW -> h = f * g
    e  -> df = DW * f
    g  -> de
    """
    batch_seq_len, hd = e.shape
    e_f = e.float()
    DW_orig = DW.clone()
    g_orig = g.clone()

    se = torch.sigmoid(e_f)
    f = (se * e_f).to(DW.dtype)
    h = f * g_orig

    df = DW_orig * f
    dg = DW_orig * g_orig
    de = (dg.float() * se * (1.0 + e_f * (1.0 - se))).to(DW.dtype)

    DW.data.copy_(h)
    e.data.copy_(df)
    g.data.copy_(de)
    return DW, e, g


# ============================================================================
# GEGLU
# ============================================================================

def geglu_exact_forward_kernel(gate, up):
    """GEGLU exact forward: h = GELU(gate) * up"""
    batch, seq_len, hd = gate.shape
    e_f = gate.float()
    # f = 0.5 * e * (1 + erf(e / sqrt(2)))
    f = 0.5 * e_f * (torch.erf(e_f * (2.0 ** -0.5)) + 1.0)
    f = f.to(up.dtype)
    return f * up


def geglu_exact_backward_kernel(DW, e, g):
    """GEGLU exact backward kernel.
    DW -> h = f * g
    e  -> df = DW * f
    g  -> de
    """
    batch_seq_len, hd = e.shape
    e_f = e.float()
    DW_orig = DW.clone()
    g_orig = g.clone()

    f_partial = 0.5 * (torch.erf(e_f * (2.0 ** -0.5)) + 1.0)
    f = (f_partial * e_f).to(DW.dtype)

    h = f * g_orig
    df = DW_orig * f
    dg = DW_orig * g_orig

    # df/de = 0.5 * (1 + erf(e/sqrt(2))) + (1/sqrt(2*pi)) * e * exp(-0.5 * e^2)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial + t * e_f * torch.exp(-0.5 * e_f * e_f)
    de = (dg.float() * df_de).to(DW.dtype)

    DW.data.copy_(h)
    e.data.copy_(df)
    g.data.copy_(de)
    return DW, e, g


def geglu_approx_forward_kernel(gate, up):
    """GEGLU approximate forward: h = GELU_approx(gate) * up"""
    batch, seq_len, hd = gate.shape
    e_f = gate.float()
    s = 0.7978845608028654  # sqrt(2/pi)
    f = 0.5 * e_f * (torch.tanh(s * e_f * (1.0 + 0.044715 * e_f * e_f)) + 1.0)
    f = f.to(up.dtype)
    return f * up


def geglu_approx_backward_kernel(DW, e, g):
    """GEGLU approximate backward kernel.
    DW -> h = f * g
    e  -> df = DW * f
    g  -> de
    """
    batch_seq_len, hd = e.shape
    e_f = e.float()
    DW_orig = DW.clone()
    g_orig = g.clone()

    s = 0.7978845608028654  # sqrt(2/pi)
    a = s * e_f
    b = a * 0.044715 * e_f * e_f
    T = 1.0 + torch.tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2

    f = (T2 * e_f).to(DW.dtype)
    h = f * g_orig
    df = DW_orig * f
    dg = DW_orig * g_orig
    de = (dg.float() * df_de).to(DW.dtype)

    DW.data.copy_(h)
    e.data.copy_(df)
    g.data.copy_(de)
    return DW, e, g


# ============================================================================
# RoPE Embedding
# ============================================================================

class Fast_RoPE_Embedding_M40(torch.autograd.Function):
    """Pure PyTorch RoPE embedding for Maxwell GPUs."""

    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        half = head_dim // 2

        # Q * cos + rotate_half(Q) * sin
        cos_expanded = cos[:seq_len, :half].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, half)
        sin_expanded = sin[:seq_len, :half].unsqueeze(0).unsqueeze(2)

        Q0 = Q[..., :half]
        Q1 = Q[..., half:]
        Q_out = Q.clone()
        Q_out[..., :half] = Q0 * cos_expanded - Q1 * sin_expanded
        Q_out[..., half:] = Q1 * cos_expanded + Q0 * sin_expanded

        ctx.save_for_backward(cos, sin)
        ctx.seq_len = seq_len
        return Q_out

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        seq_len = ctx.seq_len
        half = dY.shape[-1] // 2

        cos_expanded = cos[:seq_len, :half].unsqueeze(0).unsqueeze(2)
        sin_expanded = sin[:seq_len, :half].unsqueeze(0).unsqueeze(2)

        # Backward pass: negate sin
        dY0 = dY[..., :half]
        dY1 = dY[..., half:]
        dQ = dY.clone()
        dQ[..., :half] = dY0 * cos_expanded + dY1 * sin_expanded
        dQ[..., half:] = dY1 * cos_expanded - dY0 * sin_expanded
        return dQ, None, None


class Fast_RoPE_Embedding_QK_M40(torch.autograd.Function):
    """Pure PyTorch RoPE QK embedding for Maxwell GPUs."""

    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        has_indices = rope_indices is not None
        cos, sin = cos.squeeze(), sin.squeeze()

        batch, n_heads_Q, seq_len, head_dim = Q.shape
        _, n_heads_K, _, _ = K.shape
        half = head_dim // 2

        Q_out = Q.clone() if not Q.is_contiguous() else Q.clone()
        K_out = K.clone() if not K.is_contiguous() else K.clone()

        if has_indices:
            # Use rope_indices to select cos/sin rows
            indices = rope_indices.reshape(-1).long()
            cos_sel = cos[indices, :half]  # (batch*seq_len, half)
            sin_sel = sin[indices, :half]
            cos_sel = cos_sel.view(batch, seq_len, 1, half).transpose(1, 2)  # (batch, 1, seq_len, half)
            sin_sel = sin_sel.view(batch, seq_len, 1, half).transpose(1, 2)
        else:
            cos_sel = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, half)
            sin_sel = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)

        # Apply to Q (clone halves to avoid in-place aliasing)
        Q0 = Q_out[..., :half].clone()
        Q1 = Q_out[..., half:].clone()
        Q_out[..., :half] = Q0 * cos_sel - Q1 * sin_sel
        Q_out[..., half:] = Q1 * cos_sel + Q0 * sin_sel

        # Apply to K (clone halves to avoid in-place aliasing)
        K0 = K_out[..., :half].clone()
        K1 = K_out[..., half:].clone()
        K_out[..., :half] = K0 * cos_sel - K1 * sin_sel
        K_out[..., half:] = K1 * cos_sel + K0 * sin_sel

        ctx.save_for_backward(cos, sin)
        ctx.has_indices = has_indices
        ctx.rope_indices = rope_indices
        ctx.seq_len = seq_len
        ctx.n_heads_Q = n_heads_Q
        ctx.n_heads_K = n_heads_K
        return Q_out, K_out

    @staticmethod
    def backward(ctx, dQ, dK):
        cos, sin = ctx.saved_tensors
        batch = dQ.shape[0]
        head_dim = dQ.shape[-1]
        half = head_dim // 2
        seq_len = ctx.seq_len

        if ctx.has_indices:
            indices = ctx.rope_indices.reshape(-1).long()
            cos_sel = cos[indices, :half].view(batch, seq_len, 1, half).transpose(1, 2)
            sin_sel = sin[indices, :half].view(batch, seq_len, 1, half).transpose(1, 2)
        else:
            cos_sel = cos[:seq_len, :half].unsqueeze(0).unsqueeze(0)
            sin_sel = sin[:seq_len, :half].unsqueeze(0).unsqueeze(0)

        # Backward: negate sin (use original dQ/dK slices, not from _out)
        dQ_out = dQ.clone()
        dQ0 = dQ[..., :half]
        dQ1 = dQ[..., half:]
        dQ_out[..., :half] = dQ0 * cos_sel + dQ1 * sin_sel
        dQ_out[..., half:] = dQ1 * cos_sel - dQ0 * sin_sel

        dK_out = dK.clone()
        dK0 = dK[..., :half]
        dK1 = dK[..., half:]
        dK_out[..., :half] = dK0 * cos_sel + dK1 * sin_sel
        dK_out[..., half:] = dK1 * cos_sel - dK0 * sin_sel

        return dQ_out, dK_out, None, None, None
