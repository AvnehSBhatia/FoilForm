"""Geometry sequence → Cl, Cd predictor (autoregressive + variable length).

  • Appended conditioning uses ``tuple_to_embed([Cl, Cd, AoA°])``; AoA is **ground truth**
    when not using full ``teacher_tuples`` (see ``decode_append``).
  • Pool sequence: **last** token → (B, 8).
  • Head: h @ W_{8×8} + b_8, h @ W_{8×2}, h @ W_{2×2} + b_2 → [Cl, Cd].
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8
N_LAYERS = 4
DEFAULT_DROPOUT = 0.05


class AttentionPairwiseBlock(nn.Module):
    """One block: attention + pairwise + MLP + second pairwise (all residual).

    Pre-LayerNorm and dropout on each branch (default dropout ``DEFAULT_DROPOUT``).
    """

    def __init__(self, d: int = D_MODEL, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.d = d
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.norm4 = nn.LayerNorm(d)
        self.W_q = nn.Parameter(torch.empty(d, d))
        self.W_k = nn.Parameter(torch.empty(d, d))
        self.W_v = nn.Parameter(torch.empty(d, d))
        # Pairwise (before MLP): left-multiply each d×d outer product
        self.W_p1 = nn.Parameter(torch.empty(d, d))
        self.B_p1 = nn.Parameter(torch.empty(d, d))
        self.W_p2 = nn.Parameter(torch.empty(d, d))
        self.B_p2 = nn.Parameter(torch.empty(d, d))
        self.mlp_w1 = nn.Parameter(torch.empty(16, d))
        self.mlp_b1 = nn.Parameter(torch.empty(16))
        self.mlp_w2 = nn.Parameter(torch.empty(d, 16))
        self.mlp_b2 = nn.Parameter(torch.empty(d))
        # Pairwise (after MLP)
        self.W_p3 = nn.Parameter(torch.empty(d, d))
        self.B_p3 = nn.Parameter(torch.empty(d, d))
        self.W_p4 = nn.Parameter(torch.empty(d, d))
        self.B_p4 = nn.Parameter(torch.empty(d, d))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        d = self.d
        for p in (self.W_q, self.W_k, self.W_v, self.W_p1, self.W_p2, self.W_p3, self.W_p4):
            nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.B_p1)
        nn.init.zeros_(self.B_p2)
        nn.init.zeros_(self.B_p3)
        nn.init.zeros_(self.B_p4)
        nn.init.xavier_uniform_(self.mlp_w1)
        nn.init.xavier_uniform_(self.mlp_w2)
        nn.init.zeros_(self.mlp_b1)
        nn.init.zeros_(self.mlp_b2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, L, d)
        b, l, d = x.shape
        scale = 1.0 / math.sqrt(d)
        xa = self.norm1(x)
        x_attn = torch.tanh(xa)
        q = torch.matmul(x_attn, self.W_q)
        k = torch.matmul(x_attn, self.W_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
        attn = F.softmax(scores, dim=-1)
        v = torch.matmul(x_attn, self.W_v)
        attn_out = torch.matmul(attn, v)
        x = x + self.dropout(attn_out)

        # Outer product per token: (B, L, d, d)
        xp = self.norm2(x)
        x_pair = torch.tanh(xp)
        outer = x_pair.unsqueeze(-1) * x_pair.unsqueeze(-2)
        # T1 = W1 @ M + B1  — einsum 'df,bife->bide'
        t1 = torch.einsum("df,bife->bide", self.W_p1, outer) + self.B_p1
        t1 = F.leaky_relu(t1)
        t2 = torch.einsum("df,bife->bide", self.W_p2, t1) + self.B_p2
        t2 = F.leaky_relu(t2)
        # Mean across columns of each 8×8 → (B, L, d) (more stable than sum)
        pooled = t2.mean(dim=-1)
        x = x + self.dropout(pooled)

        xm = self.norm3(x)
        h = F.leaky_relu(F.linear(xm, self.mlp_w1, self.mlp_b1))
        x = x + self.dropout(F.linear(h, self.mlp_w2, self.mlp_b2))

        xq = self.norm4(x)
        x_pair2 = torch.tanh(xq)
        outer2 = x_pair2.unsqueeze(-1) * x_pair2.unsqueeze(-2)
        t1b = torch.einsum("df,bife->bide", self.W_p3, outer2) + self.B_p3
        t1b = F.leaky_relu(t1b)
        t2b = torch.einsum("df,bife->bide", self.W_p4, t1b) + self.B_p4
        t2b = F.leaky_relu(t2b)
        pooled2 = t2b.mean(dim=-1)
        x = x + self.dropout(pooled2)
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with KV cache for incremental autoregressive decoding.

        When *kv_cache* is ``None`` the full sequence is processed with a
        causal mask (initial pass).  When a ``(K, V)`` tuple is supplied,
        only the new token(s) in *x* are processed against the cached keys
        and values (incremental pass).

        Returns ``(x_out, (K_all, V_all))``.
        """
        b, l, d = x.shape
        scale = 1.0 / math.sqrt(d)

        xa = self.norm1(x)
        x_attn = torch.tanh(xa)
        q = torch.matmul(x_attn, self.W_q)
        k_new = torch.matmul(x_attn, self.W_k)
        v_new = torch.matmul(x_attn, self.W_v)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k_new], dim=1)
            v = torch.cat([kv_cache[1], v_new], dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        else:
            k, v = k_new, v_new
            mask = causal_mask(l, device=x.device, dtype=x.dtype)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale + mask

        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        x = x + self.dropout(attn_out)

        xp = self.norm2(x)
        x_pair = torch.tanh(xp)
        outer = x_pair.unsqueeze(-1) * x_pair.unsqueeze(-2)
        t1 = torch.einsum("df,bife->bide", self.W_p1, outer) + self.B_p1
        t1 = F.leaky_relu(t1)
        t2 = torch.einsum("df,bife->bide", self.W_p2, t1) + self.B_p2
        t2 = F.leaky_relu(t2)
        x = x + self.dropout(t2.mean(dim=-1))

        xm = self.norm3(x)
        h = F.leaky_relu(F.linear(xm, self.mlp_w1, self.mlp_b1))
        x = x + self.dropout(F.linear(h, self.mlp_w2, self.mlp_b2))

        xq = self.norm4(x)
        x_pair2 = torch.tanh(xq)
        outer2 = x_pair2.unsqueeze(-1) * x_pair2.unsqueeze(-2)
        t1b = torch.einsum("df,bife->bide", self.W_p3, outer2) + self.B_p3
        t1b = F.leaky_relu(t1b)
        t2b = torch.einsum("df,bife->bide", self.W_p4, t1b) + self.B_p4
        t2b = F.leaky_relu(t2b)
        x = x + self.dropout(t2b.mean(dim=-1))
        return x, (k, v)


def causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(L, L) additive causal mask with -inf above diagonal."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    return torch.triu(
        torch.full((length, length), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )


def shift_aero_for_next_input(
    aero: torch.Tensor,
    seq_len: int,
    *,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(B,8) or (B,L,8) -> (B,L,8) with column 0 zero, column i = aero[:,i-1]."""
    if aero.dim() == 2:
        if aero.shape[0] != batch or aero.shape[1] != D_MODEL:
            raise ValueError(f"aero (B,8) expected ({batch}, {D_MODEL}), got {tuple(aero.shape)}")
        seq = aero.unsqueeze(1).expand(batch, seq_len, D_MODEL).contiguous()
    elif aero.dim() == 3:
        if aero.shape[0] != batch or aero.shape[1] != seq_len or aero.shape[2] != D_MODEL:
            raise ValueError(
                f"aero (B,L,8) expected ({batch}, {seq_len}, {D_MODEL}), got {tuple(aero.shape)}"
            )
        seq = aero
    else:
        raise ValueError(f"aero must be (B,8) or (B,L,8), got dim {aero.dim()}")

    z = torch.zeros(batch, 1, D_MODEL, device=device, dtype=dtype)
    return torch.cat([z, seq[:, :-1].clone()], dim=1)


class GeomPolarTransformer(nn.Module):
    """L×8 geometry tokens (+ shifted aero tuple embedding) -> [Cl, Cd]."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        # [Cl, Cd, AoA°] → 8-D appended token (AoA from data when decoding without full teacher).
        self.tuple_to_embed = nn.Linear(3, d_model)
        nn.init.xavier_uniform_(self.tuple_to_embed.weight)
        nn.init.zeros_(self.tuple_to_embed.bias)
        self.in_proj = nn.Linear(2 * d_model, d_model)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        self.norm_in = nn.LayerNorm(d_model)
        self.drop_in = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [AttentionPairwiseBlock(d_model, dropout=dropout) for _ in range(n_layers)]
        )
        self.head_w8 = nn.Parameter(torch.empty(d_model, d_model))
        self.head_b8 = nn.Parameter(torch.empty(d_model))
        self.head_w82 = nn.Parameter(torch.empty(d_model, 2))
        self.head_w22 = nn.Parameter(torch.empty(2, 2))
        self.head_b2 = nn.Parameter(torch.empty(2))
        nn.init.xavier_uniform_(self.head_w8)
        nn.init.zeros_(self.head_b8)
        nn.init.xavier_uniform_(self.head_w82)
        nn.init.xavier_uniform_(self.head_w22)
        nn.init.zeros_(self.head_b2)

    def _forward_with_aero_embed(
        self, geom: torch.Tensor, aero_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        geom: (B, L, 8) geometry-token embeddings.
        aero_embed: (B, 8) or (B, L, 8) condition embeddings; right-shifted so
              patch i uses aero_embed[:, i-1].
              If None, aero_shift is all zeros (no conditioning).
        returns: (B, 2) with [Cl, Cd].
        """
        x = geom
        b, l, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"geom expected (B, L, {self.d_model}), got {tuple(x.shape)}"
            )
        if aero_embed is None:
            aero_s = torch.zeros(b, l, d, device=x.device, dtype=x.dtype)
        else:
            aero_s = shift_aero_for_next_input(
                aero_embed.to(device=x.device, dtype=x.dtype),
                l,
                batch=b,
                device=x.device,
                dtype=x.dtype,
            )
        x = self.drop_in(self.norm_in(self.in_proj(torch.cat([geom, aero_s], dim=-1))))

        mask = causal_mask(l, device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        h = x[:, -1, :]
        h = torch.matmul(h, self.head_w8) + self.head_b8
        h = torch.matmul(h, self.head_w82)
        h = torch.matmul(h, self.head_w22) + self.head_b2
        return h

    def forward(
        self,
        geom: torch.Tensor,
        aero: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-rolling forward pass.
        geom: (B, L, 8)
        aero: optional precomputed 8-D condition embeddings (B,8) or (B,L,8).
        """
        return self._forward_with_aero_embed(geom, aero_embed=aero)

    def forward_autoregressive(
        self,
        geom: torch.Tensor,
        start_tuple: Optional[torch.Tensor] = None,
        teacher_tuples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout over sequence length L.

        At step i we predict y_i=[Cl,Cd]. The next tuple uses
        ``[pred_Cl, pred_Cd, aoa]`` where ``aoa`` is ``teacher_tuples[:,i,2]``
        if teacher is provided, else 0.

        Args:
            geom: (B, L, 8) geometry tokens.
            start_tuple: optional (B, 3) seed used as tuple at step 0.
            teacher_tuples: optional (B, L, 3) — AoA used from column 2 for the chain.
        Returns:
            (B, L, 2) predictions for each step.
        """
        b, l, d = geom.shape
        if d != self.d_model:
            raise ValueError(f"geom expected last dim {self.d_model}, got {d}")
        if teacher_tuples is not None and teacher_tuples.shape != (b, l, 3):
            raise ValueError(
                f"teacher_tuples expected {(b, l, 3)}, got {tuple(teacher_tuples.shape)}"
            )
        if start_tuple is not None and start_tuple.shape != (b, 3):
            raise ValueError(f"start_tuple expected {(b, 3)}, got {tuple(start_tuple.shape)}")

        aero_tokens = torch.zeros(b, l, d, device=geom.device, dtype=geom.dtype)
        if start_tuple is not None:
            aero_tokens[:, 0, :] = self.tuple_to_embed(start_tuple.to(geom.dtype))

        outs = []
        for t in range(l):
            y_t = self._forward_with_aero_embed(
                geom[:, : t + 1, :], aero_embed=aero_tokens[:, : t + 1, :]
            )
            outs.append(y_t.unsqueeze(1))
            if t + 1 < l:
                if teacher_tuples is not None:
                    aoa = teacher_tuples[:, t, 2]
                else:
                    aoa = torch.zeros(b, device=geom.device, dtype=geom.dtype)
                next_tuple = torch.stack([y_t[:, 0], y_t[:, 1], aoa], dim=-1)
                aero_tokens[:, t + 1, :] = self.tuple_to_embed(next_tuple.to(geom.dtype))
        return torch.cat(outs, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def decode_append(
        self,
        geom_context: torch.Tensor,
        target_steps: int,
        teacher_tuples: Optional[torch.Tensor] = None,
        *,
        aoa_ground_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive decoding by appending tuple embeddings as new tokens.

        After each step, the appended token is ``tuple_to_embed(nxt)`` where:

        - If ``teacher_tuples`` is set: ``nxt = teacher_tuples[:, t]`` (full teacher).
        - Else: ``nxt = [pred_Cl, pred_Cd, aoa_ground_truth[:, t]]`` (AoA in degrees).

        ``aoa_ground_truth`` is required when ``teacher_tuples`` is None.

        Args:
            geom_context: (B, Lg, 8) starting geometry tokens (e.g., Lg=167).
            target_steps: number of tuple outputs to generate.
            teacher_tuples: optional (B, target_steps, 3) full teacher-forcing tuples.
            aoa_ground_truth: (B, target_steps) AoA (°) per step when not using full teacher.

        Returns:
            preds: (B, target_steps, 2) [Cl, Cd]
        """
        if target_steps <= 0:
            raise ValueError(f"target_steps must be positive, got {target_steps}")
        b, _, d = geom_context.shape
        if d != self.d_model:
            raise ValueError(f"geom_context expected last dim {self.d_model}, got {d}")
        if teacher_tuples is not None and teacher_tuples.shape != (b, target_steps, 3):
            raise ValueError(
                f"teacher_tuples expected {(b, target_steps, 3)}, got {tuple(teacher_tuples.shape)}"
            )
        if teacher_tuples is None:
            if aoa_ground_truth is None or aoa_ground_truth.shape != (b, target_steps):
                raise ValueError(
                    f"aoa_ground_truth must be (B, target_steps)={(b, target_steps)}, "
                    f"got {None if aoa_ground_truth is None else tuple(aoa_ground_truth.shape)}"
                )

        # -- Step 0: full forward on geometry tokens, initialise KV caches -
        aero_s = torch.zeros_like(geom_context)
        x = self.drop_in(
            self.norm_in(self.in_proj(torch.cat([geom_context, aero_s], dim=-1)))
        )
        n_blocks = len(self.blocks)
        kv_caches: list[Tuple[torch.Tensor, torch.Tensor]] = [(torch.empty(0), torch.empty(0))] * n_blocks
        for i, blk in enumerate(self.blocks):
            x, kv_caches[i] = blk.forward_cached(x, kv_cache=None)

        h = x[:, -1, :]
        h = torch.matmul(h, self.head_w8) + self.head_b8
        h = torch.matmul(h, self.head_w82)
        y_t = torch.matmul(h, self.head_w22) + self.head_b2
        outs = [y_t.unsqueeze(1)]

        # -- Steps 1 .. target_steps-1: incremental decode with cache ---
        for t in range(1, target_steps):
            if teacher_tuples is not None:
                nxt = teacher_tuples[:, t - 1]
            else:
                nxt = torch.stack(
                    [y_t[:, 0], y_t[:, 1], aoa_ground_truth[:, t - 1].to(y_t.dtype)],
                    dim=-1,
                )
            nxt_tok = self.tuple_to_embed(nxt.to(geom_context.dtype)).unsqueeze(1)
            aero_s_new = torch.zeros(
                b, 1, d, device=geom_context.device, dtype=geom_context.dtype
            )
            x_new = self.drop_in(
                self.norm_in(self.in_proj(torch.cat([nxt_tok, aero_s_new], dim=-1)))
            )
            for i, blk in enumerate(self.blocks):
                x_new, kv_caches[i] = blk.forward_cached(x_new, kv_cache=kv_caches[i])

            h = x_new[:, 0, :]
            h = torch.matmul(h, self.head_w8) + self.head_b8
            h = torch.matmul(h, self.head_w82)
            y_t = torch.matmul(h, self.head_w22) + self.head_b2
            outs.append(y_t.unsqueeze(1))

        return torch.cat(outs, dim=1)


def parameter_breakdown() -> Tuple[int, dict[str, int]]:
    """Return (total, per-component counts) matching this file's definitions."""
    m = GeomPolarTransformer()
    total = m.count_parameters()
    d = D_MODEL
    per_block = 0
    b0 = m.blocks[0]
    # Q, K, V
    per_block += b0.W_q.numel() + b0.W_k.numel() + b0.W_v.numel()
    # Pairwise (×2)
    per_block += b0.W_p1.numel() + b0.B_p1.numel() + b0.W_p2.numel() + b0.B_p2.numel()
    per_block += b0.W_p3.numel() + b0.B_p3.numel() + b0.W_p4.numel() + b0.B_p4.numel()
    # MLP 8→16→8
    per_block += b0.mlp_w1.numel() + b0.mlp_b1.numel() + b0.mlp_w2.numel() + b0.mlp_b2.numel()
    # Pre-LN per branch (norm1..norm4)
    per_block += 4 * (2 * d)

    head = (
        m.head_w8.numel()
        + m.head_b8.numel()
        + m.head_w82.numel()
        + m.head_w22.numel()
        + m.head_b2.numel()
    )
    breakdown = {
        "per_block_attention_qkv": b0.W_q.numel() + b0.W_k.numel() + b0.W_v.numel(),
        "per_block_pairwise": b0.W_p1.numel()
        + b0.B_p1.numel()
        + b0.W_p2.numel()
        + b0.B_p2.numel()
        + b0.W_p3.numel()
        + b0.B_p3.numel()
        + b0.W_p4.numel()
        + b0.B_p4.numel(),
        "per_block_mlp": b0.mlp_w1.numel()
        + b0.mlp_b1.numel()
        + b0.mlp_w2.numel()
        + b0.mlp_b2.numel(),
        "tuple_to_embed": m.tuple_to_embed.weight.numel() + m.tuple_to_embed.bias.numel(),
        "in_proj": m.in_proj.weight.numel() + m.in_proj.bias.numel(),
        "norm_in": m.norm_in.weight.numel() + m.norm_in.bias.numel(),
        "per_block_total": per_block,
        "all_blocks": per_block * N_LAYERS,
        "output_head": head,
        "grand_total": total,
    }
    return total, breakdown


if __name__ == "__main__":
    t, bd = parameter_breakdown()
    print("GeomPolarTransformer parameter breakdown:")
    for k, v in bd.items():
        print(f"  {k}: {v}")
    print(f"\nTotal trainable parameters: {t}")
    x = torch.randn(2, 5, D_MODEL)
    m = GeomPolarTransformer()
    y = m(x)
    y_ar = m.forward_autoregressive(x)
    print(f"Forward check: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"Autoregressive check: input {tuple(x.shape)} -> output {tuple(y_ar.shape)}")
    z = torch.randn(2, 4, D_MODEL)
    dec = m.decode_append(z, target_steps=3, aoa_ground_truth=torch.zeros(2, 3))
    print(f"decode_append (pred+gt AoA): {tuple(dec.shape)}")
