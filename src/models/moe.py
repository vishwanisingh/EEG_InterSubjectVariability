"""Mixture-of-Experts modules (SimpleMoE and sparse top-k MoE)."""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


@dataclass
class MoEConfig:
    d_in: int
    d_hidden: int
    d_out: int
    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25
    router_temp: float = 1.0
    router_noise_std: float = 0.0
    expert_dropout: float = 0.0
    lb_loss_weight: float = 1e-2
    entropy_bonus: float = 0.0
    z_loss_weight: float = 0.0
    use_fp32_router: bool = True


class SimpleMoE(nn.Module):
    """Dense (soft) Mixture of Experts — every expert runs on every token."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, dropout=dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(x), dim=-1)                    # (B, E)
        outputs = torch.stack([e(x) for e in self.experts], dim=1)   # (B, E, d_out)
        return torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)   # (B, d_out)


class _TopKRouter(nn.Module):
    """Softmax top-k gating router."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.w_gate = nn.Linear(config.d_in, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor, training: bool):
        x_in = x.to(torch.float32) if self.config.use_fp32_router else x
        scores = self.w_gate(x_in)
        if training and self.config.router_noise_std > 0:
            scores = scores + torch.randn_like(scores) * self.config.router_noise_std
        probs = F.softmax(scores / self.config.router_temp, dim=-1)
        topk_p, topk_i = torch.topk(probs, k=self.config.top_k, dim=-1)
        return topk_p.to(x.dtype), topk_i, scores.to(x.dtype), probs.to(x.dtype)


class MoE(nn.Module):
    """Sparse top-k Mixture of Experts with load-balancing auxiliary losses."""

    def __init__(self, config: MoEConfig, experts: Optional[List[nn.Module]] = None):
        super().__init__()
        self.config = config
        self.router = _TopKRouter(config)
        if experts is None:
            experts = [MLP(config.d_in, config.d_hidden, config.d_out, dropout=0.1)
                       for _ in range(config.num_experts)]
        self.experts = nn.ModuleList(experts)

    def _capacity(self, n_tokens: int) -> int:
        return max(1, math.ceil(self.config.capacity_factor * n_tokens / len(self.experts)))

    def forward(self, x: torch.Tensor):
        orig   = x.shape
        d_in   = orig[-1]
        d_out  = self.config.d_out
        x_flat = x.reshape(-1, d_in)
        N      = x_flat.size(0)
        E      = len(self.experts)
        K      = self.config.top_k
        device, dtype = x.device, x.dtype

        topk_p, topk_i, scores, full_probs = self.router(x_flat, self.training)

        # Auxiliary losses
        mean_p    = full_probs.mean(0)
        lb        = -(mean_p * (mean_p + 1e-9).log()).sum() / math.log(E)
        token_ent = -(full_probs * (full_probs + 1e-9).log()).sum(-1) / math.log(E)
        ent_bonus = token_ent.mean()
        z_loss    = scores.float().pow(2).mean().to(dtype)

        # Dispatch with per-expert capacity
        cap        = self._capacity(N)
        expert_in  = [torch.zeros(cap, d_in,  dtype=dtype, device=device) for _ in range(E)]
        expert_gate= [torch.zeros(cap, 1,     dtype=dtype, device=device) for _ in range(E)]
        counters   = torch.zeros(E, dtype=torch.int32, device=device)
        positions  = []

        for k in range(K):
            ei    = topk_i[:, k]
            pos_k = torch.full((N,), -1, dtype=torch.int32, device=device)
            for e in range(E):
                idx   = (ei == e).nonzero(as_tuple=False).flatten()
                n     = idx.numel()
                if n == 0:
                    continue
                start = counters[e].item()
                end   = min(start + n, cap)
                take  = max(0, end - start)
                if take > 0:
                    chosen       = idx[:take]
                    pos_k[chosen]= torch.arange(start, start + take, device=device, dtype=torch.int32)
                    counters[e]  = start + take
            positions.append(pos_k)

        accepted = []
        for k in range(K):
            ei  = topk_i[:, k]
            pos = positions[k]
            m   = (pos >= 0).nonzero(as_tuple=False).flatten()
            accepted.append(m)
            if m.numel() == 0:
                continue
            e_m = ei[m]; p_m = pos[m].long()
            for e in range(E):
                sel = (e_m == e).nonzero(as_tuple=False).flatten()
                if sel.numel() == 0:
                    continue
                expert_in[e][p_m[sel]]   = x_flat[m[sel]]
                expert_gate[e][p_m[sel]] = topk_p[m[sel], k].unsqueeze(-1)

        # Run experts
        expert_out = []
        for e in range(E):
            out_e = self.experts[e](expert_in[e])
            if self.training and self.config.expert_dropout > 0 and torch.rand(1, device=device) < self.config.expert_dropout:
                out_e = torch.zeros_like(out_e)
            expert_out.append(out_e * expert_gate[e])

        # Combine
        y_flat = torch.zeros((N, d_out), dtype=dtype, device=device)
        for k in range(K):
            ei  = topk_i[:, k]; pos = positions[k]
            m   = accepted[k]
            if m.numel() == 0:
                continue
            e_m = ei[m]; p_m = pos[m].long()
            buf = torch.empty((m.numel(), d_out), dtype=dtype, device=device)
            for e in range(E):
                sel = (e_m == e).nonzero(as_tuple=False).flatten()
                if sel.numel() == 0:
                    continue
                buf[sel] = expert_out[e][p_m[sel]]
            y_flat[m] += buf

        y   = y_flat.reshape(*orig[:-1], d_out)
        aux = {
            "load_balance":    lb,
            "entropy_bonus":   ent_bonus,
            "z_loss":          z_loss,
            "expert_mean_usage": mean_p.detach(),
        }
        return y, aux