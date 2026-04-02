"""
Optimal Transport (Sinkhorn) based ROI Soft Correspondence Fusion

Combines structural T1 ROI tokens with functional rs-fMRI ROI tokens using
differentiable Sinkhorn optimal transport to learn soft correspondence.

Reference: Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from typing import Tuple, Dict


class SinkhornOT(nn.Module):
    """Sinkhorn Optimal Transport for ROI correspondence.

    Args:
        struct_dim: Dimension of structural ROI tokens
        func_dim: Dimension of functional ROI tokens
        proj_dim: Common projection dimension for cost computation
        eps: Entropic regularization (temperature)
        n_iters: Number of Sinkhorn iterations
        row_normalize: If True, row-normalize P for stability
    """

    def __init__(self, struct_dim: int, func_dim: int, proj_dim: int = 128,
                 eps: float = 0.1, n_iters: int = 30, row_normalize: bool = False):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters
        self.row_normalize = row_normalize
        self.proj_dim = proj_dim

        self.proj_struct = nn.Sequential(
            nn.Linear(struct_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.proj_func = nn.Sequential(
            nn.Linear(func_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def compute_cost_matrix(self, S: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        S_proj = self.proj_struct(S)
        F_proj = self.proj_func(F)
        S_norm = Fn.normalize(S_proj, p=2, dim=-1)
        F_norm = Fn.normalize(F_proj, p=2, dim=-1)
        cosine_sim = torch.bmm(S_norm, F_norm.transpose(1, 2))
        C = torch.clamp(1.0 - cosine_sim, min=0.0, max=2.0)
        return C

    def sinkhorn(self, C: torch.Tensor, n_rows: int = 200,
                 n_cols: int = 200) -> torch.Tensor:
        B = C.shape[0]
        device = C.device
        dtype = C.dtype

        K = torch.exp(-C / self.eps)
        r = torch.ones(B, n_rows, device=device, dtype=dtype) / n_rows
        c = torch.ones(B, n_cols, device=device, dtype=dtype) / n_cols
        u = torch.ones(B, n_rows, device=device, dtype=dtype)
        v = torch.ones(B, n_cols, device=device, dtype=dtype)

        for _ in range(self.n_iters):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
            u = r / (Kv + 1e-8)
            Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
            v = c / (Ktu + 1e-8)
            u = torch.clamp(u, min=1e-8, max=1e8)
            v = torch.clamp(v, min=1e-8, max=1e8)

        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        return P

    def forward(self, S: torch.Tensor, F: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = S.shape
        C = self.compute_cost_matrix(S, F)
        P = self.sinkhorn(C, n_rows=N, n_cols=N)

        if self.row_normalize:
            P_sum = P.sum(dim=-1, keepdim=True)
            P = P / (P_sum + 1e-8)

        F_tilde = torch.bmm(P, F)
        return P, F_tilde


class MultimodalOTFusion(nn.Module):
    """Full multimodal fusion pipeline with OT-based ROI correspondence.

    Args:
        struct_dim: Dimension of structural ROI tokens
        func_dim: Dimension of functional ROI tokens
        proj_dim: OT projection dimension
        hidden_dim: Fusion MLP hidden dimension
        num_rois: Number of ROIs
        eps: Sinkhorn regularization
        n_iters: Sinkhorn iterations
        row_normalize: Row-normalize transport plan
        dropout: Dropout rate
        pool_type: 'mean' or 'max' pooling over ROIs
    """

    def __init__(self, struct_dim: int, func_dim: int, proj_dim: int = 128,
                 hidden_dim: int = 256, num_rois: int = 200, eps: float = 0.1,
                 n_iters: int = 30, row_normalize: bool = False,
                 dropout: float = 0.3, pool_type: str = 'mean'):
        super().__init__()
        self.num_rois = num_rois
        self.pool_type = pool_type

        self.ot = SinkhornOT(
            struct_dim=struct_dim, func_dim=func_dim,
            proj_dim=proj_dim, eps=eps, n_iters=n_iters,
            row_normalize=row_normalize
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(struct_dim + func_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, S: torch.Tensor, F: torch.Tensor,
                return_transport: bool = True) -> Dict[str, torch.Tensor]:
        P, F_tilde = self.ot(S, F)
        fused_input = torch.cat([S, F_tilde], dim=-1)
        H = self.fusion_mlp(fused_input)

        if self.pool_type == 'mean':
            H_pooled = H.mean(dim=1)
        elif self.pool_type == 'max':
            H_pooled = H.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        logits = self.classifier(H_pooled)

        output = {'logits': logits, 'H': H, 'H_pooled': H_pooled, 'F_tilde': F_tilde}
        if return_transport:
            output['P'] = P
        return output
