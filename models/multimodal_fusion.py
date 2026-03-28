"""
Multimodal Fusion Models for Depression Prediction
Combine T1 MRI (ResNet features) + rs-fMRI (GNN features)

Fusion strategies:
- concat: Direct concatenation
- gated: Learnable per-sample modality weighting
- attn: Multi-head self-attention over modality tokens
- cross_attn: Bidirectional cross-attention
- ot: Sinkhorn optimal transport for soft ROI correspondence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ot_fusion import SinkhornOT
from .roi_pooling import ROIPooling3D
from .contrastive import ContrastiveModule


class MultimodalFusion(nn.Module):
    """Full multimodal fusion of T1 (ResNet) + rs-fMRI (GNN) with OT/contrastive support.

    Args:
        gnn: GNN backbone module
        t1: T1 3D CNN backbone module
        fusion_hidden: Fusion hidden dimension
        fusion_dropout: Dropout rate
        fusion_type: 'concat', 'gated', 'attn', 'attn_cls', 'cross_attn', 'cross_attn_uni', 'ot'
        modality_dropout: Probability of dropping a modality during training
        atlas_path: Path to Schaefer200 atlas (required for OT and contrastive)
        use_contrastive: Enable contrastive learning
        contrastive_tau: Temperature for InfoNCE loss
    """

    def __init__(self, gnn, t1, fusion_hidden=256, fusion_dropout=0.5,
                 fusion_type="concat", modality_dropout=0.1,
                 ot_eps=0.1, ot_iters=30, ot_proj_dim=128,
                 ot_row_normalize=False, num_rois=200, atlas_path=None,
                 ot_t1_layer="layer2", use_contrastive=False,
                 contrastive_tau=0.07, contrastive_queue=False,
                 contrastive_queue_size=256, contrastive_mode="unsupervised",
                 diagonal_weight=2.0):
        super().__init__()
        self.gnn = gnn
        self.t1 = t1
        self.fusion_type = fusion_type.lower()
        self.modality_dropout = modality_dropout
        self.num_rois = num_rois
        self.ot_t1_layer = ot_t1_layer
        self.use_contrastive = use_contrastive

        self.ln_gnn = nn.LayerNorm(gnn.out_dim)
        self.ln_t1 = nn.LayerNorm(t1.out_dim)

        # Contrastive learning module
        if use_contrastive:
            if atlas_path is None:
                raise ValueError("atlas_path required for contrastive learning")
            if not hasattr(self, 'roi_pooling') or self.roi_pooling is None:
                self.roi_pooling = ROIPooling3D(atlas_path=atlas_path, num_rois=num_rois)

            self.contrastive_t1_layer = ot_t1_layer
            self.contrastive_t1_roi_dim = 128 if ot_t1_layer == "layer2" else 256

            self.contrastive = ContrastiveModule(
                struct_dim=self.contrastive_t1_roi_dim,
                func_dim=gnn.out_dim,
                embed_dim=128, tau=contrastive_tau,
                use_queue=contrastive_queue, queue_size=contrastive_queue_size,
                mode=contrastive_mode, diagonal_weight=diagonal_weight
            )
        else:
            self.contrastive = None
            self.contrastive_t1_layer = None
            self.contrastive_t1_roi_dim = None

        # Projections
        self.proj_gnn = nn.Linear(gnn.out_dim, fusion_hidden)
        self.proj_t1 = nn.Linear(t1.out_dim, fusion_hidden)
        self.ln_proj_gnn = nn.LayerNorm(fusion_hidden)
        self.ln_proj_t1 = nn.LayerNorm(fusion_hidden)

        # Fusion-specific layers
        if self.fusion_type == "ot":
            if atlas_path is None:
                raise ValueError("atlas_path required for OT fusion")
            self.roi_pooling = ROIPooling3D(atlas_path=atlas_path, num_rois=num_rois)
            self.t1_roi_dim = 128 if ot_t1_layer == "layer2" else 256
            self.sinkhorn_ot = SinkhornOT(
                struct_dim=self.t1_roi_dim, func_dim=gnn.out_dim,
                proj_dim=ot_proj_dim, eps=ot_eps, n_iters=ot_iters,
                row_normalize=ot_row_normalize
            )
            self.ot_fusion_mlp = nn.Sequential(
                nn.Linear(self.t1_roi_dim + gnn.out_dim, fusion_hidden),
                nn.LayerNorm(fusion_hidden), nn.ReLU(), nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, fusion_hidden),
                nn.LayerNorm(fusion_hidden), nn.ReLU()
            )
            fused_dim = fusion_hidden
        elif self.fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(fusion_hidden * 2, fusion_hidden), nn.Sigmoid()
            )
            fused_dim = fusion_hidden
        elif self.fusion_type == "attn":
            self.attn = nn.MultiheadAttention(
                embed_dim=fusion_hidden, num_heads=4, batch_first=False
            )
            self.attn_weight = nn.Parameter(torch.tensor([0.3, 0.7]))
            fused_dim = fusion_hidden
        elif self.fusion_type == "attn_cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_hidden))
            self.attn_cls = nn.MultiheadAttention(
                embed_dim=fusion_hidden, num_heads=4, batch_first=False
            )
            self.ln_attn = nn.LayerNorm(fusion_hidden)
            fused_dim = fusion_hidden
        elif self.fusion_type == "cross_attn":
            self.cross_attn_g2t = nn.MultiheadAttention(
                embed_dim=fusion_hidden, num_heads=4, batch_first=True
            )
            self.cross_attn_t2g = nn.MultiheadAttention(
                embed_dim=fusion_hidden, num_heads=4, batch_first=True
            )
            self.ln_cross = nn.LayerNorm(fusion_hidden * 2)
            fused_dim = fusion_hidden * 2
        elif self.fusion_type == "cross_attn_uni":
            self.cross_attn_uni = nn.MultiheadAttention(
                embed_dim=fusion_hidden, num_heads=4, batch_first=True
            )
            self.ln_cross_uni = nn.LayerNorm(fusion_hidden)
            fused_dim = fusion_hidden
        else:  # concat
            fused_dim = fusion_hidden * 2

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.ELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ELU(),
            nn.Dropout(fusion_dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, data, return_contrastive=False, contrastive_labels=None):
        gnn_emb = self.gnn(data)
        t1_emb = self.t1(data.t1)

        contrastive_loss = None
        if self.use_contrastive and self.contrastive is not None:
            t1_feat_map = self.t1.forward_feature_map(data.t1, layer=self.contrastive_t1_layer)
            t1_roi_tokens, _ = self.roi_pooling(t1_feat_map)
            gnn_nodes = self.gnn.forward_nodes_batched(data, self.num_rois)
            _, _, contrastive_loss = self.contrastive(t1_roi_tokens, gnn_nodes, contrastive_labels)

        gnn_emb = self.ln_gnn(gnn_emb)
        t1_emb = self.ln_t1(t1_emb)
        gnn_proj = self.ln_proj_gnn(self.proj_gnn(gnn_emb))
        t1_proj = self.ln_proj_t1(self.proj_t1(t1_emb))
        gnn_proj = F.normalize(gnn_proj, p=2, dim=-1)
        t1_proj = F.normalize(t1_proj, p=2, dim=-1)

        # Modality dropout
        if self.training and self.modality_dropout > 0:
            B = gnn_proj.size(0)
            drop_mask = torch.rand(B, 1, device=gnn_proj.device)
            gnn_proj = torch.where(drop_mask < self.modality_dropout / 2,
                                   torch.zeros_like(gnn_proj), gnn_proj)
            t1_proj = torch.where((drop_mask >= self.modality_dropout / 2) &
                                  (drop_mask < self.modality_dropout),
                                  torch.zeros_like(t1_proj), t1_proj)

        # Fusion
        if self.fusion_type == "ot":
            t1_feat_map = self.t1.forward_feature_map(data.t1, layer=self.ot_t1_layer)
            S, roi_valid_mask = self.roi_pooling(t1_feat_map)
            F_roi = self.gnn.forward_nodes_batched(data, self.num_rois)
            P, F_tilde = self.sinkhorn_ot(S, F_roi)
            fused_roi = torch.cat([S, F_tilde], dim=-1)
            H = self.ot_fusion_mlp(fused_roi)
            mask_expanded = roi_valid_mask.unsqueeze(-1).float()
            valid_counts = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
            fused = (H * mask_expanded).sum(dim=1) / valid_counts.squeeze(-1)
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([gnn_proj, t1_proj], dim=1))
            fused = gate * t1_proj + (1 - gate) * gnn_proj
        elif self.fusion_type == "attn":
            tokens = torch.stack([gnn_proj, t1_proj], dim=0)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            weights = torch.softmax(self.attn_weight, dim=0)
            fused = weights[0] * attn_out[0] + weights[1] * attn_out[1]
        elif self.fusion_type == "attn_cls":
            B = gnn_proj.size(0)
            cls = self.cls_token.expand(-1, B, -1)
            tokens = torch.cat([cls, gnn_proj.unsqueeze(0), t1_proj.unsqueeze(0)], dim=0)
            attn_out, _ = self.attn_cls(tokens, tokens, tokens)
            fused = self.ln_attn(attn_out[0])
        elif self.fusion_type == "cross_attn":
            gnn_q = gnn_proj.unsqueeze(1)
            t1_q = t1_proj.unsqueeze(1)
            g2t, _ = self.cross_attn_g2t(gnn_q, t1_q, t1_q)
            t2g, _ = self.cross_attn_t2g(t1_q, gnn_q, gnn_q)
            fused = self.ln_cross(torch.cat([g2t.squeeze(1), t2g.squeeze(1)], dim=1))
        elif self.fusion_type == "cross_attn_uni":
            gnn_q = gnn_proj.unsqueeze(1)
            t1_kv = t1_proj.unsqueeze(1)
            out, _ = self.cross_attn_uni(gnn_q, t1_kv, t1_kv)
            fused = self.ln_cross_uni(out.squeeze(1))
        else:  # concat
            fused = torch.cat([gnn_proj, t1_proj], dim=1)

        logits = self.classifier(fused).squeeze(-1)

        if return_contrastive:
            return logits, contrastive_loss
        return logits

    def get_fused_embedding(self, data):
        """Extract fused embedding BEFORE classifier (for t-SNE/UMAP visualization)."""
        gnn_emb = self.gnn(data)
        t1_emb = self.t1(data.t1)

        gnn_emb = self.ln_gnn(gnn_emb)
        t1_emb = self.ln_t1(t1_emb)

        gnn_proj = self.ln_proj_gnn(self.proj_gnn(gnn_emb))
        t1_proj = self.ln_proj_t1(self.proj_t1(t1_emb))

        gnn_proj = F.normalize(gnn_proj, p=2, dim=-1)
        t1_proj = F.normalize(t1_proj, p=2, dim=-1)

        if self.fusion_type == "ot":
            t1_feat_map = self.t1.forward_feature_map(data.t1, layer=self.ot_t1_layer)
            S, roi_valid_mask = self.roi_pooling(t1_feat_map)
            F_roi = self.gnn.forward_nodes_batched(data, self.num_rois)
            P, F_tilde = self.sinkhorn_ot(S, F_roi)
            fused_roi = torch.cat([S, F_tilde], dim=-1)
            H = self.ot_fusion_mlp(fused_roi)
            mask_expanded = roi_valid_mask.unsqueeze(-1).float()
            valid_counts = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
            fused = (H * mask_expanded).sum(dim=1) / valid_counts.squeeze(-1)
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([gnn_proj, t1_proj], dim=1))
            fused = gate * t1_proj + (1 - gate) * gnn_proj
        elif self.fusion_type == "attn":
            tokens = torch.stack([gnn_proj, t1_proj], dim=0)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            weights = torch.softmax(self.attn_weight, dim=0)
            fused = weights[0] * attn_out[0] + weights[1] * attn_out[1]
        else:
            fused = torch.cat([gnn_proj, t1_proj], dim=1)

        return fused

    def get_modality_embeddings(self, data):
        """Extract separate T1 and fMRI embeddings."""
        gnn_emb = self.gnn(data)
        t1_emb = self.t1(data.t1)
        gnn_emb = self.ln_gnn(gnn_emb)
        t1_emb = self.ln_t1(t1_emb)
        gnn_proj = self.ln_proj_gnn(self.proj_gnn(gnn_emb))
        t1_proj = self.ln_proj_t1(self.proj_t1(t1_emb))
        return t1_proj, gnn_proj

    def get_contrastive_embeddings(self, data):
        """Extract Stage A contrastive embeddings (z_s, z_f)."""
        if not self.use_contrastive or self.contrastive is None:
            raise ValueError("Contrastive module not enabled")
        t1_feat_map = self.t1.forward_feature_map(data.t1, layer=self.contrastive_t1_layer)
        t1_roi_tokens, _ = self.roi_pooling(t1_feat_map)
        gnn_nodes = self.gnn.forward_nodes_batched(data, self.num_rois)
        z_s, z_f, _ = self.contrastive(t1_roi_tokens, gnn_nodes)
        return z_s, z_f


class SingleModalityModel(nn.Module):
    """Single modality model (T1 only or fMRI only) for baseline comparison."""

    def __init__(self, backbone, modality: str, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.modality = modality.lower()
        self.hidden = hidden

        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone.out_dim),
            nn.Linear(backbone.out_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        if self.modality == "t1":
            emb = self.backbone(data.t1)
        else:  # fmri
            emb = self.backbone(data)
        return self.classifier(emb).squeeze(-1)
