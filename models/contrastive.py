"""
CLIP-style Cross-Modal Contrastive Learning for Multimodal Brain Imaging

Implements InfoNCE loss with optional queue memory for small batch sizes.
Aligns T1 structural and fMRI functional representations in a shared embedding space.

Reference: Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


class AttentionPooling(nn.Module):
    """Learnable attention-based pooling over ROIs."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - N ROI tokens of dimension D
        Returns:
            (B, D) - attention-weighted sum over ROIs
        """
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled


class QueueMemory(nn.Module):
    """Queue-based memory bank for additional negatives."""

    def __init__(self, embed_dim: int = 128, queue_size: int = 256):
        super().__init__()
        self.queue_size = queue_size
        self.embed_dim = embed_dim
        self.register_buffer('queue_s', F.normalize(torch.randn(queue_size, embed_dim), dim=1))
        self.register_buffer('queue_f', F.normalize(torch.randn(queue_size, embed_dim), dim=1))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, z_s: torch.Tensor, z_f: torch.Tensor):
        batch_size = z_s.shape[0]
        ptr = int(self.ptr)
        if ptr + batch_size > self.queue_size:
            first_part = self.queue_size - ptr
            self.queue_s[ptr:] = z_s[:first_part].detach()
            self.queue_f[ptr:] = z_f[:first_part].detach()
            self.queue_s[:batch_size - first_part] = z_s[first_part:].detach()
            self.queue_f[:batch_size - first_part] = z_f[first_part:].detach()
            self.ptr[0] = batch_size - first_part
        else:
            self.queue_s[ptr:ptr + batch_size] = z_s.detach()
            self.queue_f[ptr:ptr + batch_size] = z_f.detach()
            self.ptr[0] = (ptr + batch_size) % self.queue_size

    def get_negatives(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.queue_s.clone(), self.queue_f.clone()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss (symmetric CLIP-style).

    Supports unsupervised, supervised, and hybrid modes.

    Args:
        tau: Temperature parameter
        use_queue: Whether to use queue memory for additional negatives
        queue_size: Size of queue memory
        embed_dim: Embedding dimension (for queue)
        mode: 'unsupervised', 'supervised', or 'hybrid'
        diagonal_weight: Weight for diagonal (self-match) in supervised/hybrid mode
    """

    def __init__(self, tau: float = 0.07, use_queue: bool = False,
                 queue_size: int = 256, embed_dim: int = 128,
                 mode: str = "unsupervised", diagonal_weight: float = 2.0):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.mode = mode.lower()
        self.diagonal_weight = diagonal_weight

        if self.mode not in ["unsupervised", "supervised", "hybrid"]:
            raise ValueError(f"mode must be 'unsupervised', 'supervised', or 'hybrid', got {mode}")

        if use_queue:
            self.queue = QueueMemory(embed_dim=embed_dim, queue_size=queue_size)
        else:
            self.queue = None

    def forward(self, z_s: torch.Tensor, z_f: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "unsupervised" or labels is None:
            return self._unsupervised_forward(z_s, z_f)
        elif self.mode == "supervised":
            return self._supervised_forward(z_s, z_f, labels)
        elif self.mode == "hybrid":
            return self._hybrid_supervised_forward(z_s, z_f, labels)
        else:
            return self._unsupervised_forward(z_s, z_f)

    def _unsupervised_forward(self, z_s: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        B = z_s.shape[0]
        device = z_s.device

        if self.use_queue and self.queue is not None:
            queue_s, queue_f = self.queue.get_negatives()
            queue_s, queue_f = queue_s.to(device), queue_f.to(device)
            all_f = torch.cat([z_f, queue_f], dim=0)
            logits_s2f = z_s @ all_f.T / self.tau
            all_s = torch.cat([z_s, queue_s], dim=0)
            logits_f2s = z_f @ all_s.T / self.tau
            self.queue.enqueue(z_s, z_f)
        else:
            logits_s2f = z_s @ z_f.T / self.tau
            logits_f2s = z_f @ z_s.T / self.tau

        targets = torch.arange(B, device=device)
        loss_s2f = F.cross_entropy(logits_s2f, targets)
        loss_f2s = F.cross_entropy(logits_f2s, targets)
        return (loss_s2f + loss_f2s) / 2

    def _supervised_forward(self, z_s, z_f, labels):
        B = z_s.shape[0]
        device = z_s.device

        labels = labels.view(-1, 1)
        same_class_mask = (labels == labels.T).float()
        diag_mask = torch.eye(B, device=device)
        weighted_mask = same_class_mask + (self.diagonal_weight - 1.0) * diag_mask
        row_sums = weighted_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weighted_mask = weighted_mask / row_sums

        logits_s2f = z_s @ z_f.T / self.tau
        logits_f2s = z_f @ z_s.T / self.tau

        logits_s2f_max = logits_s2f.max(dim=1, keepdim=True)[0].detach()
        logits_f2s_max = logits_f2s.max(dim=1, keepdim=True)[0].detach()

        exp_s2f = torch.exp(logits_s2f - logits_s2f_max)
        exp_f2s = torch.exp(logits_f2s - logits_f2s_max)

        denom_s2f = exp_s2f.sum(dim=1, keepdim=True)
        denom_f2s = exp_f2s.sum(dim=1, keepdim=True)

        log_prob_s2f = logits_s2f - logits_s2f_max - torch.log(denom_s2f + 1e-8)
        log_prob_f2s = logits_f2s - logits_f2s_max - torch.log(denom_f2s + 1e-8)

        loss_s2f = -(log_prob_s2f * weighted_mask).sum(dim=1)
        loss_f2s = -(log_prob_f2s * weighted_mask).sum(dim=1)

        return (loss_s2f.mean() + loss_f2s.mean()) / 2

    def _hybrid_supervised_forward(self, z_s, z_f, labels):
        B = z_s.shape[0]
        device = z_s.device

        labels = labels.view(-1, 1)
        same_class_mask = (labels == labels.T).float()
        diag_mask = torch.eye(B, device=device)
        hybrid_mask = same_class_mask.clone()
        hybrid_mask = hybrid_mask + (self.diagonal_weight - 1.0) * diag_mask
        row_sums = hybrid_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        hybrid_mask = hybrid_mask / row_sums

        logits_s2f = z_s @ z_f.T / self.tau
        logits_f2s = z_f @ z_s.T / self.tau

        logits_s2f_max = logits_s2f.max(dim=1, keepdim=True)[0].detach()
        logits_f2s_max = logits_f2s.max(dim=1, keepdim=True)[0].detach()

        exp_s2f = torch.exp(logits_s2f - logits_s2f_max)
        exp_f2s = torch.exp(logits_f2s - logits_f2s_max)

        denom_s2f = exp_s2f.sum(dim=1, keepdim=True)
        denom_f2s = exp_f2s.sum(dim=1, keepdim=True)

        log_prob_s2f = logits_s2f - logits_s2f_max - torch.log(denom_s2f + 1e-8)
        log_prob_f2s = logits_f2s - logits_f2s_max - torch.log(denom_f2s + 1e-8)

        loss_s2f = -(log_prob_s2f * hybrid_mask).sum(dim=1)
        loss_f2s = -(log_prob_f2s * hybrid_mask).sum(dim=1)

        return (loss_s2f.mean() + loss_f2s.mean()) / 2


class ContrastiveModule(nn.Module):
    """Complete contrastive learning module for multimodal fusion.

    Args:
        struct_dim: Structural (T1) feature dimension
        func_dim: Functional (fMRI) feature dimension
        embed_dim: Shared embedding dimension
        tau: Temperature for InfoNCE
        use_queue: Use queue memory for small batches
        queue_size: Queue size if using queue
        mode: 'unsupervised', 'supervised', or 'hybrid'
        diagonal_weight: Weight for diagonal in hybrid mode
    """

    def __init__(self, struct_dim: int, func_dim: int, embed_dim: int = 128,
                 tau: float = 0.07, use_queue: bool = False, queue_size: int = 256,
                 mode: str = "unsupervised", diagonal_weight: float = 2.0):
        super().__init__()
        self.pool_s = AttentionPooling(struct_dim)
        self.pool_f = AttentionPooling(func_dim)
        self.proj_s = ProjectionHead(struct_dim, embed_dim, embed_dim)
        self.proj_f = ProjectionHead(func_dim, embed_dim, embed_dim)
        self.loss_fn = InfoNCELoss(
            tau=tau, use_queue=use_queue, queue_size=queue_size,
            embed_dim=embed_dim, mode=mode, diagonal_weight=diagonal_weight
        )
        self.embed_dim = embed_dim
        self.mode = mode

    def forward(self, s_tokens: torch.Tensor, f_tokens: torch.Tensor,
                labels: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_pool = self.pool_s(s_tokens)
        f_pool = self.pool_f(f_tokens)
        z_s = self.proj_s(s_pool)
        z_f = self.proj_f(f_pool)
        loss = self.loss_fn(z_s, z_f, labels)
        return z_s, z_f, loss

    def get_embeddings(self, s_tokens: torch.Tensor,
                       f_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_pool = self.pool_s(s_tokens)
        f_pool = self.pool_f(f_tokens)
        z_s = self.proj_s(s_pool)
        z_f = self.proj_f(f_pool)
        return z_s, z_f
