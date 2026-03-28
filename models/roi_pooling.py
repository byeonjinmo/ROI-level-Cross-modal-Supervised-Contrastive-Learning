"""
ROI-aligned Pooling Module for Structure-Function Coupling
Extracts ROI-specific features from 3D CNN feature maps using Schaefer200 atlas parcellation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

try:
    import nibabel as nib
except ImportError:
    nib = None


class ROIPooling3D(nn.Module):
    """Extract ROI-aligned features from 3D CNN feature maps using atlas parcellation.

    Args:
        atlas_path: Path to Schaefer200 atlas NIfTI file (labels 0=bg, 1-200=ROIs)
        num_rois: Number of ROIs (default: 200 for Schaefer200)
        original_shape: Expected original T1 volume shape (D, H, W)
    """

    def __init__(self, atlas_path: str, num_rois: int = 200,
                 original_shape: Tuple[int, int, int] = (91, 109, 91)):
        super().__init__()
        if nib is None:
            raise ImportError("nibabel is required for ROIPooling3D")

        self.atlas_path = atlas_path
        self.num_rois = num_rois
        self.original_shape = original_shape

        atlas_img = nib.load(atlas_path)
        atlas_data = atlas_img.get_fdata().astype(np.float32)
        self.register_buffer('atlas_original', torch.from_numpy(atlas_data))
        self._atlas_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

        unique_labels = np.unique(atlas_data.astype(int))
        self._valid_labels = set(unique_labels) - {0}

        print(f"[ROIPooling3D] Loaded atlas from {atlas_path}")
        print(f"  Atlas shape: {atlas_data.shape}")
        print(f"  ROI labels: 1-{max(self._valid_labels)} ({len(self._valid_labels)} ROIs)")

    def _resample_atlas(self, target_shape: Tuple[int, int, int],
                        device: torch.device) -> torch.Tensor:
        cache_key = target_shape
        if cache_key in self._atlas_cache:
            cached = self._atlas_cache[cache_key]
            if cached.device == device:
                return cached
            else:
                return cached.to(device)

        atlas = self.atlas_original.to(device).unsqueeze(0).unsqueeze(0)
        resampled = F.interpolate(atlas, size=target_shape, mode='nearest'
                                  ).squeeze(0).squeeze(0)
        self._atlas_cache[cache_key] = resampled
        return resampled

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: (B, C, D, H, W) CNN feature map
        Returns:
            roi_features: (B, num_rois, C)
            roi_valid_mask: (B, num_rois) boolean mask
        """
        B, C, D, H, W = feature_map.shape
        device = feature_map.device

        atlas = self._resample_atlas((D, H, W), device)
        roi_features = torch.zeros(B, self.num_rois, C, device=device, dtype=feature_map.dtype)
        roi_valid_mask = torch.zeros(B, self.num_rois, dtype=torch.bool, device=device)

        feature_flat = feature_map.view(B, C, -1)
        atlas_flat = atlas.view(-1)

        for roi_idx in range(self.num_rois):
            atlas_label = roi_idx + 1
            mask = (atlas_flat == atlas_label)
            voxel_count = mask.sum().item()
            if voxel_count > 0:
                masked_features = feature_flat[:, :, mask]
                roi_features[:, roi_idx, :] = masked_features.mean(dim=2)
                roi_valid_mask[:, roi_idx] = True

        return roi_features, roi_valid_mask

    def get_roi_voxel_counts(self, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        atlas = self._resample_atlas(target_shape, self.atlas_original.device)
        counts = torch.zeros(self.num_rois, device=atlas.device)
        for roi_idx in range(self.num_rois):
            counts[roi_idx] = (atlas == roi_idx + 1).sum().float()
        return counts


