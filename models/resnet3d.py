"""
3D CNN Backbones for T1 MRI Feature Extraction

Includes:
- MedicalNetResNet18: 3D ResNet-18 compatible with MedicalNet pretrained weights
- Simple3DCNN: Vanilla 3D CNN baseline (no pretrained weights)
"""

import os

import torch
import torch.nn as nn


class MedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class MedicalNetResNet18(nn.Module):
    """3D ResNet-18 compatible with MedicalNet pretrained weights.

    Args:
        dropout: Dropout rate before final output.
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(MedBasicBlock, 64, 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(MedBasicBlock, 128, 2, stride=2, dilation=1)
        self.layer3 = self._make_layer(MedBasicBlock, 256, 2, stride=1, dilation=2)
        self.layer4 = self._make_layer(MedBasicBlock, 512, 2, stride=1, dilation=4)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 512
        self._reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                           dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_bn(self):
        """Freeze BatchNorm layers - keep running stats fixed during training."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last(self):
        for param in self.layer4.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for name, param in self.named_parameters():
            if 'bn' not in name.lower():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.shape[1] != 1:
            x = x[:, None, ...]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return x

    def forward_feature_map(self, x: torch.Tensor, layer: str = 'layer2') -> torch.Tensor:
        """Extract intermediate feature map for ROI pooling.

        Args:
            x: Input tensor (B, 1, D, H, W)
            layer: Which layer to extract from ('layer2' or 'layer3')

        Returns:
            Feature map tensor (B, C, D', H', W')
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.shape[1] != 1:
            x = x[:, None, ...]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        if layer == 'layer2':
            return x

        x = self.layer3(x)

        if layer == 'layer3':
            return x

        raise ValueError(f"layer must be 'layer2' or 'layer3', got {layer}")


class Simple3DCNN(nn.Module):
    """Vanilla 3D CNN without pretrained weights (baseline)."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 256
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() == 5 and x.shape[1] != 1:
            x = x[:, None, ...]
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return x


def load_medicalnet_pretrained(model: nn.Module, ckpt_path: str):
    """Load MedicalNet pretrained weights."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"MedicalNet checkpoint not found: {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    filtered = {k: v for k, v in cleaned.items()
                if k in model_state and v.shape == model_state[k].shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded MedicalNet weights: {len(filtered)} layers, "
          f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
