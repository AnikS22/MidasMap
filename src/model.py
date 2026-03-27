"""
CenterNet with CEM500K-pretrained ResNet-50 backbone for immunogold detection.

Architecture:
    Input:   1ch grayscale, variable size (padded to multiple of 32)
    Encoder: CEM500K ResNet-50 (pretrained), conv1 adapted for 1ch input
    Neck:    BiFPN (2 rounds, 128ch)
    Decoder: Transposed conv → stride-2 output
    Heads:   Heatmap (2ch sigmoid), Offset (2ch)
    Output:  Stride-2 maps → (H/2, W/2) resolution

Output stride is 2, NOT 4 or 8. At stride 4, a 6nm bead (4-6px radius)
collapses to 1px in feature space — insufficient for detection.
At stride 2, same bead occupies 2-3px, enough for Gaussian peak extraction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional


# ---------------------------------------------------------------------------
# BiFPN: Bidirectional Feature Pyramid Network
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution as used in BiFPN."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride=stride,
            padding=padding, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class BiFPNFusionNode(nn.Module):
    """
    Single BiFPN fusion node with fast normalized weighted fusion.

    w_normalized = relu(w) / (sum(relu(w)) + eps)
    output = conv(sum(w_i * input_i))
    """

    def __init__(self, channels: int, n_inputs: int = 2, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # Learnable fusion weights
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.conv = DepthwiseSeparableConv(channels, channels)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Fast normalized fusion
        w = F.relu(self.weights)
        w_norm = w / (w.sum() + self.eps)

        fused = sum(w_i * inp for w_i, inp in zip(w_norm, inputs))
        return self.conv(fused)


class BiFPNLayer(nn.Module):
    """
    One round of BiFPN: top-down + bottom-up bidirectional fusion.

    Input levels: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)
    """

    def __init__(self, channels: int):
        super().__init__()
        # Top-down fusion nodes (P5 → P4_td, P4_td+P3 → P3_td, P3_td+P2 → P2_td)
        self.td_p4 = BiFPNFusionNode(channels, n_inputs=2)
        self.td_p3 = BiFPNFusionNode(channels, n_inputs=2)
        self.td_p2 = BiFPNFusionNode(channels, n_inputs=2)

        # Bottom-up fusion nodes (combine top-down outputs with original)
        self.bu_p3 = BiFPNFusionNode(channels, n_inputs=3)  # p3_orig + p3_td + p2_out
        self.bu_p4 = BiFPNFusionNode(channels, n_inputs=3)  # p4_orig + p4_td + p3_out
        self.bu_p5 = BiFPNFusionNode(channels, n_inputs=2)  # p5_orig + p4_out

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P2, P3, P4, P5] at channels ch, with decreasing spatial dims

        Returns:
            [P2_out, P3_out, P4_out, P5_out]
        """
        p2, p3, p4, p5 = features

        # --- Top-down pathway ---
        # P5 → upscale → fuse with P4
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        p4_td = self.td_p4([p4, p5_up])

        # P4_td → upscale → fuse with P3
        p4_td_up = F.interpolate(p4_td, size=p3.shape[2:], mode="nearest")
        p3_td = self.td_p3([p3, p4_td_up])

        # P3_td → upscale → fuse with P2
        p3_td_up = F.interpolate(p3_td, size=p2.shape[2:], mode="nearest")
        p2_td = self.td_p2([p2, p3_td_up])

        # --- Bottom-up pathway ---
        p2_out = p2_td

        # P2_out → downsample → fuse with P3_td and P3_orig
        p2_down = F.max_pool2d(p2_out, kernel_size=2)
        p3_out = self.bu_p3([p3, p3_td, p2_down])

        # P3_out → downsample → fuse with P4_td and P4_orig
        p3_down = F.max_pool2d(p3_out, kernel_size=2)
        p4_out = self.bu_p4([p4, p4_td, p3_down])

        # P4_out → downsample → fuse with P5_orig
        p4_down = F.max_pool2d(p4_out, kernel_size=2)
        p5_out = self.bu_p5([p5, p4_down])

        return [p2_out, p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """Multi-round BiFPN with lateral projections."""

    def __init__(self, in_channels: List[int], out_channels: int = 128,
                 num_rounds: int = 2):
        super().__init__()
        # Lateral 1x1 projections to unify channel count
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels
        ])

        # BiFPN rounds
        self.rounds = nn.ModuleList([
            BiFPNLayer(out_channels) for _ in range(num_rounds)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Project to uniform channels
        projected = [lat(feat) for lat, feat in zip(self.laterals, features)]

        # Run BiFPN rounds
        for bifpn_round in self.rounds:
            projected = bifpn_round(projected)

        return projected


# ---------------------------------------------------------------------------
# Detection Heads
# ---------------------------------------------------------------------------

class HeatmapHead(nn.Module):
    """Heatmap prediction head at stride-2 resolution."""

    def __init__(self, in_channels: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # Initialize final conv bias for focal loss: -log((1-pi)/pi) where pi=0.01
        # This prevents the network from producing high false positive rate early
        nn.init.constant_(self.conv2.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return torch.sigmoid(self.conv2(x))


class OffsetHead(nn.Module):
    """Sub-pixel offset regression head."""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1)  # dx, dy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


# ---------------------------------------------------------------------------
# Full CenterNet Model
# ---------------------------------------------------------------------------

class ImmunogoldCenterNet(nn.Module):
    """
    CenterNet with CEM500K-pretrained ResNet-50 backbone.

    Detects 6nm and 12nm immunogold particles at stride-2 resolution.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        bifpn_channels: int = 128,
        bifpn_rounds: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # --- Encoder: ResNet-50 ---
        backbone = models.resnet50(weights=None)
        # Adapt conv1 for 1-channel grayscale input
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )

        # Load pretrained weights
        if pretrained_path:
            self._load_pretrained(backbone, pretrained_path)
        else:
            # Use ImageNet weights as fallback, adapting conv1
            imagenet_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            state = imagenet_backbone.state_dict()
            # Mean-pool RGB conv1 weights → grayscale
            state["conv1.weight"] = state["conv1.weight"].mean(dim=1, keepdim=True)
            backbone.load_state_dict(state, strict=False)

        # Extract encoder stages
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 256ch, stride 4
        self.layer2 = backbone.layer2  # 512ch, stride 8
        self.layer3 = backbone.layer3  # 1024ch, stride 16
        self.layer4 = backbone.layer4  # 2048ch, stride 32

        # --- BiFPN Neck ---
        self.bifpn = BiFPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=bifpn_channels,
            num_rounds=bifpn_rounds,
        )

        # --- Decoder: upsample P2 (stride 4) → stride 2 ---
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                bifpn_channels, 64, kernel_size=4, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- Detection Heads (at stride-2 resolution) ---
        self.heatmap_head = HeatmapHead(64, num_classes)
        self.offset_head = OffsetHead(64)

    def _load_pretrained(self, backbone: nn.Module, path: str):
        """Load CEM500K MoCoV2 pretrained weights."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        state = {}
        # CEM500K uses MoCo format: keys prefixed with 'module.encoder_q.'
        src_state = ckpt.get("state_dict", ckpt)
        for k, v in src_state.items():
            # Strip MoCo prefix
            new_key = k
            for prefix in ["module.encoder_q.", "module.", "encoder_q."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            state[new_key] = v

        # Adapt conv1: mean-pool 3ch RGB → 1ch grayscale
        if "conv1.weight" in state and state["conv1.weight"].shape[1] == 3:
            state["conv1.weight"] = state["conv1.weight"].mean(dim=1, keepdim=True)

        # Load with strict=False (head layers won't match)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        # Expected: fc.weight, fc.bias will be missing/unexpected
        print(f"CEM500K loaded: {len(state)} keys, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, 1, H, W) grayscale input

        Returns:
            heatmap: (B, 2, H/2, W/2) sigmoid-activated class heatmaps
            offsets: (B, 2, H/2, W/2) sub-pixel offset predictions
        """
        # Encoder
        x0 = self.stem(x)        # stride 4
        p2 = self.layer1(x0)     # 256ch, stride 4
        p3 = self.layer2(p2)     # 512ch, stride 8
        p4 = self.layer3(p3)     # 1024ch, stride 16
        p5 = self.layer4(p4)     # 2048ch, stride 32

        # BiFPN neck
        features = self.bifpn([p2, p3, p4, p5])

        # Decoder: upsample P2 to stride 2
        x_up = self.upsample(features[0])

        # Detection heads
        heatmap = self.heatmap_head(x_up)   # (B, 2, H/2, W/2)
        offsets = self.offset_head(x_up)    # (B, 2, H/2, W/2)

        return heatmap, offsets

    def freeze_encoder(self):
        """Freeze entire encoder (Phase 1 training)."""
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_deep_layers(self):
        """Unfreeze layer3 and layer4 (Phase 2 training)."""
        for module in [self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all layers (Phase 3 training)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self, phase: int, cfg: dict) -> list:
        """
        Get parameter groups with discriminative learning rates per phase.

        Args:
            phase: 1, 2, or 3
            cfg: training phase config from config.yaml

        Returns:
            List of param group dicts for optimizer.
        """
        if phase == 1:
            # Only neck + heads trainable
            return [
                {"params": self.bifpn.parameters(), "lr": cfg["lr"]},
                {"params": self.upsample.parameters(), "lr": cfg["lr"]},
                {"params": self.heatmap_head.parameters(), "lr": cfg["lr"]},
                {"params": self.offset_head.parameters(), "lr": cfg["lr"]},
            ]
        elif phase == 2:
            return [
                {"params": self.stem.parameters(), "lr": 0},
                {"params": self.layer1.parameters(), "lr": 0},
                {"params": self.layer2.parameters(), "lr": 0},
                {"params": self.layer3.parameters(), "lr": cfg["lr_layer3"]},
                {"params": self.layer4.parameters(), "lr": cfg["lr_layer4"]},
                {"params": self.bifpn.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.upsample.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.heatmap_head.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.offset_head.parameters(), "lr": cfg["lr_decoder"]},
            ]
        else:  # phase 3
            return [
                {"params": self.stem.parameters(), "lr": cfg["lr_stem"]},
                {"params": self.layer1.parameters(), "lr": cfg["lr_layer1"]},
                {"params": self.layer2.parameters(), "lr": cfg["lr_layer2"]},
                {"params": self.layer3.parameters(), "lr": cfg["lr_layer3"]},
                {"params": self.layer4.parameters(), "lr": cfg["lr_layer4"]},
                {"params": self.bifpn.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.upsample.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.heatmap_head.parameters(), "lr": cfg["lr_decoder"]},
                {"params": self.offset_head.parameters(), "lr": cfg["lr_decoder"]},
            ]
