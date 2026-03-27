"""Unit tests for model architecture."""

import pytest
import torch

from src.model import ImmunogoldCenterNet, BiFPN


class TestModelForwardPass:
    def test_output_shapes(self):
        """Verify output shapes match stride-2 specification."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        x = torch.randn(1, 1, 512, 512)
        hm, off = model(x)

        assert hm.shape == (1, 2, 256, 256), f"Expected (1,2,256,256), got {hm.shape}"
        assert off.shape == (1, 2, 256, 256), f"Expected (1,2,256,256), got {off.shape}"

    def test_heatmap_sigmoid_range(self):
        """Heatmap outputs should be in [0, 1] from sigmoid."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        x = torch.randn(1, 1, 512, 512)
        hm, _ = model(x)

        assert hm.min() >= 0.0
        assert hm.max() <= 1.0

    def test_batch_dimension(self):
        """Model should handle batch size > 1."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        x = torch.randn(4, 1, 512, 512)
        hm, off = model(x)

        assert hm.shape[0] == 4
        assert off.shape[0] == 4

    def test_variable_input_size(self):
        """Model should handle different input sizes (multiples of 32)."""
        model = ImmunogoldCenterNet(pretrained_path=None)

        for size in [256, 384, 512]:
            x = torch.randn(1, 1, size, size)
            hm, off = model(x)
            assert hm.shape == (1, 2, size // 2, size // 2)

    def test_parameter_count(self):
        """Model should have approximately 25M parameters."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        n_params = sum(p.numel() for p in model.parameters())
        # ResNet-50 is ~25M, plus BiFPN and heads
        assert 20_000_000 < n_params < 40_000_000


class TestFreezeUnfreeze:
    def test_freeze_encoder(self):
        """Frozen encoder should have no gradients."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        model.freeze_encoder()

        for name, param in model.named_parameters():
            if any(x in name for x in ["stem", "layer1", "layer2", "layer3", "layer4"]):
                assert not param.requires_grad, f"{name} should be frozen"

        # BiFPN and heads should still be trainable
        for name, param in model.bifpn.named_parameters():
            assert param.requires_grad, f"bifpn.{name} should be trainable"

    def test_unfreeze_deep(self):
        """Unfreezing deep layers should enable gradients for layer3/4."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        model.freeze_encoder()
        model.unfreeze_deep_layers()

        for param in model.layer3.parameters():
            assert param.requires_grad
        for param in model.layer4.parameters():
            assert param.requires_grad
        # Stem and layer1/2 still frozen
        for param in model.stem.parameters():
            assert not param.requires_grad

    def test_unfreeze_all(self):
        """Unfreeze all should enable all gradients."""
        model = ImmunogoldCenterNet(pretrained_path=None)
        model.freeze_encoder()
        model.unfreeze_all()

        for param in model.parameters():
            assert param.requires_grad


class TestBiFPN:
    def test_bifpn_output_shapes(self):
        """BiFPN should output 4 feature maps at 128 channels."""
        bifpn = BiFPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=128,
            num_rounds=2,
        )
        features = [
            torch.randn(1, 256, 128, 128),   # P2: stride 4
            torch.randn(1, 512, 64, 64),      # P3: stride 8
            torch.randn(1, 1024, 32, 32),     # P4: stride 16
            torch.randn(1, 2048, 16, 16),     # P5: stride 32
        ]

        outputs = bifpn(features)
        assert len(outputs) == 4
        for i, out in enumerate(outputs):
            assert out.shape[1] == 128, f"P{i+2} channels should be 128"
            assert out.shape[2:] == features[i].shape[2:], \
                f"P{i+2} spatial dims should match input"
