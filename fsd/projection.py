"""Learned feature transforms applied to FSD vectors."""

import torch
import torch.nn as nn


class FeatureTransform(nn.Module):
    """Learned feature transform module."""

    def __init__(self, dim=960, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(hidden, dim, dtype=torch.float64),
        )

    def forward(self, x):
        return x + self.net(x)


def apply_projections(x, projections):
    """Apply transforms in sequence. No-op if list is empty."""
    for proj in projections:
        x = proj(x)
    return x


def load_transforms(path, device="cpu"):
    """Load pre-trained transforms from a single weights file.

    Args:
        path: Path to the .pt file.
        device: Device to load onto.

    Returns:
        List of FeatureTransform modules in eval mode.
    """
    data = torch.load(path, map_location=device, weights_only=True)
    config = data["config"]
    transforms = []
    for state_dict in data["transforms"]:
        t = FeatureTransform(dim=config["dim"], hidden=config["hidden"])
        t.load_state_dict(state_dict)
        t = t.to(device).eval()
        transforms.append(t)
    return transforms
