"""Learned feature transforms applied to FSD vectors."""

import torch
import torch.nn as nn


class FeatureTransform(nn.Module):
    """Learned residual feature transform: f(x) = x + MLP(x).

    Two architectures are supported, selected by constructor arguments:
      - Narrow (default): dim → hidden → hidden → dim  (3 linear layers)
      - Wide: dim → h1 → h2 → h3 → h2 → h1 → dim     (6 linear layers)
    """

    def __init__(self, dim=960, hidden=128, *, h1=None, h2=None, h3=None):
        super().__init__()
        if h1 is not None:
            # Wide (hourglass) architecture
            self.net = nn.Sequential(
                nn.Linear(dim, h1, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(h1, h2, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(h2, h3, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(h3, h2, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(h2, h1, dtype=torch.float64),
                nn.GELU(),
                nn.Linear(h1, dim, dtype=torch.float64),
            )
        else:
            # Narrow (default) architecture
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


def _make_transform(config):
    """Instantiate a FeatureTransform from a config dict."""
    if "h1" in config:
        return FeatureTransform(dim=config["dim"], h1=config["h1"], h2=config["h2"], h3=config["h3"])
    return FeatureTransform(dim=config["dim"], hidden=config["hidden"])


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
        t = _make_transform(config)
        t.load_state_dict(state_dict)
        t = t.to(device).eval()
        transforms.append(t)
    return transforms
