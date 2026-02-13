"""Forensic Residual Extractor (FRE).

Constrained convolution that extracts forensic residuals from grayscale images.
The residual is computed as: residual = image - conv(image).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConv2d(nn.Module):
    """Convolution with prediction-error filter constraints.

    Weights are constrained so that each filter sums to zero at the center
    and the off-center elements sum to one, enforcing a prediction-error
    structure.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        w = torch.empty((out_channels, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(w, 1 / 3)
        self.w = nn.Parameter(w)
        one_middle = torch.zeros((kernel_size * kernel_size,))
        one_middle[kernel_size * kernel_size // 2] = 1
        self.one_middle = nn.Parameter(one_middle, requires_grad=False)

    def constrain(self, w: torch.Tensor) -> torch.Tensor:
        w = w.view(-1, self.kernel_size * self.kernel_size)
        w = w - w.mean(1)[..., None] + 1 / (self.kernel_size * self.kernel_size - 1)
        scaling_coeff = (w * (1 - self.one_middle)).sum(1)
        w = w / scaling_coeff[..., None]
        w = w - w * self.one_middle
        w = w.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return w

    @property
    def constrained_w(self):
        return self.constrain(self.w)

    def forward(self, x):
        w = self.constrain(self.w)
        y = F.conv2d(x, w, padding=self.kernel_size // 2)
        return y


class FRE(nn.Module):
    """Forensic Residual Extractor.

    Computes forensic residuals: residual = image - constrained_conv(image).

    Args:
        in_channels: Number of input channels (1 for grayscale).
        out_channels: Number of output residual channels.
        kernel_size: Convolution kernel size.
    """

    def __init__(self, in_channels=1, out_channels=8, kernel_size=15):
        super().__init__()
        self.conv = ConstrainedConv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forensic residuals.

        Args:
            x: (1, H, W) or (B, 1, H, W) grayscale image tensor, values in 0-255.

        Returns:
            Residual tensor of shape (K, H, W) or (B, K, H, W).
        """
        return x - self.conv(x)

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        """Load FRE from a plain state_dict file.

        The config (in_channels, out_channels, kernel_size) is inferred from
        the weight tensor shape.

        Args:
            path: Path to the .pt state_dict file.
            device: Device to load onto.

        Returns:
            FRE model in eval mode.
        """
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Infer config from weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        w = state_dict["w"]
        out_channels, in_channels, kernel_size, _ = w.shape
        fre = cls(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        fre.conv.load_state_dict(state_dict)
        fre = fre.to(device)
        fre.eval()
        return fre

    @property
    def device(self):
        return self.conv.w.device
