"""Forensic Self-Description (FSD) computation.

Extracts a fixed-length FSD vector from an image by:
1. Computing forensic residuals via the FRE
2. Multi-scale patch decomposition
3. Constrained least squares solving (KKT system)

The resulting FSD is a compact descriptor of the image's forensic characteristics.
"""

import torch
import torch.nn.functional as F

from PIL import Image


def _solve_constrained_least_squares(
    patches_list,
    mask,
    center,
    K,
    n_features,
    device,
    chunk_size=16384,
    lambda_reg=1e-5,
):
    """Memory-efficient constrained least squares via chunked accumulation.

    Solves for the FSD vector using KKT conditions with sum-to-one constraints
    per channel. Accumulates X^T X and X^T y in chunks to avoid materializing
    the full design matrix.
    """
    total_features = K * n_features

    XTX = torch.zeros(total_features, total_features, dtype=torch.float64, device=device)
    XTy = torch.zeros(total_features, dtype=torch.float64, device=device)

    for scale_patches in patches_list:
        for i in range(0, scale_patches.shape[0], chunk_size):
            chunk = scale_patches[i : i + chunk_size]
            x = chunk[:, :, mask].reshape(-1, total_features).to(torch.float64)
            y = chunk[:, :, center, center].sum(dim=1).to(torch.float64)
            XTX += x.T @ x
            XTy += x.T @ y

    # Regularization for numerical stability
    XTX += lambda_reg * torch.eye(total_features, device=device, dtype=torch.float64)

    # KKT system: sum-to-one constraint per channel
    A = torch.zeros(K, total_features, device=device, dtype=torch.float64)
    for k in range(K):
        A[k, k * n_features : (k + 1) * n_features] = 1
    b = torch.ones(K, device=device, dtype=torch.float64)

    top = torch.cat([XTX, A.T], dim=1)
    bottom = torch.cat([A, torch.zeros(K, K, device=device, dtype=torch.float64)], dim=1)
    LHS = torch.cat([top, bottom], dim=0)
    RHS = torch.cat([XTy, b], dim=0)

    solution = torch.linalg.solve(LHS, RHS)
    return solution[:total_features]


def compute_fsd(
    image,
    fre,
    kernel_size=11,
    num_scales=3,
    max_size=1024,
    resize_mode="resize_and_crop",
):
    """Compute Forensic Self-Description from an image.

    Args:
        image: PIL Image, file path (str/Path), or grayscale tensor (1, H, W).
        fre: FRE model instance.
        kernel_size: Patch size for FSD extraction.
        num_scales: Number of downsampling scales.
        max_size: Maximum dimension after resize/crop.
        resize_mode: One of "resize", "crop", "resize_and_crop".

    Returns:
        FSD vector as a 1D float64 tensor of dimension K * (kernel_size^2 - 1).
    """
    device = fre.device

    # Handle input types
    if isinstance(image, (str,)):
        image = Image.open(image)
    from pathlib import Path as _Path

    if isinstance(image, _Path):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.mode != "L":
            image = image.convert("L")
        image_t = torch.from_numpy(__import__("numpy").array(image)).float().unsqueeze(0)  # (1, H, W)
    elif isinstance(image, torch.Tensor):
        image_t = image.float()
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    K = fre.conv.out_channels
    fre_kernel_size = fre.conv.kernel_size
    border_size = fre_kernel_size // 2
    B = kernel_size
    center = B // 2
    mask = torch.ones(B, B, dtype=torch.bool, device=device)
    mask[center, center] = False
    n_features = mask.sum().item()  # B^2 - 1

    with torch.no_grad():
        # Compute forensic residuals
        residuals = fre(image_t.to(device))
        residuals = residuals[:, border_size:-border_size, border_size:-border_size]

        # Resize and/or crop
        if "resize" in resize_mode:
            h, w = residuals.shape[-2:]
            scale_factor = max_size / min(h, w)
            new_h, new_w = round(h * scale_factor), round(w * scale_factor)
            residuals = F.interpolate(
                residuals[None, ...],
                size=(new_h, new_w),
                mode="bilinear",
                antialias=False,
                align_corners=False,
            )[0]
        if "crop" in resize_mode:
            h, w = residuals.shape[-2:]
            crop_h = max_size if h > max_size else h
            crop_w = max_size if w > max_size else w
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            residuals = residuals[:, start_h : start_h + crop_h, start_w : start_w + crop_w]

        # Multi-scale patch extraction
        patches = []
        for l in range(num_scales):
            scaled = F.interpolate(
                residuals[None, ...],
                scale_factor=1 / 2**l,
                mode="bilinear",
                antialias=False,
                align_corners=False,
            )
            scaled = F.pad(scaled, (B // 2, B // 2, B // 2, B // 2), mode="reflect")
            unfolded = scaled[0].unfold(1, B, 1).unfold(2, B, 1)
            unfolded = unfolded.reshape(K, -1, B, B).permute(1, 0, 2, 3)
            patches.append(unfolded)

        # Solve constrained least squares
        fsd = _solve_constrained_least_squares(patches, mask, center, K, n_features, device)

    return fsd.cpu()
