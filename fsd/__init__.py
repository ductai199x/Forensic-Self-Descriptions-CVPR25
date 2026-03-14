"""FSD: Detecting AI-Generated Images via Forensic Self-Descriptions.

Paper: https://arxiv.org/abs/2503.21003 (CVPR 2025)

Quick start:
    from fsd import FSDDetector

    detector = FSDDetector.load()  # auto-downloads weights on first use
    result = detector.score("photo.jpg")
    print(result.z_score, result.is_fake)
"""

__version__ = "1.0.0"

from .detector import FSDDetector, DetectionResult
from .weights import download_weights, get_weights_dir

__all__ = ["FSDDetector", "DetectionResult", "download_weights", "get_weights_dir", "__version__"]
