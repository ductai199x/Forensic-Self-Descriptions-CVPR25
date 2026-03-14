"""Source attribution for AI-generated images.

Given an image already detected as AI-generated, identifies which generator
produced it by scoring against per-source GMMs.

Usage:
    from fsd import FSDDetector

    detector = FSDDetector.load(attribution=True)
    result = detector.attribute("fake_photo.jpg")
    print(result.source, result.confidence)
"""

import torch

from dataclasses import dataclass

from .gmm import TorchGMM


@dataclass
class AttributionResult:
    """Result of source attribution for a single image.

    Attributes:
        source: Predicted source name (e.g., "DALL-E 3", "Stable Diffusion XL").
        confidence: Confidence score (normalized probability of the best source).
        scores: Dict mapping each source name to its log-likelihood.
        z_score: Detection z-score (from the detection pipeline).
        is_fake: Whether the image was detected as fake.
    """

    source: str
    confidence: float
    scores: dict[str, float]
    z_score: float
    is_fake: bool

    def __repr__(self):
        return (
            f"AttributionResult(source={self.source!r}, confidence={self.confidence:.3f}, "
            f"z_score={self.z_score:.4f}, is_fake={self.is_fake})"
        )


def load_source_gmms(path, device="cpu"):
    """Load packed per-source GMMs from a single weights file.

    Args:
        path: Path to the .pt file containing all source GMMs.
        device: Device to load onto.

    Returns:
        Dict mapping source name to TorchGMM instance.
    """
    data = torch.load(path, map_location="cpu", weights_only=True)
    source_gmms = {}

    for source in data["sources"]:
        name = source["name"]
        gmm = TorchGMM(
            n_components=int(source["n_components"]),
            covariance_type=source["covariance_type"],
            device="cpu",
        )
        gmm.means_ = source["means_"].to(dtype=torch.float64)
        gmm.weights_ = source["weights_"].to(dtype=torch.float64)
        gmm.covariances_ = source["covariances_"].to(dtype=torch.float64)
        gmm.precisions_cholesky_ = source["precisions_cholesky_"].to(dtype=torch.float64)
        gmm._update_inference_cache()

        if device != "cpu":
            gmm.to(device)

        source_gmms[name] = gmm

    return source_gmms


def classify(fsd_vec, source_gmms):
    """Classify a single FSD vector by scoring against all source GMMs.

    Args:
        fsd_vec: (1, D) projected FSD tensor.
        source_gmms: Dict mapping source name to TorchGMM.

    Returns:
        (source_name, confidence, scores_dict) where:
        - source_name: name of the best-matching source
        - confidence: softmax probability of the best source
        - scores_dict: dict mapping each source name to its log-likelihood
    """
    names = list(source_gmms.keys())
    log_liks = {}
    for name, gmm in source_gmms.items():
        log_liks[name] = gmm.score_samples(fsd_vec).item()

    # Softmax over log-likelihoods for confidence
    ll_tensor = torch.tensor([log_liks[n] for n in names], dtype=torch.float64)
    probs = torch.softmax(ll_tensor, dim=0)
    best_idx = probs.argmax().item()

    return names[best_idx], probs[best_idx].item(), log_liks
