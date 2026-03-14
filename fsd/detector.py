"""High-level FSD detector API.

Usage:
    from fsd import FSDDetector

    detector = FSDDetector.load("weights/")
    result = detector.score("photo.jpg")
    print(result)  # DetectionResult(z_score=-3.5, is_fake=True, ...)

    # With source attribution
    detector = FSDDetector.load("weights/", attribution=True)
    result = detector.attribute("fake_photo.jpg")
    print(result.source, result.confidence)
"""

import json
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .fre import FRE
from .fsd_computation import compute_fsd
from .gmm import load_gmm
from .projection import apply_projections, load_transforms


@dataclass
class DetectionResult:
    """Result of scoring a single image.

    Attributes:
        z_score: Normalized score. More negative = more likely AI-generated.
        raw_score: Raw GMM log-likelihood before normalization.
        is_fake: Whether the z-score falls below the threshold.
        threshold: The z-score threshold used.
    """

    z_score: float
    raw_score: float
    is_fake: bool
    threshold: float

    def __repr__(self):
        label = "FAKE" if self.is_fake else "REAL"
        return f"DetectionResult(z_score={self.z_score:.4f}, is_fake={self.is_fake} [{label}], threshold={self.threshold})"


class FSDDetector:
    """Forensic Self-Description detector for AI-generated images.

    Loads pre-trained weights and provides a simple scoring API.
    """

    def __init__(self, fre, gmm, projections, config, threshold,
                 attribution_projections=None, source_gmms=None):
        self.fre = fre
        self.gmm = gmm
        self.projections = projections
        self.config = config
        self.threshold = threshold
        self.train_mean = config["scoring"]["train_mean"]
        self.train_std = config["scoring"]["train_std"]
        self.attribution_projections = attribution_projections or []
        self.source_gmms = source_gmms

    @classmethod
    def load(cls, weights_dir=None, device="auto", threshold=None, attribution=False):
        """Load pre-trained detector.

        Args:
            weights_dir: Path to directory containing config.json and weight files.
                If None, auto-detects or downloads weights to ~/.cache/fsd/.
            device: Device to load onto. "auto" selects CUDA if available.
            threshold: Z-score threshold for fake detection. If None, uses the
                default from config.json. More negative = stricter.
            attribution: If True, also load source attribution weights.
                Enables the attribute() method.

        Returns:
            FSDDetector instance ready for scoring.
        """
        if weights_dir is None:
            from .weights import get_weights_dir
            weights_dir = get_weights_dir(attribution=attribution)
        weights_dir = Path(weights_dir)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Disable TF32 on Ampere+ GPUs to ensure numerical consistency with CPU.
        # TF32 reduces float32 mantissa to 10 bits, causing divergent FRE residuals
        # that get amplified through the constrained least-squares solver.
        if device != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        # Load config
        with open(weights_dir / "config.json") as f:
            config = json.load(f)

        if threshold is None:
            threshold = config["scoring"]["default_threshold"]

        # Load FRE
        fre = FRE.from_pretrained(weights_dir / config["fre"]["weights_file"], device=device)

        # Load GMM
        gmm = load_gmm(weights_dir / config["gmm"]["weights_file"], device=device)

        # Load learned transforms
        projections = load_transforms(
            weights_dir / config["transforms"]["weights_file"], device=device
        )

        # Load attribution weights if requested
        attribution_projections = None
        source_gmms = None
        if attribution:
            attr_config = config.get("attribution")
            if attr_config is None:
                raise RuntimeError(
                    "Attribution weights not found in config.json. "
                    "Ensure you have the latest weights with attribution support."
                )
            attribution_projections = load_transforms(
                weights_dir / attr_config["weights_file"], device=device
            )
            from .attribution import load_source_gmms
            source_gmms = load_source_gmms(
                weights_dir / attr_config["source_gmms_file"], device=device
            )

        return cls(fre, gmm, projections, config, threshold,
                   attribution_projections, source_gmms)

    def score(self, image) -> DetectionResult:
        """Score a single image.

        Args:
            image: File path (str/Path), PIL Image, or grayscale tensor (1, H, W).

        Returns:
            DetectionResult with z_score, raw_score, is_fake, and threshold.
        """
        fsd_config = self.config["fsd"]
        fsd_vec = compute_fsd(
            image,
            self.fre,
            kernel_size=fsd_config["kernel_size"],
            num_scales=fsd_config["num_scales"],
            max_size=fsd_config["max_size"],
            resize_mode=fsd_config["resize_mode"],
        )

        # Apply learned transforms
        device = self.fre.device
        fsd_vec = fsd_vec.to(device).unsqueeze(0)  # (1, D)
        with torch.no_grad():
            fsd_vec = apply_projections(fsd_vec, self.projections)

        # Score with GMM
        raw_score = self.gmm.score_samples(fsd_vec).item()
        z_score = (raw_score - self.train_mean) / self.train_std

        return DetectionResult(
            z_score=z_score,
            raw_score=raw_score,
            is_fake=z_score < self.threshold,
            threshold=self.threshold,
        )

    def score_batch(self, images, show_progress=True) -> list[DetectionResult]:
        """Score multiple images.

        Args:
            images: Iterable of file paths, PIL Images, or tensors.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of DetectionResult, one per image.
        """
        images = list(images)
        if show_progress:
            try:
                from tqdm import tqdm

                images = tqdm(images, desc="Scoring", unit="img")
            except ImportError:
                pass

        return [self.score(img) for img in images]

    def attribute(self, image):
        """Identify the source of an AI-generated image.

        Requires loading with attribution=True. Runs the full detection pipeline
        plus attribution-specific transforms and per-source GMM scoring.

        Args:
            image: File path (str/Path), PIL Image, or grayscale tensor (1, H, W).

        Returns:
            AttributionResult with source name, confidence, per-source scores,
            z_score, and is_fake.
        """
        if self.source_gmms is None:
            raise RuntimeError(
                "Attribution not loaded. Use FSDDetector.load(attribution=True)."
            )

        from .attribution import AttributionResult, classify

        fsd_config = self.config["fsd"]
        fsd_vec = compute_fsd(
            image,
            self.fre,
            kernel_size=fsd_config["kernel_size"],
            num_scales=fsd_config["num_scales"],
            max_size=fsd_config["max_size"],
            resize_mode=fsd_config["resize_mode"],
        )

        device = self.fre.device
        fsd_vec = fsd_vec.to(device).unsqueeze(0)  # (1, D)

        with torch.no_grad():
            # Detection transforms
            fsd_det = apply_projections(fsd_vec, self.projections)

            # Detection score
            raw_score = self.gmm.score_samples(fsd_det).item()
            z_score = (raw_score - self.train_mean) / self.train_std

            # Attribution transforms (applied on top of detection)
            fsd_attr = apply_projections(fsd_det, self.attribution_projections)

        # Classify source
        source, confidence, scores = classify(fsd_attr, self.source_gmms)

        return AttributionResult(
            source=source,
            confidence=confidence,
            scores=scores,
            z_score=z_score,
            is_fake=z_score < self.threshold,
        )

    def compute_fsd(self, image) -> torch.Tensor:
        """Advanced: compute the raw FSD vector (before scoring).

        Args:
            image: File path (str/Path), PIL Image, or grayscale tensor (1, H, W).

        Returns:
            1D float64 tensor of dimension K * (kernel_size^2 - 1).
        """
        fsd_config = self.config["fsd"]
        return compute_fsd(
            image,
            self.fre,
            kernel_size=fsd_config["kernel_size"],
            num_scales=fsd_config["num_scales"],
            max_size=fsd_config["max_size"],
            resize_mode=fsd_config["resize_mode"],
        )
