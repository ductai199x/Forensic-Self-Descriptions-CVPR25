"""Ray Serve deployment for FSD scoring.

Deploys the full scoring pipeline across multiple GPUs via Ray Serve.
Each replica loads its own copy of the model on a dedicated GPU fraction.

Usage via CLI:
    fsd-score-ray serve --weights-dir ./weights
    fsd-score-ray score --dir ./images/
"""

import json
import logging
import traceback

import torch
from pathlib import Path
from ray import serve
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse

from .fre import FRE
from .fsd_computation import compute_fsd
from .gmm import load_gmm
from .projection import apply_projections, load_transforms


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 1},
    max_ongoing_requests=1,
)
class FSDScorer:
    """Ray Serve deployment that scores images for AI-generated content.

    Accepts JSON requests with either a file path or base64-encoded image.
    """

    def __init__(self, weights_dir: str, threshold: float = -2.0):
        self._logger = logging.getLogger("ray.serve")
        weights_dir = Path(weights_dir)
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load config
        with open(weights_dir / "config.json") as f:
            self.config = json.load(f)

        # Load FRE
        self.fre = FRE.from_pretrained(
            weights_dir / self.config["fre"]["weights_file"], device=self.device
        )

        # Load GMM
        self.gmm = load_gmm(
            weights_dir / self.config["gmm"]["weights_file"], device=self.device
        )
        self.train_mean = self.config["scoring"]["train_mean"]
        self.train_std = self.config["scoring"]["train_std"]

        # Load learned transforms
        self.projections = load_transforms(
            weights_dir / self.config["transforms"]["weights_file"], device=self.device
        )

        self._logger.info(f"FSDScorer ready on {self.device}")

    def _score_image(self, image) -> dict:
        """Score a single image (path string or PIL Image)."""
        fsd_config = self.config["fsd"]
        fsd_vec = compute_fsd(
            image,
            self.fre,
            kernel_size=fsd_config["kernel_size"],
            num_scales=fsd_config["num_scales"],
            max_size=fsd_config["max_size"],
            resize_mode=fsd_config["resize_mode"],
        )

        fsd_vec = fsd_vec.to(self.device).unsqueeze(0)
        with torch.no_grad():
            fsd_vec = apply_projections(fsd_vec, self.projections)

        raw_score = self.gmm.score_samples(fsd_vec).item()
        z_score = (raw_score - self.train_mean) / self.train_std

        return {
            "z_score": z_score,
            "raw_score": raw_score,
            "is_fake": z_score < self.threshold,
            "threshold": self.threshold,
        }

    async def score(self, image_path: str) -> dict:
        """Score an image by file path. Used for programmatic Ray calls."""
        try:
            return self._score_image(image_path)
        except Exception as e:
            self._logger.error(f"Error scoring {image_path}: {repr(e)}")
            return {"error": repr(e), "path": image_path}

    async def __call__(self, request: StarletteRequest) -> JSONResponse:
        """HTTP endpoint. Accepts JSON with 'path' or 'image_b64'."""
        try:
            data = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        try:
            if "path" in data:
                result = self._score_image(data["path"])
            elif "image_b64" in data:
                import base64
                from io import BytesIO
                from PIL import Image

                image_bytes = base64.b64decode(data["image_b64"])
                image = Image.open(BytesIO(image_bytes))
                result = self._score_image(image)
            else:
                return JSONResponse(
                    status_code=422,
                    content={"error": "Request must contain 'path' or 'image_b64'"},
                )
            return JSONResponse(content=result)
        except Exception as e:
            self._logger.error(f"Error: {repr(e)}\n{traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"error": repr(e)},
            )


def build_app(
    weights_dir="weights",
    threshold=-2.0,
    num_gpus=None,
    gpu_per_replica=1.0,
):
    """Build and deploy the Ray Serve application.

    Args:
        weights_dir: Path to weights directory.
        threshold: Z-score threshold.
        num_gpus: Total GPUs to use. If None, uses all available.
        gpu_per_replica: GPU fraction per replica.

    Returns:
        (handle, num_replicas) tuple.
    """
    import ray

    if num_gpus is None:
        if ray.is_initialized():
            num_gpus = int(ray.cluster_resources().get("GPU", 1))
        else:
            num_gpus = torch.cuda.device_count() or 1

    num_replicas = max(1, int(num_gpus / gpu_per_replica))
    weights_dir = str(Path(weights_dir).resolve())

    app = FSDScorer.options(
        ray_actor_options={"num_gpus": gpu_per_replica, "num_cpus": 1},
        num_replicas=num_replicas,
        max_ongoing_requests=1,
    ).bind(weights_dir=weights_dir, threshold=threshold)

    handle = serve.run(
        app,
        route_prefix="/",
        name="fsd-scorer",
        blocking=False,
    )

    return handle, num_replicas
