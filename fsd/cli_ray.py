"""Ray-based CLI for parallel FSD scoring across multiple GPUs.

Usage:
    # Start the scoring service:
    fsd-score-ray serve

    # Score images against the running service:
    fsd-score-ray score photo.jpg img2.png img3.webp
    fsd-score-ray score --dir path/to/images/
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import click

from .cli import IMAGE_EXTENSIONS, _find_weights_dir


@click.group()
def main():
    """FSD scoring with Ray Serve for multi-GPU parallel processing.

    Start the service with 'serve', then score images with 'score'.
    """
    pass


@main.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--dir", "image_dir", type=click.Path(exists=True), help="Directory of images to score.")
@click.option("--url", default="http://localhost:8000", help="URL of the running fsd-score-ray serve instance.")
@click.option("--csv", "csv_output", is_flag=True, help="Output results as CSV.")
@click.option("--workers", type=int, default=32, help="Number of concurrent requests.")
def score(images, image_dir, url, csv_output, workers):
    """Score images against a running FSD Ray Serve instance.

    Start the service first with: fsd-score-ray serve
    """
    image_paths = list(images)
    if image_dir:
        dir_path = Path(image_dir)
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(str(p) for p in dir_path.glob(f"*{ext}"))
            image_paths.extend(str(p) for p in dir_path.glob(f"*{ext.upper()}"))

    if not image_paths:
        click.echo("Error: No images specified. Provide image paths or use --dir.", err=True)
        sys.exit(1)

    image_paths = sorted(set(image_paths))

    # Check service is reachable (HTTPError means service is up but rejected GET)
    try:
        urlopen(url, timeout=2)
    except HTTPError:
        pass  # Service is up, just doesn't handle GET
    except URLError:
        click.echo(f"Error: Cannot reach service at {url}. Start it first with: fsd-score-ray serve", err=True)
        sys.exit(1)

    click.echo(f"Scoring {len(image_paths)} image(s) via {url}...\n", err=True)

    if csv_output:
        click.echo("file,z_score,raw_score,is_fake,threshold")

    def _score_one(path):
        abs_path = str(Path(path).resolve())
        body = json.dumps({"path": abs_path}).encode()
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as resp:
            return path, json.loads(resp.read())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, p): p for p in image_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                path, result = future.result()
                if "error" in result:
                    if csv_output:
                        click.echo(f"{path},,,error: {result['error']}")
                    else:
                        click.echo(f"[ERROR] {path}: {result['error']}", err=True)
                else:
                    if csv_output:
                        click.echo(f"{path},{result['z_score']:.6f},{result['raw_score']:.6f},{result['is_fake']},{result['threshold']}")
                    else:
                        label = "FAKE" if result["is_fake"] else "REAL"
                        click.echo(f"[{label}]  z={result['z_score']:+.4f}  {path}")
            except Exception as e:
                if csv_output:
                    click.echo(f"{path},,,error: {e}")
                else:
                    click.echo(f"[ERROR] {path}: {e}", err=True)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--port", type=int, default=8000, help="Port to bind to.")
@click.option("--threshold", type=float, default=None, help="Z-score threshold (default: -2.0).")
@click.option("--weights-dir", type=click.Path(exists=True), default=None, help="Path to weights directory.")
@click.option("--num-gpus", type=int, default=None, help="Number of GPUs (default: all available).")
@click.option("--gpu-per-replica", type=float, default=1.0, help="GPU fraction per replica.")
def serve(host, port, threshold, weights_dir, num_gpus, gpu_per_replica):
    """Start a persistent HTTP scoring service.

    Once running, score images with 'fsd-score-ray score' or via HTTP:

    \b
        curl -X POST http://localhost:8000 \\
          -H "Content-Type: application/json" \\
          -d '{"path": "/absolute/path/to/image.jpg"}'

    \b
    Response:
        {"z_score": -3.5, "raw_score": -2512.3, "is_fake": true, "threshold": -2.0}
    """
    if weights_dir is None:
        weights_dir = _find_weights_dir()
        if weights_dir is None:
            click.echo(
                "Error: Could not find weights directory. Use --weights-dir.",
                err=True,
            )
            sys.exit(1)

    if threshold is None:
        with open(Path(weights_dir) / "config.json") as f:
            threshold = json.load(f)["scoring"]["default_threshold"]

    try:
        import ray
    except ImportError:
        click.echo(
            "Error: Ray is required. Install with: pip install 'fsd-detector[ray]'",
            err=True,
        )
        sys.exit(1)

    ray.init(ignore_reinit_error=True)

    from ray import serve as ray_serve

    click.echo(f"Starting Ray Serve on {host}:{port}...", err=True)
    ray_serve.start(http_options={"host": host, "port": port})

    from .serve import build_app

    handle, num_replicas = build_app(
        weights_dir=weights_dir,
        threshold=threshold,
        num_gpus=num_gpus,
        gpu_per_replica=gpu_per_replica,
    )

    click.echo(f"\nFSD scoring service running at http://{host}:{port} ({num_replicas} replica(s))", err=True)
    click.echo("Score images with:", err=True)
    click.echo(f"  fsd-score-ray score --url http://localhost:{port} photo.jpg", err=True)
    click.echo("\nPress Ctrl+C to stop.", err=True)

    try:
        import signal

        signal.pause()
    except KeyboardInterrupt:
        click.echo("\nShutting down...", err=True)
    finally:
        ray_serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
