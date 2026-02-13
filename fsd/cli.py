"""Command-line interface for FSD scoring.

Usage:
    fsd-score photo.jpg
    fsd-score img1.jpg img2.png img3.webp
    fsd-score --dir path/to/images/
    fsd-score photo.jpg --threshold -3.0
    fsd-score photo.jpg --device cuda
    fsd-score photo.jpg --weights-dir ./weights
"""

import sys

from pathlib import Path

import click


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heif",
    ".heic",
}


def _find_weights_dir():
    """Find weights directory relative to the package."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent / "weights",  # installed or repo root
        Path.cwd() / "weights",
    ]
    for c in candidates:
        if (c / "config.json").exists():
            return c
    return None


@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--dir", "image_dir", type=click.Path(exists=True), help="Directory of images to score.")
@click.option("--threshold", type=float, default=None, help="Z-score threshold (default: -2.0). More negative = stricter.")
@click.option("--device", type=str, default="auto", help="Device: auto, cpu, or cuda.")
@click.option("--weights-dir", type=click.Path(exists=True), default=None, help="Path to weights directory.")
@click.option("--csv", "csv_output", is_flag=True, help="Output results as CSV.")
def main(images, image_dir, threshold, device, weights_dir, csv_output):
    """Score images for AI-generated content using Forensic Self-Descriptions.

    More negative z-scores indicate higher likelihood of being AI-generated.
    """
    from .detector import FSDDetector

    # Collect image paths
    image_paths = list(images)
    if image_dir:
        dir_path = Path(image_dir)
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(str(p) for p in dir_path.glob(f"*{ext}"))
            image_paths.extend(str(p) for p in dir_path.glob(f"*{ext.upper()}"))

    if not image_paths:
        click.echo("Error: No images specified. Provide image paths or use --dir.", err=True)
        sys.exit(1)

    # Deduplicate and sort
    image_paths = sorted(set(image_paths))

    # Find weights
    if weights_dir is None:
        weights_dir = _find_weights_dir()
        if weights_dir is None:
            click.echo(
                "Error: Could not find weights directory. "
                "Use --weights-dir or run from the repo root.",
                err=True,
            )
            sys.exit(1)

    # Load detector
    click.echo(f"Loading detector from {weights_dir} (device={device})...", err=True)
    detector = FSDDetector.load(weights_dir, device=device, threshold=threshold)
    click.echo(f"Scoring {len(image_paths)} image(s)...\n", err=True)

    # CSV header
    if csv_output:
        click.echo("file,z_score,raw_score,is_fake,threshold")

    # Score images
    for path in image_paths:
        try:
            result = detector.score(path)
            if csv_output:
                click.echo(f"{path},{result.z_score:.6f},{result.raw_score:.6f},{result.is_fake},{result.threshold}")
            else:
                label = "FAKE" if result.is_fake else "REAL"
                click.echo(f"[{label}]  z={result.z_score:+.4f}  {path}")
        except Exception as e:
            if csv_output:
                click.echo(f"{path},,,error: {e}")
            else:
                click.echo(f"[ERROR] {path}: {e}", err=True)


if __name__ == "__main__":
    main()
