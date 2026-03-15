"""Auto-download and cache pre-trained weights from GitHub releases."""

import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

# GitHub release URL pattern
_REPO = "ductai199x/Forensic-Self-Descriptions-CVPR25"
_RELEASE_TAG = "v1.1.0"
_BASE_URL = f"https://github.com/{_REPO}/releases/download/{_RELEASE_TAG}"

_WEIGHT_FILES = ["config.json", "fre.pt", "gmm.pt", "fsd_transforms.pt"]
_ATTRIBUTION_FILES = ["attribution_transforms.pt", "source_gmms.pt"]

_CACHE_DIR = Path.home() / ".cache" / "fsd"


def get_weights_dir(attribution=False):
    """Find or download pre-trained weights.

    Search order:
        1. ./weights/ (repo checkout)
        2. <package>/weights/ (installed alongside code)
        3. ~/.cache/fsd/ (auto-downloaded)

    Args:
        attribution: If True, also ensure attribution weight files are present.

    Returns:
        Path to weights directory containing config.json and weight files.
    """
    def _has_weights(d, need_attribution=False):
        if not (d / "config.json").exists():
            return False
        if need_attribution:
            return all((d / f).exists() for f in _ATTRIBUTION_FILES)
        return True

    # 1. Repo checkout
    cwd_weights = Path.cwd() / "weights"
    if _has_weights(cwd_weights, attribution):
        return cwd_weights

    # 2. Installed alongside package
    pkg_weights = Path(__file__).parent.parent / "weights"
    if _has_weights(pkg_weights, attribution):
        return pkg_weights

    # 3. Cached download
    if _has_weights(_CACHE_DIR, attribution):
        return _CACHE_DIR

    # Need to download
    return download_weights(attribution=attribution)


def download_weights(dest=None, attribution=False):
    """Download pre-trained weights from GitHub releases.

    Args:
        dest: Destination directory. Defaults to ~/.cache/fsd/.
        attribution: If True, also download attribution weight files.

    Returns:
        Path to the weights directory.
    """
    dest = Path(dest) if dest is not None else _CACHE_DIR
    dest.mkdir(parents=True, exist_ok=True)

    files = _WEIGHT_FILES + (_ATTRIBUTION_FILES if attribution else [])
    for filename in files:
        filepath = dest / filename
        if filepath.exists():
            continue

        url = f"{_BASE_URL}/{filename}"
        print(f"Downloading {filename}...", end=" ", flush=True, file=sys.stderr)

        try:
            req = Request(url, headers={"User-Agent": "fsd-detector"})
            with urlopen(req, timeout=300) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None

                # Download to temp file then rename (atomic)
                tmp = filepath.with_suffix(".tmp")
                downloaded = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\rDownloading {filename}... {pct}%", end="", flush=True, file=sys.stderr)

                tmp.rename(filepath)
                size_mb = downloaded / (1 << 20)
                print(f"\rDownloading {filename}... done ({size_mb:.1f} MB)", file=sys.stderr)

        except (URLError, OSError) as e:
            # Clean up partial download
            tmp = filepath.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {filename} from {url}: {e}\n"
                f"You can download weights manually from:\n"
                f"  https://github.com/{_REPO}/releases/tag/{_RELEASE_TAG}"
            ) from e

    return dest
