"""Render entrypoint: make sure the model bundle is on the persistent disk, then run the API.

The bundle (trained models, about 550 MB) lives in a private Hugging Face dataset (BUNDLE_REPO, read with
HF_TOKEN). It is downloaded onto the disk the first time, and again only when the dataset changes; the
API then saves new releases into that same folder, so they survive restarts too. If Hugging Face cannot
be reached and a bundle is already on the disk, the API starts with it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_bundle() -> None:
    bundle = Path(os.environ.get("REDBOX_BUNDLE_DIR", "/var/data/bundle"))
    repo = os.environ.get("BUNDLE_REPO")
    if not repo:
        return
    marker = bundle / ".revision"
    try:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().repo_info(repo, repo_type="dataset").sha
        if marker.exists() and marker.read_text().strip() == revision:
            print(f"model bundle up to date ({repo}@{revision[:7]})", flush=True)
            return
        print(f"downloading the model bundle {repo}@{revision[:7]} to {bundle}", flush=True)
        bundle.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=bundle)
        marker.write_text(revision)
        print("model bundle ready", flush=True)
    except Exception as e:
        if (bundle / "manifest.json").exists():
            print(
                f"could not check the bundle ({type(e).__name__}: {e}); using the one on the disk", flush=True
            )
        else:
            raise


def main() -> None:
    ensure_bundle()
    port = os.environ.get("PORT", "10000")
    args = ["-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port]
    args += ["--proxy-headers", "--forwarded-allow-ips", "*"]
    os.execv(sys.executable, [sys.executable, *args])


if __name__ == "__main__":
    main()
