"""Assemble (and optionally publish) the Hugging Face Space that runs the Red Box API.

    # 1. build the folder (no account needed): dist/hf-space/
    uv run --project ml python deploy/huggingface/build_space.py

    # 2. publish it (a free Hugging Face account: a write token in HF_TOKEN, or `hf auth login`)
    uv run --project ml python deploy/huggingface/build_space.py --push <your-hf-username>/redbox-api
    # without a Postgres database, keep the Space's SQLite database in a private dataset (app/backup.py):
    #   ... --push <your-hf-username>/redbox-api --backup-repo <your-hf-username>/redbox-data

With --push, configuration is copied into the Space from these environment variables when set:
    secrets:   REDBOX_DATABASE_URL, REDBOX_JWT_SECRET, REDBOX_SUPABASE_JWT_SECRET
    variables: REDBOX_AUTH_MODE, REDBOX_ADMIN_EMAILS, REDBOX_SUPABASE_URL,
               REDBOX_CORS_ORIGINS, REDBOX_AB_EXPERIMENT, REDBOX_RELEASES_REFRESH_HOURS
With --backup-repo, the Space keeps its SQLite database in that private dataset (REDBOX_BACKUP_REPO), and
the token goes in as the Space secret HF_TOKEN so it can read and write it.

The bundle defaults to data/bundles/ml-32m (the app's MovieLens-32M models), else ml-1m.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "dist" / "hf-space"

SECRETS = ["REDBOX_DATABASE_URL", "REDBOX_JWT_SECRET", "REDBOX_SUPABASE_JWT_SECRET"]
VARIABLES = [
    "REDBOX_AUTH_MODE",
    "REDBOX_ADMIN_EMAILS",
    "REDBOX_SUPABASE_URL",
    "REDBOX_CORS_ORIGINS",
    "REDBOX_AB_EXPERIMENT",
    "REDBOX_RELEASES_REFRESH_HOURS",
]
LFS_PATTERNS = ["*.npy", "*.npz", "*.onnx", "*.parquet", "tokenizer.json"]


def build(bundle: Path, include_histories: bool = False) -> Path:
    """By default the Space gets aggregate item statistics instead of dataset_users.npz: those are the raw
    MovieLens viewing histories, which the MovieLens licence does not allow us to redistribute in a public
    Space. Recommendations are identical (tested); only the Model Lab's "MovieLens viewer" option goes."""
    if not (bundle / "manifest.json").exists():
        raise SystemExit(f"{bundle} is not an exported bundle: run `recsys export -c ml-1m-tuned` first")
    from recsys.serving.engine import write_item_stats

    if not (bundle / "item_stats.npz").exists():
        write_item_stats(bundle)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".pytest_cache", ".ruff_cache")
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        (OUT / "ml").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "ml" / name, OUT / "ml" / name)
        (OUT / "backend").mkdir(exist_ok=True)
        if (ROOT / "backend" / name).exists():
            shutil.copy2(ROOT / "backend" / name, OUT / "backend" / name)
    shutil.copytree(ROOT / "ml" / "src", OUT / "ml" / "src", ignore=ignore)
    shutil.copytree(ROOT / "backend" / "app", OUT / "backend" / "app", ignore=ignore)
    skip = [] if include_histories else ["dataset_users.npz"]
    shutil.copytree(bundle, OUT / "bundle", ignore=shutil.ignore_patterns("__pycache__", *skip))
    shutil.copy2(HERE / "Dockerfile", OUT / "Dockerfile")
    shutil.copy2(HERE / "SPACE_README.md", OUT / "README.md")
    (OUT / ".gitattributes").write_text(
        "".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS_PATTERNS), encoding="utf-8"
    )
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Space folder: {OUT} ({size / 1e6:.0f} MB)")
    return OUT


def push(space_id: str, backup_repo: str | None = None) -> None:
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "set HF_TOKEN to a Hugging Face *write* token (huggingface.co/settings/tokens), "
            "or sign in with `hf auth login`"
        )
    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in SECRETS:
        if os.environ.get(key):
            api.add_space_secret(space_id, key, os.environ[key])
    for key in VARIABLES:
        if os.environ.get(key):
            api.add_space_variable(space_id, key, os.environ[key])
    if backup_repo:
        api.create_repo(backup_repo, repo_type="dataset", private=True, exist_ok=True)
        api.add_space_variable(space_id, "REDBOX_BACKUP_REPO", backup_repo)
        api.add_space_secret(space_id, "HF_TOKEN", token)
    api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=OUT, commit_message="Deploy Red Box API"
    )
    owner, name = space_id.split("/")
    print(f"Pushed. Build logs: https://huggingface.co/spaces/{space_id}")
    print(f"API (once built): https://{owner}-{name}.hf.space/health".lower().replace("_", "-"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bundles = ROOT / "data" / "bundles"
    default = bundles / "ml-32m" if (bundles / "ml-32m" / "manifest.json").exists() else bundles / "ml-1m"
    parser.add_argument("--bundle", type=Path, default=default)
    parser.add_argument("--push", metavar="USER/SPACE", help="publish to this Hugging Face Space")
    parser.add_argument(
        "--backup-repo",
        metavar="USER/DATASET",
        help="keep the Space's SQLite database in this private dataset (no Postgres needed)",
    )
    parser.add_argument(
        "--include-histories",
        action="store_true",
        help="also ship dataset_users.npz (raw MovieLens histories): only for a PRIVATE Space",
    )
    args = parser.parse_args()
    build(args.bundle, include_histories=args.include_histories)
    if args.push:
        push(args.push, backup_repo=args.backup_repo)


if __name__ == "__main__":
    main()
