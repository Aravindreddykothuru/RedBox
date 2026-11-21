"""Deploy the Red Box API to Google Cloud Run (free tier), from the same folder as the Hugging Face Space.

Hugging Face now charges for Docker Spaces, so the API runs on Cloud Run instead. Cloud Run's free
allowance (per billing account, per month: 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million
requests) covers a demo; the service scales to zero when idle, so the first visit after a pause waits
about half a minute while it starts.

Once, in your own terminal:
    1. Install the Google Cloud CLI (https://cloud.google.com/sdk/docs/install) and sign in:
           gcloud init          # sign in, then create or pick a project
    2. Link the project to a billing account (Cloud Console > Billing; the free tier needs one on file).

Then, from the repository root:
    uv run --project ml python deploy/cloudrun/deploy.py --project <your-project-id> \\
        --cors https://redbox-recommender.vercel.app --backup-repo <hf-user>/redbox-data

The database is SQLite in the container, kept in a private Hugging Face dataset between restarts
(backend/app/backup.py), so the service runs as ONE instance (--max-instances 1): two instances would
each keep their own copy. The Hugging Face token for that backup is read from HF_TOKEN or your
`hf auth login`. The sign-in key (REDBOX_JWT_SECRET) of an existing deployment is kept, so redeploying
never signs anyone out.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "huggingface"))

SERVICES = ["run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com"]


def gcloud_path() -> str:
    found = shutil.which("gcloud")
    if found:
        return found
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
    if default.exists():
        return str(default)
    raise SystemExit("the Google Cloud CLI is not installed: https://cloud.google.com/sdk/docs/install")


def run(*args: str, capture: bool = False) -> str:
    cmd = [gcloud_path(), *args]
    print("$ gcloud", " ".join(args), flush=True)  # safe to show: secrets only ever travel in the env file
    done = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return done.stdout if capture else ""


def existing_env(service: str, project: str, region: str) -> dict[str, str]:
    """The environment of the service already deployed (empty if there is none)."""
    try:
        out = run(
            "run",
            "services",
            "describe",
            service,
            f"--project={project}",
            f"--region={region}",
            "--format=json",
            capture=True,
        )
    except subprocess.CalledProcessError:
        return {}
    containers = json.loads(out)["spec"]["template"]["spec"]["containers"]
    return {e["name"]: e.get("value", "") for e in containers[0].get("env", [])}


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, help="Google Cloud project id")
    parser.add_argument("--region", default="us-central1", help="a free-tier region (default us-central1)")
    parser.add_argument("--service", default="redbox-api")
    parser.add_argument("--cors", action="append", default=[], help="the website's address (repeatable)")
    parser.add_argument("--backup-repo", help="private Hugging Face dataset that keeps the SQLite database")
    parser.add_argument("--skip-build", action="store_true", help="reuse dist/hf-space as it is")
    args = parser.parse_args()

    if not args.skip_build:
        from build_space import build

        bundles = ROOT / "data" / "bundles"
        build(bundles / "ml-32m" if (bundles / "ml-32m" / "manifest.json").exists() else bundles / "ml-1m")
    source = ROOT / "dist" / "hf-space"

    run("services", "enable", *SERVICES, f"--project={args.project}")
    before = existing_env(args.service, args.project, args.region)
    env = {
        "REDBOX_JWT_SECRET": before.get("REDBOX_JWT_SECRET") or secrets.token_urlsafe(48),
        "REDBOX_CORS_ORIGINS": json.dumps(args.cors or json.loads(before.get("REDBOX_CORS_ORIGINS", "[]"))),
    }
    if args.backup_repo:
        token = hf_token()
        if not token:
            raise SystemExit(
                "--backup-repo needs a Hugging Face write token: set HF_TOKEN or `hf auth login`"
            )
        from huggingface_hub import HfApi

        HfApi(token=token).create_repo(args.backup_repo, repo_type="dataset", private=True, exist_ok=True)
        env.update({"REDBOX_BACKUP_REPO": args.backup_repo, "HF_TOKEN": token})

    # values go through a file, not the command line: no secrets in the process list or shell history
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "env.yaml"
        env_file.write_text("".join(f"{k}: {json.dumps(v)}\n" for k, v in env.items()), encoding="utf-8")
        run(
            "run",
            "deploy",
            args.service,
            f"--project={args.project}",
            f"--region={args.region}",
            f"--source={source}",
            "--port=7860",
            "--memory=2Gi",
            "--cpu=1",
            "--min-instances=0",
            "--max-instances=1",  # one SQLite database (see the docstring)
            "--concurrency=20",
            "--timeout=300",
            "--cpu-boost",
            "--allow-unauthenticated",
            f"--env-vars-file={env_file}",
            "--quiet",
        )
    url = run(
        "run",
        "services",
        "describe",
        args.service,
        f"--project={args.project}",
        f"--region={args.region}",
        "--format=value(status.url)",
        capture=True,
    ).strip()
    print(f"\nAPI: {url}  (try {url}/health)")
    print("Next: point the website at it (Vercel env NEXT_PUBLIC_API_URL) and redeploy the website.")


if __name__ == "__main__":
    main()
