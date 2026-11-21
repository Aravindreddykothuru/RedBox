"""Run Red Box publicly from this computer: the API here, a free Cloudflare tunnel, and the live website
(Vercel) pointed at it.

Free hosts no longer run a 1 GB Python API without a card, so the API runs on this laptop at full speed,
and Cloudflare's free "quick tunnel" (no account) gives it a public https address. The address is new
each time, so this script also sets it as the website's NEXT_PUBLIC_API_URL and redeploys the website
(about two minutes). The live site works while this script runs; Ctrl+C stops everything.

From the repository root:
    uv run --project backend python deploy/local/start_public.py

Needs: tools/cloudflared/cloudflared.exe (Cloudflare's tunnel program; see docs/DEPLOYMENT.md) and the
Vercel CLI signed in to the project (frontend/.vercel/project.json, made by `vercel link`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
BACKEND, FRONTEND = ROOT / "backend", ROOT / "frontend"
CLOUDFLARED = ROOT / "tools" / "cloudflared" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
SITE = "https://redbox-recommender.vercel.app"
VERCEL = "vercel@65.16.0"  # newer CLIs first download a native binary, which can hang
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def say(text: str) -> None:
    print(f"[red box] {text}", flush=True)


def healthy(url: str, timeout: float = 5.0) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def wait_until(check, seconds: float, what: str) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return
        time.sleep(2)
    raise SystemExit(f"{what} did not come up within {seconds:.0f} s")


def allows_site(api: str) -> bool:
    """Whether the running API accepts requests from the website (CORS)."""
    try:
        r = requests.options(
            f"{api}/health",
            headers={"Origin": SITE, "Access-Control-Request-Method": "GET"},
            timeout=5,
        )
        return r.headers.get("access-control-allow-origin") == SITE
    except requests.RequestException:
        return False


def start_api(port: int) -> subprocess.Popen | None:
    api = f"http://127.0.0.1:{port}"
    if healthy(api):
        if not allows_site(api):
            raise SystemExit(
                f"an API is already running on port {port} but does not accept {SITE}: "
                "stop it and run this again"
            )
        say(f"using the API already running at {api}")
        return None
    env = dict(os.environ)
    env["REDBOX_CORS_ORIGINS"] = json.dumps([SITE, "http://localhost:3000", "http://127.0.0.1:3000"])
    say(f"starting the API at {api} (loading the models takes about 20 seconds)")
    uv = shutil.which("uv") or "uv"
    proc = subprocess.Popen(
        [uv, "run", "--no-sync", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=BACKEND,
        env=env,
    )
    wait_until(lambda: healthy(api), 300, "the API")
    return proc


def start_tunnel(port: int) -> tuple[subprocess.Popen, str]:
    if not CLOUDFLARED.exists():
        raise SystemExit(f"Cloudflare's tunnel program is missing: {CLOUDFLARED} (see docs/DEPLOYMENT.md)")
    say("opening a free Cloudflare tunnel")
    proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    found: list[str] = []

    def read() -> None:  # keep draining its output, or the pipe fills and the tunnel stalls
        for line in proc.stdout:
            match = TUNNEL_URL.search(line)
            if match and not found:
                found.append(match.group(0))

    threading.Thread(target=read, daemon=True).start()
    wait_until(lambda: bool(found) or proc.poll() is not None, 90, "the tunnel address")
    if not found:
        raise SystemExit("the tunnel stopped before giving an address")
    url = found[0]
    say(f"tunnel: {url} (waiting until it answers)")
    wait_until(lambda: healthy(url, timeout=10), 180, "the tunnel")
    return proc, url


def vercel(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    npx = shutil.which("npx") or "npx"
    return subprocess.run(
        [npx, "--yes", VERCEL, *args],
        cwd=FRONTEND,
        input=stdin if stdin is not None else "",
        text=True,
        capture_output=True,
        check=False,
    )


def point_website(api_url: str) -> None:
    say("pointing the website at the tunnel and redeploying it (about two minutes)")
    # --force overwrites the address from the last run in one step
    added = vercel("env", "add", "NEXT_PUBLIC_API_URL", "production", "--force", stdin=api_url)
    if added.returncode != 0:
        raise SystemExit(f"could not set the website's API address:\n{added.stdout}{added.stderr}")
    deployed = vercel("deploy", "--prod", "--yes")
    if deployed.returncode != 0:
        raise SystemExit(f"the website did not redeploy:\n{deployed.stdout[-2000:]}{deployed.stderr[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-website", action="store_true", help="only the API and the tunnel")
    args = parser.parse_args()

    api = tunnel = None
    try:
        api = start_api(args.port)
        tunnel, url = start_tunnel(args.port)
        if not args.no_website:
            point_website(url)
        say(f"LIVE: {SITE}  (API {url})")
        say("keep this window open; Ctrl+C stops the site's API")
        while tunnel.poll() is None and (api is None or api.poll() is None):
            time.sleep(5)
        say("the tunnel or the API stopped")
    except KeyboardInterrupt:
        say("stopping")
    finally:
        for proc in (tunnel, api):
            if proc is not None and proc.poll() is None:
                proc.terminate()
    sys.exit(0)


if __name__ == "__main__":
    main()
