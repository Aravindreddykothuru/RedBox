"""Smoke test a running Red Box deployment (local or live). Standard library only.

    python deploy/smoke_test.py --api http://127.0.0.1:8000 [--web http://localhost:3000]
    python deploy/smoke_test.py --api https://<user>-redbox-api.hf.space --web https://<app>.vercel.app

Read-only checks run everywhere. With real accounts ("local" auth) it also registers a throwaway account
(deleted again at the end), rates films and checks that recommendations react immediately; in "dev" mode
it does the same as a named test user.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

RESULTS: list[tuple[bool, str]] = []


def call(base: str, path: str, method: str = "GET", body: dict | None = None, user: str | None = None):
    """``user``: "Bearer <token>" for real accounts, or a name sent as X-User-Id (API in dev mode)."""
    req = urllib.request.Request(base.rstrip("/") + path, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode()
    if user and user.startswith("Bearer "):
        req.add_header("Authorization", user)
    elif user:
        req.add_header("X-User-Id", user)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
        ms = (time.perf_counter() - t0) * 1000
        return r.status, (json.loads(raw) if raw and r.headers.get_content_type() == "application/json" else raw), ms


def check(ok: bool, label: str) -> bool:
    RESULTS.append((ok, label))
    print(("PASS " if ok else "FAIL ") + label)
    return ok


def api_checks(api: str) -> None:
    status, health, ms = call(api, "/health")  # the first call may wake a sleeping free-tier Space
    check(status == 200 and health["status"] == "ok", f"/health ok ({health['dataset']}, {health['n_items']} items)")
    status, page, ms = call(api, "/movies?q=matrix")
    check(status == 200 and page["total"] >= 1, f"title search finds 'matrix' ({ms:.0f} ms)")
    movie_id = page["items"][0]["movie_id"]
    _, detail, _ = call(api, f"/movies/{movie_id}")
    check(detail["title"] != "", f"movie detail: {detail['title']}")
    _, similar, ms = call(api, f"/movies/{movie_id}/similar?mode=cf&k=5")
    check(len(similar) == 5, f"similar movies ({ms:.0f} ms)")
    if health.get("semantic_search"):
        _, hits, ms = call(api, "/search/semantic?q=time%20travel&k=5")
        check(len(hits) == 5, f"semantic search: {hits[0]['title']} ({ms:.0f} ms)")
    _, metrics, _ = call(api, "/lab/metrics")
    check("serving_model" in metrics, "offline metrics for the dashboard")

    user = f"smoke-{uuid.uuid4()}"
    password = None
    try:
        _, before, _ = call(api, "/recommendations?k=10", user=user)
    except urllib.error.HTTPError as e:
        check(e.code == 401, "personal endpoints require sign-in")
        # real accounts: register a throwaway account (deleted again at the end)
        password = uuid.uuid4().hex
        try:
            _, token, _ = call(
                api, "/auth/register", "POST", {"email": f"{user}@smoke-test.invalid", "password": password}
            )
        except urllib.error.HTTPError as e2:
            check(e2.code == 404, "accounts are managed by Supabase: personal checks skipped")
            return
        user = f"Bearer {token['access_token']}"
        check(True, "account registration works")
        _, before, _ = call(api, "/recommendations?k=10", user=user)
    seeds = [m["movie_id"] for m in call(api, "/onboarding/seeds?n=8", user=user)[1]]
    body = {"favorite_genres": [], "ratings": [{"movie_id": m, "rating": 5} for m in seeds[:6]]}
    status, first, _ = call(api, "/onboarding", "POST", body, user=user)
    check(status == 200 and len(first) > 0, "onboarding returns first recommendations")
    timings = []
    for _ in range(5):
        _, after, ms = call(api, "/recommendations?k=20", user=user)
        timings.append(ms)
    check([r["movie_id"] for r in after] != [r["movie_id"] for r in before], "ratings change recommendations")
    check(not {r["movie_id"] for r in after} & set(seeds[:6]), "rated films are never recommended")
    timings.sort()
    check(timings[2] < 1000, f"/recommendations median {timings[2]:.0f} ms (budget 1000 ms incl. network)")
    if password is not None:
        status, _, _ = call(api, "/me/delete", "POST", {"password": password}, user=user)
        check(status == 204, "test account deleted again")


def web_checks(web: str) -> None:
    for path in ("/", "/search", "/results", "/lab", "/results/ml-1m.csv"):
        status, _, ms = call(web, path)
        check(status == 200, f"web {path} ({ms:.0f} ms)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True)
    parser.add_argument("--web")
    args = parser.parse_args()
    try:
        api_checks(args.api)
        if args.web:
            web_checks(args.web)
    except (urllib.error.URLError, KeyError, IndexError) as e:
        check(False, f"unexpected error: {e!r}")
    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
