"""Film posters from the API's own cache (app/posters.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse
from recsys.serving.engine import Engine

from app import posters
from app.services import get_engine
from app.settings import Settings, get_settings

router = APIRouter()
CACHED = {"Cache-Control": "public, max-age=31536000, immutable"}  # a film's poster never changes


@router.get(
    "/posters/{movie_id}.webp",
    tags=["catalog"],
    response_class=FileResponse,
    response_model=None,
    responses={
        200: {"content": {"image/webp": {}}, "description": "the poster from the cache, 342 px wide"},
        307: {"description": "not cached yet: Wikipedia's file, while the API caches it in the background"},
        404: {"description": "no such film, or the film has no poster"},
    },
)
def poster(
    movie_id: int, engine: Engine = Depends(get_engine), settings: Settings = Depends(get_settings)
) -> FileResponse | RedirectResponse:
    """A film's poster from Wikipedia at card size, served from the API's disk cache once it is there."""
    path = settings.poster_dir / f"{movie_id}.webp"
    if path.exists():
        return FileResponse(path, media_type="image/webp", headers=CACHED)
    idx = engine.movie_to_idx.get(movie_id)
    url = engine.items["poster"].iloc[idx] if idx is not None and "poster" in engine.items else None
    if not isinstance(url, str) or not url:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no poster for this film")
    posters.enqueue(path, url)
    # not cached: the browser loads Wikipedia's file this time (not remembered, so the next view asks again)
    return RedirectResponse(url, status.HTTP_307_TEMPORARY_REDIRECT, headers={"Cache-Control": "no-store"})
