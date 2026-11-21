"""Known datasets and where to fetch them.

Integrity: every archive is checked against a pinned SHA-256 when we have one, otherwise
against the MD5 that GroupLens publishes next to the file. A corrupted or tampered download
fails loudly instead of silently producing different experiment numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_name: str
    extracted_dir: str  # top-level folder inside the archive
    sha256: str | None = None  # pinned after first verified download
    md5_url: str | None = None  # official checksum published by the provider
    description: str = ""


_GROUPLENS = "https://files.grouplens.org/datasets/movielens"

DATASETS: dict[str, DatasetSpec] = {
    "ml-1m": DatasetSpec(
        name="ml-1m",
        url=f"{_GROUPLENS}/ml-1m.zip",
        archive_name="ml-1m.zip",
        extracted_dir="ml-1m",
        # Verified against GroupLens' published MD5 c4d9eecfca2ab87c1945afe126590906
        sha256="a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        md5_url=f"{_GROUPLENS}/ml-1m.zip.md5",
        description="MovieLens 1M: 1,000,209 ratings from 6,040 users on ~3,900 movies (2000-2003)",
    ),
    "ml-32m": DatasetSpec(
        name="ml-32m",
        url=f"{_GROUPLENS}/ml-32m.zip",
        archive_name="ml-32m.zip",
        extracted_dir="ml-32m",
        # Verified against GroupLens' published MD5 on first download (2026-11-19)
        sha256="e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee",
        md5_url=f"{_GROUPLENS}/ml-32m.zip.md5",
        description="MovieLens 32M: used (filtered) to train the models behind the live app",
    ),
    "amazon-videogames": DatasetSpec(
        name="amazon-videogames",
        url=(
            "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/"
            "rating_only/Video_Games.csv.gz"
        ),
        archive_name="Video_Games.csv.gz",
        extracted_dir="amazon-videogames",
        # No checksum is published for this file; pinned on first download (2026-11-19).
        sha256="a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c",
        description="Amazon Reviews 2023, Video Games, 5-core ratings (Hou et al., 2024): 814,586 ratings",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}") from None
