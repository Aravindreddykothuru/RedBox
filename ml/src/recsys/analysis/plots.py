"""Report-ready EDA figures (PNG, 200 dpi) in one consistent style.

Style: one hue per chart unless a second carries meaning, recessive grid, direct labels
instead of legends for single series, no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from recsys.analysis.eda import ACTIVITY_BUCKETS, bucket_label, genre_table
from recsys.analysis.eda import gini as _gini
from recsys.data.schema import RecDataset

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2 / highlight
MUTED = "#b9b7b0"  # de-emphasized bars

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
    }
)

_thousands = FuncFormatter(lambda v, _: f"{v / 1000:,.4g}k" if v >= 1000 else f"{v:,.0f}")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def rating_distribution(ds: RecDataset, out: Path, threshold: float = 4.0) -> Path:
    counts = ds.interactions["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [BLUE if r >= threshold else MUTED for r in counts.index]
    bars = ax.bar(counts.index.astype(int).astype(str), counts.to_numpy(), color=colors, width=0.62)
    total = counts.sum()
    for bar, v in zip(bars, counts.to_numpy(), strict=True):
        ax.annotate(
            f"{v / total:.1%}",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    pos_share = counts[counts.index >= threshold].sum() / total
    ax.set_title("Rating distribution", pad=22)
    ax.text(
        0.0,
        1.025,
        f"Blue = counted as 'liked' (rating ≥ {threshold:g}): {pos_share:.1%} of ratings",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
    )
    ax.set_xlabel("Star rating")
    ax.set_ylabel("Number of ratings")
    ax.yaxis.set_major_formatter(_thousands)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)
    return _save(fig, out)


def user_activity(ds: RecDataset, out: Path) -> Path:
    per_user = ds.interactions.groupby("user_id").size()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    bins = np.logspace(np.log10(per_user.min()), np.log10(per_user.max()), 40)
    ax1.hist(per_user, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax1.set_xscale("log")
    ax1.axvline(per_user.median(), color=ORANGE, linewidth=2)
    ax1.annotate(
        f"median {per_user.median():.0f}",
        (per_user.median(), ax1.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax1.set_title("Ratings per user")
    ax1.set_xlabel("Ratings per user (log scale)")
    ax1.set_ylabel("Number of users")
    ax1.grid(axis="x", visible=False)

    labels, values = [], []
    for lo, hi in ACTIVITY_BUCKETS:
        mask = per_user >= lo if hi is None else per_user.between(lo, hi)
        labels.append(bucket_label(lo, hi))
        values.append(int(mask.sum()))
    bars = ax2.bar(labels, values, color=BLUE, width=0.62)
    for bar, v in zip(bars, values, strict=True):
        ax2.annotate(
            f"{v:,}",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax2.set_title("Users per activity group")
    ax2.set_xlabel("Ratings per user")
    ax2.grid(axis="x", visible=False)
    ax2.margins(y=0.12)
    return _save(fig, out)


def item_long_tail(ds: RecDataset, out: Path) -> Path:
    counts = np.sort(ds.interactions.groupby("item_id").size().to_numpy())[::-1]
    n = counts.size
    head = round(0.2 * n)
    head_share = counts[:head].sum() / counts.sum()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ranks = np.arange(1, n + 1)
    ax1.fill_between(ranks[:head], counts[:head], color=BLUE, linewidth=0)
    ax1.fill_between(ranks[head - 1 :], counts[head - 1 :], color=MUTED, linewidth=0)
    ax1.set_title("Item popularity: the long tail")
    ax1.set_xlabel("Movies ranked by number of ratings")
    ax1.set_ylabel("Number of ratings")
    ax1.yaxis.set_major_formatter(_thousands)
    ax1.annotate(
        f"Top 20% of movies\n= {head_share:.0%} of all ratings",
        (head * 0.55, counts[0] * 0.55),
        color=INK,
        fontsize=9,
    )

    # Lorenz curve: cumulative share of ratings vs cumulative share of items (least popular first)
    asc = counts[::-1]
    cum_items = np.concatenate([[0], np.arange(1, n + 1) / n])
    cum_ratings = np.concatenate([[0], np.cumsum(asc) / asc.sum()])
    ax2.plot(cum_items, cum_items, color=INK_2, linewidth=1, linestyle="--")
    ax2.plot(cum_items, cum_ratings, color=BLUE, linewidth=2)
    ax2.annotate(
        "perfect equality", (0.22, 0.31), color=INK_2, fontsize=9, rotation=45, rotation_mode="anchor"
    )
    ax2.annotate(
        f"{ds.name}\nGini {_gini(counts):.2f}",
        (0.80, 0.04),
        color=BLUE,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax2.set_title("Lorenz curve of item popularity")
    ax2.set_xlabel("Share of movies (least → most popular)")
    ax2.set_ylabel("Share of ratings")
    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_aspect("equal")
    return _save(fig, out)


def ratings_over_time(ds: RecDataset, out: Path) -> Path:
    ts = pd.to_datetime(ds.interactions["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    idx = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(idx, monthly.to_numpy(), width=24, color=BLUE)
    ax.set_title("Ratings per month")
    ax.set_ylabel("Number of ratings")
    ax.yaxis.set_major_formatter(_thousands)
    ax.grid(axis="x", visible=False)
    top = monthly.idxmax()
    ax.annotate(
        f"peak: {top.strftime('%b %Y')} ({monthly.max() / monthly.sum():.0%} of all ratings)",
        (top.to_timestamp(), monthly.max()),
        xytext=(10, -4),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="top",
    )
    return _save(fig, out)


def genres(ds: RecDataset, out: Path) -> Path:
    table = genre_table(ds).sort_values("ratings")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.barh(table.index, table["ratings"], color=BLUE, height=0.62)
    for bar, (n_ratings, mean) in zip(bars, table[["ratings", "mean_rating"]].to_numpy(), strict=True):
        ax.annotate(
            f"{n_ratings / 1000:,.0f}k  ·  avg {mean:.2f}★",
            (n_ratings, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_title("Ratings per genre (a movie can have several genres)")
    ax.set_xlabel("Number of ratings")
    ax.xaxis.set_major_formatter(_thousands)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.28)
    return _save(fig, out)


def demographics(ds: RecDataset, out: Path) -> Path:
    if ds.users is None:
        raise ValueError(f"{ds.name} has no user demographics")
    users = ds.users
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [0.6, 1.2, 1.8]})

    gender = users["gender"].map({"M": "Male", "F": "Female"}).value_counts()
    age = users.sort_values("age_code").groupby("age_group", sort=False).size()
    occ = users["occupation"].value_counts().sort_values()

    for ax, series, title in [(axes[0], gender, "Gender"), (axes[1], age, "Age group")]:
        bars = ax.bar(series.index, series.to_numpy(), color=BLUE, width=0.62)
        for bar, v in zip(bars, series.to_numpy(), strict=True):
            ax.annotate(
                f"{v / len(users):.0%}",
                (bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK,
            )
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
        if title == "Age group":
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("Number of users")

    axes[2].barh(occ.index, occ.to_numpy(), color=BLUE, height=0.62)
    axes[2].set_title("Occupation")
    axes[2].grid(axis="y", visible=False)
    axes[2].tick_params(axis="y", labelsize=8)
    return _save(fig, out)


ALL_FIGURES = {
    "rating_distribution": rating_distribution,
    "user_activity": user_activity,
    "item_long_tail": item_long_tail,
    "ratings_over_time": ratings_over_time,
    "genres": genres,
    "demographics": demographics,
}


def render_all(ds: RecDataset, out_dir: Path) -> list[Path]:
    paths = []
    for name, fn in ALL_FIGURES.items():
        if name == "demographics" and ds.users is None:
            continue
        if name == "genres" and "genres" not in ds.items.columns:
            continue
        paths.append(fn(ds, out_dir / f"{name}.png"))
    return paths
