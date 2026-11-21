"""Analysis of experiment results: per-user breakdowns, significance tests and report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from recsys.analysis.eda import ACTIVITY_BUCKETS, bucket_label
from recsys.analysis.plots import BLUE, INK, INK_2, ORANGE, SURFACE, _save
from recsys.experiments.report import DISPLAY_NAMES
from recsys.paths import run_outputs_dir


def load_per_user(dataset: str, split_name: str, models: list[str], phase: str = "test") -> pd.DataFrame:
    """Long table: one row per (model, user) with every per-user metric."""
    frames = []
    for m in models:
        df = pd.read_parquet(run_outputs_dir(dataset, split_name, f"{m}_{phase}.parquet"))
        frames.append(df.assign(model=m))
    return pd.concat(frames, ignore_index=True)


def add_activity_bucket(per_user: pd.DataFrame) -> pd.DataFrame:
    edges = [lo for lo, _ in ACTIVITY_BUCKETS] + [np.inf]
    labels = [bucket_label(lo, hi) for lo, hi in ACTIVITY_BUCKETS]
    bucket = pd.cut(per_user["n_history"], bins=np.array(edges) - 0.5, labels=labels)
    return per_user.assign(bucket=bucket)


def by_activity(per_user: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """Mean ``metric`` per model and activity bucket (rows: buckets, columns: models)."""
    df = add_activity_bucket(per_user)
    # dropna=False keeps buckets that happen to have no users, so every table has the same rows
    table = df.pivot_table(
        index="bucket", columns="model", values=metric, aggfunc="mean", observed=False, dropna=False
    )
    table["users"] = df.drop_duplicates("user_idx").groupby("bucket", observed=False).size()
    return table


def paired_tests(per_user: pd.DataFrame, reference: str, metric: str = "ndcg@10") -> pd.DataFrame:
    """Wilcoxon signed-rank test of ``reference`` vs every other model on the same users,
    with Holm correction for multiple comparisons."""
    wide = per_user.pivot_table(index="user_idx", columns="model", values=metric)
    ref_name = DISPLAY_NAMES.get(reference, reference)
    rows = []
    for other in [m for m in wide.columns if m != reference]:
        diff = wide[reference] - wide[other]
        res = stats.wilcoxon(wide[reference], wide[other], zero_method="wilcox")
        better, worse, tied = (diff > 0).sum(), (diff < 0).sum(), (diff == 0).sum()
        rows.append(
            {
                "comparison": f"{ref_name} vs {DISPLAY_NAMES.get(other, other)}",
                "mean difference": diff.mean(),
                "relative difference": diff.mean() / wide[other].mean(),
                "users better / worse / tied": f"{better} / {worse} / {tied}",
                "p-value": res.pvalue,
            }
        )
    out = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    # Holm step-down: multiply the i-th smallest p by (m - i), keep monotone, cap at 1
    m = len(out)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * out["p-value"].to_numpy())
    out["p-value (Holm)"] = np.minimum(adjusted, 1.0)
    out["significant (p<0.05)"] = out["p-value (Holm)"] < 0.05
    return out


# ------------------------------------------------------------------------------- figures
def plot_panels(table: pd.DataFrame, models: list[str], out: Path, title: str, panel_title) -> Path:
    """Small multiples: one panel per row of ``table``, one labelled bar per model (identity comes from
    the axis label, not from colour, so this scales to any number of models). Best per panel in orange.
    ``panel_title(index, row)`` returns each panel's title."""
    buckets = list(table.index)
    names = [DISPLAY_NAMES.get(m, m) for m in models]
    height = 0.34 * len(models) + 1.6
    fig, axes = plt.subplots(1, len(buckets), figsize=(2.9 * len(buckets) + 1.4, height), sharey=True)
    xmax = float(np.nanmax(table[models].to_numpy(dtype=float))) * 1.35
    y = np.arange(len(models))[::-1]  # first model on top
    for ax, bucket in zip(np.atleast_1d(axes), buckets, strict=True):
        vals = table.loc[bucket, models].to_numpy(dtype=float)
        best = int(np.nanargmax(vals))
        ax.barh(y, vals, height=0.66, color=[ORANGE if j == best else BLUE for j in range(len(models))])
        for j, v in enumerate(vals):
            ax.annotate(f"{v:.3f}", (v, y[j]), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=INK if j == best else INK_2)  # fmt: skip
        ax.set_title(panel_title(bucket, table.loc[bucket]), fontsize=10)
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.tick_params(axis="y", length=0)
    np.atleast_1d(axes)[0].set_yticks(y, names)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return _save(fig, out)


def plot_by_activity(table: pd.DataFrame, models: list[str], out: Path, metric: str = "NDCG@10") -> Path:
    return plot_panels(
        table, models, out, f"Mean {metric} (test) by user activity; orange = best in group",
        lambda b, row: f"{b} liked movies\n({int(row['users']):,} users)",
    )  # fmt: skip


def plot_by_session(table: pd.DataFrame, models: list[str], out: Path, metric: str = "NDCG@10") -> Path:
    return plot_panels(
        table, models, out,
        f"Mean {metric} (test) by time between the last known rating and the first test rating",
        lambda b, row: f"{b}\n({int(row['users']):,} users)",
    )  # fmt: skip


def plot_mmr_tradeoff(rows: pd.DataFrame, out: Path) -> Path:
    """rows: columns lam, ndcg, ild (one per MMR setting). Accuracy vs. diversity as λ goes down."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rows["ild"], rows["ndcg"], color=BLUE, linewidth=2, marker="o", markersize=7, zorder=3)
    for _, r in rows.iterrows():
        label = "λ = 1 (no MMR)" if r["lam"] == 1 else f"λ = {r['lam']:g}"
        ax.annotate(
            label, (r["ild"], r["ndcg"]), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK
        )
    ax.set_xlabel("Intra-list diversity@10 (genre distance within a list)")
    ax.set_ylabel("NDCG@10 (test)")
    ax.set_title("Red Box: accuracy vs. diversity (MMR re-ranking)", pad=10)
    ax.margins(x=0.2, y=0.15)
    return _save(fig, out)


def plot_feature_importance(importance: pd.Series, out: Path, top: int = 15) -> Path:
    """Share of the ranker's total split gain per feature (top features)."""
    imp = importance.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(imp) + 1.4))
    ax.barh(imp.index, imp.to_numpy(), color=BLUE, height=0.66)
    for i, v in enumerate(imp.to_numpy()):
        ax.annotate(
            f"{v:.1%}",
            (v, i),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xlabel("Share of the ranker's total gain")
    ax.set_title("What the Red Box ranker relies on", pad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    return _save(fig, out)


def plot_validation_curves(curves: dict[str, pd.DataFrame], out: Path, metric: str = "NDCG@10") -> Path:
    """One small panel per trained model: validation metric vs epoch, best epoch marked."""
    n = len(curves)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.7 * rows), squeeze=False)
    for ax, (name, curve) in zip(axes.ravel(), curves.items(), strict=False):
        evals = curve.dropna(subset=["val"])
        ax.plot(evals["epoch"], evals["val"], color=BLUE, linewidth=2)
        best = evals.loc[evals["val"].idxmax()]
        ax.scatter([best["epoch"]], [best["val"]], s=45, color=ORANGE, zorder=3)
        label = f"best {best['val']:.4f}\nepoch {int(best['epoch'])}"
        ax.annotate(label, (best["epoch"], best["val"]), xytext=(0, -26), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)  # fmt: skip
        ax.set_title(DISPLAY_NAMES.get(name, name), fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.25)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Validation {metric} during training (early stopping)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)  # fmt: skip
    return _save(fig, out)


_LABEL_SPOTS = [  # (offset in points, horizontal alignment, vertical alignment), tried in order
    ((8, 0), "left", "center"),
    ((-8, 0), "right", "center"),
    ((0, 8), "center", "bottom"),
    ((0, -8), "center", "top"),
    ((8, 8), "left", "bottom"),
    ((8, -8), "left", "top"),
    ((-8, 8), "right", "bottom"),
    ((-8, -8), "right", "top"),
]


def _place_labels(ax, xs: np.ndarray, ys: np.ndarray, labels: list[str]) -> None:
    """Direct labels that avoid each other and the points: each label takes the first spot around its
    point whose box overlaps no label placed so far and no marker."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    centers = ax.transData.transform(np.column_stack([xs, ys]))
    pad = 6  # marker radius in pixels, plus a little air
    blocked = [
        plt.matplotlib.transforms.Bbox.from_extents(cx - pad, cy - pad, cx + pad, cy + pad)
        for cx, cy in centers
    ]
    for i in np.argsort(-np.asarray(ys)):  # top-down
        text = None
        others = [b for j, b in enumerate(blocked) if j != i]
        placed = False
        # nearby spots first; if all are taken, further away with a thin leader line to the point
        for scale in (1, 2.2, 3.4):
            for (dx, dy), ha, va in _LABEL_SPOTS:
                if text is not None:
                    text.remove()
                leader = {"arrowstyle": "-", "color": INK_2, "lw": 0.6, "shrinkB": 5} if scale > 1 else None
                offset = (dx * scale, dy * scale)
                text = ax.annotate(labels[i], (xs[i], ys[i]), xytext=offset, textcoords="offset points",
                                   ha=ha, va=va, fontsize=9, color=INK, arrowprops=leader)  # fmt: skip
                box = text.get_window_extent(renderer).expanded(1.04, 1.1)
                if not any(box.overlaps(b) for b in others):
                    placed = True
                    break
            if placed:
                break
        blocked.append(text.get_window_extent(renderer))


def plot_accuracy_vs_coverage(results: pd.DataFrame, out: Path) -> Path:
    """results: columns model, NDCG@10, Coverage@10 (one row per model)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(
        results["Coverage@10"], results["NDCG@10"], s=70, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2
    )
    ax.set_xlabel("Catalog coverage@10 (share of movies ever recommended)")
    ax.set_ylabel("NDCG@10 (test)")
    ax.set_title("Accuracy vs. catalog coverage", pad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()  # final layout first: label placement works in screen coordinates
    _place_labels(
        ax, results["Coverage@10"].to_numpy(), results["NDCG@10"].to_numpy(), list(results["model"])
    )
    return _save(fig, out)


def plot_training_curve(curve: pd.DataFrame, out: Path, title: str = "BPR-MF training") -> Path:
    """curve: columns epoch, loss, val (val may be NaN between evaluations). Two panels, no dual axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(curve["epoch"], curve["loss"], color=BLUE, linewidth=2)
    ax1.set_title("Training loss (BPR)")
    ax1.set_xlabel("Epoch")
    evals = curve.dropna(subset=["val"])
    ax2.plot(evals["epoch"], evals["val"], color=BLUE, linewidth=2, marker="o", markersize=4)
    best = evals.loc[evals["val"].idxmax()]
    ax2.scatter([best["epoch"]], [best["val"]], s=80, color="#eb6834", zorder=3)
    ax2.annotate(
        f"best: epoch {int(best['epoch'])}\nNDCG@10 {best['val']:.4f}",
        (best["epoch"], best["val"]),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax2.set_title("Validation NDCG@10 (early stopping)")
    ax2.set_xlabel("Epoch")
    for ax in (ax1, ax2):
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return _save(fig, out)


SESSION_BUCKETS = [
    (0.0, 1.0, "same session (< 1 hour)"),
    (1.0, 24.0, "same day (1-24 h)"),
    (24.0, np.inf, "later (> 1 day)"),
]


def session_gap(split) -> pd.DataFrame:
    """Per user: hours between their last known (train + validation) interaction and their first test
    interaction, bucketed. In MovieLens most test items come from the same rating session."""
    known = split.frame("train", "val").groupby("user_idx")["timestamp"].max()
    first_test = split.test.groupby("user_idx")["timestamp"].min()
    gap = ((first_test - known.reindex(first_test.index)) / 3600.0).rename("gap_hours")
    labels = [label for *_, label in SESSION_BUCKETS]
    edges = [lo for lo, *_ in SESSION_BUCKETS] + [np.inf]
    bucket = pd.cut(gap, bins=edges, labels=labels, right=False)
    return pd.DataFrame({"gap_hours": gap, "session": bucket}).reset_index()


def by_session(per_user: pd.DataFrame, split, models: list[str], metric: str = "ndcg@10") -> pd.DataFrame:
    """Mean metric per model and session bucket (rows: buckets, columns: models + user counts)."""
    df = per_user[per_user["model"].isin(models)].merge(session_gap(split), on="user_idx")
    table = df.pivot_table(index="session", columns="model", values=metric, aggfunc="mean", observed=False)
    table["users"] = df.drop_duplicates("user_idx").groupby("session", observed=False).size()
    return table[[*models, "users"]]


def available_models(dataset: str, split, models: list[str]) -> list[str]:
    """Models (in config order) with a logged MLflow run on exactly this split and per-user results on disk.
    MLflow is the source of truth, so leftovers from quick untracked runs are never picked up."""
    from recsys.experiments import tracking

    mlflow = tracking.start(dataset)
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(dataset)
    fingerprint = split.stats["fingerprint"]
    logged = {
        r.data.tags.get("model")
        for r in client.search_runs([exp.experiment_id], f"tags.split_fingerprint = '{fingerprint}'")
    }
    on_disk = {m for m in models if run_outputs_dir(dataset, split.config.name, f"{m}_test.parquet").exists()}
    return [m for m in models if m in logged and m in on_disk]


def training_costs(dataset: str, models: list[str], split_fingerprint: str) -> pd.DataFrame:
    """Per model: training time, refit time, epochs used, and test evaluation time (from MLflow)."""
    from recsys.experiments import tracking

    mlflow = tracking.start(dataset)
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(dataset)
    rows = []
    for m in models:
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.model = '{m}' and tags.split_fingerprint = '{split_fingerprint}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            continue
        metrics, params = runs[0].data.metrics, runs[0].data.params
        rows.append(
            {
                "model": DISPLAY_NAMES.get(m, m),
                "fit on train (s)": metrics.get("fit_seconds"),
                "refit on train+val (s)": metrics.get("refit_seconds"),
                "epochs (best)": int(params["model.epochs"]) if "model.epochs" in params else None,
                "test scoring (s)": metrics.get("test_eval_seconds"),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def load_curve_from_mlflow(dataset: str, model: str, primary_metric: str = "ndcg@10") -> pd.DataFrame:
    """Training curve (epoch, loss, val) of the most recent MLflow run of ``model``."""
    from recsys.experiments import tracking

    mlflow = tracking.start(dataset)
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name(dataset)
    runs = client.search_runs(
        [exp.experiment_id], filter_string=f"tags.model = '{model}'", order_by=["attributes.start_time DESC"]
    )
    if not runs:
        raise LookupError(f"no MLflow run found for model '{model}' in experiment '{dataset}'")
    run_id = runs[0].info.run_id
    loss = {m.step: m.value for m in client.get_metric_history(run_id, "train_loss")}
    val_key = f"val_curve_{tracking.metric_key(primary_metric)}"
    val = {m.step: m.value for m in client.get_metric_history(run_id, val_key)}
    epochs = sorted(loss)
    return pd.DataFrame(
        {"epoch": epochs, "loss": [loss[e] for e in epochs], "val": [val.get(e) for e in epochs]}
    )
