"""Report-ready result tables (Markdown + CSV).

Tables are built from the MLflow experiment log, not from in-memory runs, so every number in the
report can be traced to a logged run (parameters, split fingerprint, training curve).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recsys.experiments.runner import RunResult, find_run, params_hash

log = logging.getLogger(__name__)


def hybrid_hash(cfg) -> str:
    """The hybrid's settings include its generators' settings, so tuned and default hybrids differ."""
    gens = {g: params_hash(cfg.models[g]) for g in cfg.hybrid.get("generators", [])}
    return params_hash({"hybrid": cfg.hybrid, "generators": gens})


def report_entries(cfg) -> dict[str, str]:
    """Every row of a report table: config entries plus the hybrid variants, with their settings hash."""
    entries = {name: params_hash(p) for name, p in cfg.models.items()}
    if cfg.hybrid:
        h = hybrid_hash(cfg)
        for lam in cfg.hybrid.get("mmr_lambdas", [1.0]):
            entries["hybrid" if float(lam) == 1.0 else f"hybrid_mmr{float(lam):g}"] = h
    return entries


def per_user_from_mlflow(
    cfg, split, entries: dict[str, str] | None = None, phase: str = "test"
) -> pd.DataFrame:
    """Per-user metrics of each entry, read from the artifacts of the MLflow run with exactly its settings.
    (Files on disk are named by model only and get overwritten by later runs, so MLflow is the record.)"""
    import mlflow

    from recsys.experiments import tracking

    tracking.start(cfg.dataset)
    frames = []
    for name, settings_hash in (entries or report_entries(cfg)).items():
        run = find_run(cfg.dataset, split.stats["fingerprint"], name, settings_hash)
        if run is None:
            continue
        path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path=f"per_user/{name}_{phase}.parquet"
        )
        frames.append(pd.read_parquet(path).assign(model=name))
    return pd.concat(frames, ignore_index=True)


# (metric key, column header)
COLUMNS = [
    ("ndcg@10", "NDCG@10"),
    ("recall@10", "Recall@10"),
    ("hit@10", "HR@10"),
    ("precision@10", "P@10"),
    ("mrr@10", "MRR@10"),
    ("map@10", "MAP@10"),
    ("ndcg@20", "NDCG@20"),
    ("recall@20", "Recall@20"),
    ("coverage@10", "Coverage@10"),
    ("novelty@10", "Novelty@10"),
    ("ild@10", "ILD@10"),
    ("arp@10", "ARP@10"),
]
NOT_RANKED = {"arp@10"}  # lower popularity is not automatically "better"; shown without bolding

LEGEND = (
    "All metrics are averaged over users and computed by ranking the **full catalog** (no sampling). "
    "Higher is better for every column except ARP@10 (average popularity of recommended items: "
    "lower means less popularity bias, which is not automatically better). "
    "Best value per column in **bold**. "
    "fit (s) is training time only: UserKNN finds neighbours when it predicts, so its cost appears at "
    "prediction time instead (logged in MLflow as eval seconds)."
)

DISPLAY_NAMES = {
    "mostpop": "MostPop",
    "userknn": "UserKNN",
    "itemknn": "ItemKNN",
    "ease": "EASE",
    "bpr": "BPR-MF",
    "ials": "iALS",
    "neumf": "NeuMF",
    "multvae": "Mult-VAE",
    "lightgcn": "LightGCN",
    "sasrec": "SASRec",
    "sasrec_ce": "SASRec (CE loss)",
    "sasrec_ce_shuffled": "SASRec (CE, shuffled order)",
    "markov": "Markov chain",
    "hybrid": "Red Box hybrid",
}

FAMILIES = {
    "markov": "Sequential (Markov)",
    "hybrid": "Two-stage hybrid (ours)",
    "mostpop": "Baseline",
    "userknn": "Neighbourhood",
    "itemknn": "Neighbourhood",
    "ease": "Linear autoencoder",
    "bpr": "Latent factor",
    "ials": "Latent factor",
    "neumf": "Deep (MLP)",
    "multvae": "Deep (VAE)",
    "lightgcn": "Graph neural network",
    "sasrec": "Sequential (Transformer)",
}


@dataclass
class ResultRow:
    model: str  # run name (config entry)
    model_class: str  # registry name, used for the family column
    val: dict[str, float]
    test: dict[str, float]
    fit_seconds: float
    refit_seconds: float


def rows_from_results(results: list[RunResult]) -> list[ResultRow]:
    return [
        ResultRow(
            r.model, r.model_class or r.model, r.val.metrics, r.test.metrics, r.fit_seconds, r.refit_seconds
        )
        for r in results
    ]


def rows_from_mlflow(dataset: str, entries: dict[str, str], split_fingerprint: str) -> list[ResultRow]:
    """Latest logged run of each entry with exactly the given settings hash (entry -> hash) on this split.
    Entries without such a run are skipped with a warning."""
    from recsys.experiments import tracking
    from recsys.experiments.runner import find_run

    rows = []
    for name, settings_hash in entries.items():
        run = find_run(dataset, split_fingerprint, name, settings_hash)
        if run is None:
            log.warning(
                "no MLflow run for '%s' (settings %s) on split %s; left out",
                name,
                settings_hash,
                split_fingerprint,
            )
            continue
        m, tags = run.data.metrics, run.data.tags

        def phase(prefix: str, metrics=m) -> dict[str, float]:
            return {key: metrics.get(f"{prefix}_{tracking.metric_key(key)}") for key, _ in COLUMNS}

        rows.append(
            ResultRow(
                model=name,
                model_class=tags.get("model_class", name),
                val=phase("val"),
                test=phase("test"),
                fit_seconds=m.get("fit_seconds", float("nan")),
                refit_seconds=m.get("refit_seconds", float("nan")),
            )
        )
    return rows


def results_frame(rows: list[ResultRow], phase: str) -> pd.DataFrame:
    records = []
    for r in rows:
        metrics = getattr(r, phase)
        record = {"model": DISPLAY_NAMES.get(r.model, r.model), "family": FAMILIES.get(r.model_class, "")}
        record |= {header: metrics.get(key) for key, header in COLUMNS}
        record["fit (s)"] = r.fit_seconds if phase == "val" else r.refit_seconds
        records.append(record)
    return pd.DataFrame(records)


def to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    text_cols = {"model", "family"}
    align = ["---" if h in text_cols else "---:" for h in headers]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    best = {}
    for key, header in COLUMNS:
        if header in df and key not in NOT_RANKED and df[header].notna().any():
            best[header] = df[header].max()
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if h in text_cols:
                cells.append(str(v))
                continue
            if v is None or pd.isna(v):
                cells.append("n/a")
                continue
            text = f"{v:.1f}" if h in ("fit (s)", "ARP@10") else f"{v:.4f}"
            if h in best and v == best[h]:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_tables(rows: list[ResultRow], out_dir: Path, title: str, notes: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = [f"# {title}", "", *notes, "", LEGEND, ""]
    for phase, label in (("test", "Test set (final numbers)"), ("val", "Validation set")):
        df = results_frame(rows, phase)
        df.to_csv(out_dir / f"{phase}.csv", index=False)
        sections += [f"## {label}", "", to_markdown(df), ""]
    path = out_dir / "README.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path
