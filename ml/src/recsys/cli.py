"""Command-line entry point: ``uv run recsys <command>``."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from recsys.data.registry import DATASETS, get_spec
from recsys.paths import figures_dir, processed_dir, raw_dir

app = typer.Typer(help="Collaborative filtering research toolkit.", no_args_is_help=True)

_LOADERS = {
    "ml-1m": "recsys.data.movielens:load_ml1m",
    "ml-32m": "recsys.data.movielens:load_ml32m",
    "amazon-videogames": "recsys.data.amazon:load_amazon_videogames",
}

DatasetArg = Annotated[str, typer.Argument(help=f"One of: {', '.join(DATASETS)}")]


def _loader(name: str):
    if name not in _LOADERS:
        raise typer.BadParameter(f"No parser for '{name}' yet. Available: {', '.join(_LOADERS)}")
    module, func = _LOADERS[name].split(":")
    return getattr(__import__(module, fromlist=[func]), func)


@app.command()
def datasets() -> None:
    """List known datasets."""
    for spec in DATASETS.values():
        typer.echo(f"{spec.name:8s} {spec.description}")


@app.command()
def download(name: DatasetArg, force: bool = False) -> None:
    """Download, checksum-verify and extract a dataset into data/raw/."""
    from recsys.data.download import fetch

    path = fetch(get_spec(name), force=force)
    typer.echo(f"Ready: {path}")


@app.command()
def prepare(name: DatasetArg) -> None:
    """Parse raw files into the shared schema and save parquet to data/processed/."""
    ds = _loader(name)(raw_dir(get_spec(name).extracted_dir))
    out = processed_dir(name)
    ds.save(out)
    typer.echo(f"Saved {len(ds.interactions):,} interactions, {len(ds.items):,} items -> {out}")


@app.command()
def tmdb(
    name: DatasetArg = "ml-1m",
    links_only: Annotated[bool, typer.Option(help="Only map ids (no TMDB key needed)")] = False,
) -> None:
    """Map a MovieLens dataset to TMDB ids (via ML-32M links.csv) and, with a TMDB key in the environment
    (TMDB_READ_TOKEN or TMDB_API_KEY), fetch posters and overviews into data/processed/<dataset>/."""
    from recsys.data import tmdb as tmdb_mod
    from recsys.data.schema import RecDataset

    links_csv = raw_dir("ml-32m") / "links.csv"
    if not links_csv.exists():
        raise typer.BadParameter("run `recsys download ml-32m` first (it contains links.csv)")
    items = RecDataset.load(processed_dir(name)).items
    links = tmdb_mod.load_links(links_csv)
    links = links[links["item_id"].isin(items["item_id"])]
    links.to_parquet(processed_dir(name) / "links.parquet", index=False)
    typer.echo(f"{len(links):,} of {len(items):,} movies have a TMDB id")
    if links_only:
        return
    meta = tmdb_mod.enrich(links, raw_dir("tmdb"))
    meta.to_parquet(processed_dir(name) / "tmdb.parquet", index=False)
    posters = int(meta["poster_path"].notna().sum())
    typer.echo(f"TMDB metadata for {len(meta):,} movies ({posters:,} with a poster)")


@app.command()
def eda(
    name: DatasetArg,
    threshold: Annotated[float, typer.Option(help="Rating counted as 'liked'")] = 4.0,
) -> None:
    """Compute dataset statistics and render report figures into docs/figures/eda/<dataset>/."""
    from recsys.analysis import eda as eda_mod
    from recsys.analysis import plots
    from recsys.data.schema import RecDataset

    ds = RecDataset.load(processed_dir(name))
    out_dir = figures_dir("eda", name)
    stats = eda_mod.summarize(ds, positive_threshold=threshold)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if "genres" in ds.items.columns:
        eda_mod.genre_table(ds).round(3).to_csv(out_dir / "genres.csv")
    for path in plots.render_all(ds, out_dir):
        typer.echo(f"figure: {path}")
    typer.echo(json.dumps(stats, indent=2))


# ----------------------------------------------------------------------------- experiments
ConfigOpt = Annotated[str, typer.Option("--config", "-c", help="Config name in ml/configs or a YAML path")]


def _setup_logging() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def _parse_params(pairs: list[str]) -> dict:
    import yaml

    params = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"--param expects key=value, got '{pair}'")
        params[key.strip()] = yaml.safe_load(value)
    return params


@app.command()
def split(config: ConfigOpt = "ml-1m") -> None:
    """Create (or load) the train/validation/test split, verify it, and print its statistics."""
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split

    cfg = ExperimentConfig.load(config)
    sp = load_split(cfg)
    sp.check()
    typer.echo(f"Split {cfg.dataset}/{sp.config.name}: all leakage checks passed")
    typer.echo(json.dumps(sp.stats, indent=2))


def _run_and_record(
    name: str, params: dict, split_obj, cfg, item_features, track: bool, save_models: bool = False
):
    from recsys.experiments.runner import log_to_mlflow, run_model, save_per_user

    result = run_model(name, params, split_obj, cfg, item_features, save_models=save_models and track)
    paths = save_per_user(result, split_obj, untracked=not track)
    if track:
        log_to_mlflow(result, split_obj, cfg, paths)
    return result


def _hybrid_hash(cfg) -> str:
    from recsys.experiments.report import hybrid_hash

    return hybrid_hash(cfg)


def _report_entries(cfg) -> dict[str, str]:
    from recsys.experiments.report import report_entries

    return report_entries(cfg)


def _default_report_name(cfg) -> str:
    stem = cfg.source.stem if cfg.source else cfg.dataset
    return stem.removeprefix(cfg.dataset).strip("-") or "default_hparams"


@app.command()
def train(
    model: Annotated[str, typer.Argument(help="Model name, e.g. ease, itemknn, bpr")],
    config: ConfigOpt = "ml-1m",
    param: Annotated[list[str], typer.Option("--param", "-p", help="Override, e.g. -p l2=300")] = [],  # noqa: B006
    track: Annotated[bool, typer.Option(help="Log the run to MLflow")] = True,
) -> None:
    """Train one model with the shared protocol and print validation + test metrics."""
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_item_features, load_split

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    split_obj = load_split(cfg)
    params = {**cfg.models.get(model, {}), **_parse_params(param)}
    result = _run_and_record(model, params, split_obj, cfg, load_item_features(split_obj), track)
    for phase in ("val", "test"):
        m = getattr(result, phase).metrics
        shown = {k: round(m[k], 4) for k in ("ndcg@10", "recall@10", "hit@10", "coverage@10") if k in m}
        typer.echo(f"{phase:5s} {shown}")


ReportNameOpt = Annotated[str, typer.Option(help="Output folder under docs/results/<dataset>/")]


def _write_report(cfg, split_obj, name: str):
    """Rebuild the report tables for every configured model from the MLflow log."""
    from recsys.experiments.report import rows_from_mlflow, write_tables
    from recsys.paths import results_docs_dir

    stats = split_obj.stats
    rows = rows_from_mlflow(cfg.dataset, _report_entries(cfg), stats["fingerprint"])
    notes = [
        f"Dataset **{cfg.dataset}**, split `{split_obj.config.name}` "
        f"(fingerprint `{stats['fingerprint']}`): {stats['n_users']:,} users, {stats['n_items']:,} items; "
        f"train/val/test interactions = {stats['interactions']['train']:,} / "
        f"{stats['interactions']['val']:,} / {stats['interactions']['test']:,}.",
        "",
        "Protocol: fit on train (iterative models early-stop on validation "
        f"{cfg.primary_metric}), then refit on train+validation and evaluate once on test.",
        "",
        f"Hyperparameters: `{cfg.source.name if cfg.source else cfg.dataset}`. {cfg.description}",
        "",
        "Every number comes from the latest MLflow run with exactly these settings on this split.",
    ]
    return write_tables(rows, results_docs_dir(cfg.dataset, name), f"Results: {cfg.dataset} ({name})", notes)


@app.command()
def benchmark(
    config: ConfigOpt = "ml-1m",
    models: Annotated[str, typer.Option(help="Comma-separated subset; default = all in config")] = "",
    track: Annotated[bool, typer.Option(help="Log the runs to MLflow")] = True,
    force: Annotated[bool, typer.Option(help="Re-run even if a run with identical settings exists")] = False,
    report_name: Annotated[str, typer.Option(help="Folder under docs/results/<dataset>/")] = "",
) -> None:
    """Train + evaluate models with the shared protocol (saving the fitted models), then rebuild the
    report tables from MLflow. Entries already run with identical settings are skipped."""
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import find_run, load_item_features, load_split, models_saved, params_hash

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    names = [m.strip() for m in models.split(",") if m.strip()] or list(cfg.models)
    unknown = set(names) - set(cfg.models)
    if unknown:
        raise typer.BadParameter(f"not in {config}: {sorted(unknown)}")
    split_obj = load_split(cfg)
    features = load_item_features(split_obj)
    fp = split_obj.stats["fingerprint"]
    # fitted models must be on disk only for the hybrid's and the app's generators
    needed = set(cfg.hybrid.get("generators", [])) | set(cfg.serving.get("generators", []))
    for n in names:
        h = params_hash(cfg.models[n])
        cached = find_run(cfg.dataset, fp, n, h) and (n not in needed or models_saved(split_obj, n, h))
        if track and not force and cached:
            typer.echo(f"{n}: already run with identical settings ({h}); skipped (use --force to re-run)")
            continue
        _run_and_record(n, cfg.models[n], split_obj, cfg, features, track, save_models=True)
    if track:
        path = _write_report(cfg, split_obj, report_name or _default_report_name(cfg))
        typer.echo(f"\nReport tables: {path}")
        typer.echo(path.read_text(encoding="utf-8"))


@app.command()
def report(config: ConfigOpt = "ml-1m", name: ReportNameOpt = "") -> None:
    """Rebuild the report tables from MLflow without training anything."""
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split

    cfg = ExperimentConfig.load(config)
    path = _write_report(cfg, load_split(cfg), name or _default_report_name(cfg))
    typer.echo(path.read_text(encoding="utf-8"))


@app.command()
def hybrid(
    config: ConfigOpt = "ml-1m-tuned", track: Annotated[bool, typer.Option(help="Log to MLflow")] = True
) -> None:
    """Train and evaluate the Red Box hybrid from the generators saved by `recsys benchmark`."""
    import pandas as pd

    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import (
        load_item_features,
        load_split,
        log_to_mlflow,
        params_hash,
        save_per_user,
    )
    from recsys.hybrid.run import disk_loader, run_hybrid
    from recsys.paths import results_docs_dir

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    if not cfg.hybrid:
        raise typer.BadParameter(f"{config} has no `hybrid:` section")
    split_obj = load_split(cfg)
    gens = cfg.hybrid["generators"]
    loader = disk_loader(split_obj, {g: (g, params_hash(cfg.models[g])) for g in gens})
    results, details = run_hybrid(cfg, split_obj, loader, load_item_features(split_obj))
    h = _hybrid_hash(cfg)
    for r in results:
        r.params_hash = h
        paths = save_per_user(r, split_obj, untracked=not track)
        if track:
            log_to_mlflow(r, split_obj, cfg, paths)
    out = results_docs_dir(cfg.dataset, "hybrid")
    out.mkdir(parents=True, exist_ok=True)
    details["importance"].rename("gain share").to_csv(out / "feature_importance.csv")
    rows = [
        {
            "variant": r.model,
            "MMR lambda": r.params["mmr_lambda"],
            **{  # .get: datasets without genres (Amazon) have no ILD
                k: r.test.metrics.get(k)
                for k in ("ndcg@10", "recall@10", "hit@10", "coverage@10", "novelty@10", "ild@10")
            },
        }
        for r in results
    ]
    table = pd.DataFrame(rows)
    table.to_csv(out / "mmr_tradeoff.csv", index=False)
    typer.echo(table.round(4).to_string(index=False))
    rv, rt = details["candidate_recall_val"], details["candidate_recall_test"]
    typer.echo(
        f"candidate recall: val {rv:.3f}, test {rt:.3f}; ranker best iteration {details['best_iteration']}"
    )


@app.command()
def rating(
    config: ConfigOpt = "ml-1m", track: Annotated[bool, typer.Option(help="Log to MLflow")] = True
) -> None:
    """Experiment E7: rating prediction (RMSE/MAE) on a split that keeps every rating."""
    from dataclasses import asdict

    import pandas as pd

    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_or_create_split
    from recsys.paths import results_docs_dir
    from recsys.rating import all_ratings_config, results_markdown, run_rating_benchmark

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    split_obj = load_or_create_split(cfg.dataset, all_ratings_config(cfg.split))
    results = run_rating_benchmark(split_obj)
    if track:
        from recsys.experiments import tracking

        mlflow = tracking.start(f"{cfg.dataset}-rating")
        for r in results:
            with mlflow.start_run(run_name=r.model):
                mlflow.set_tags({"split_fingerprint": split_obj.stats["fingerprint"], "model": r.model})
                mlflow.log_params(r.params)
                mlflow.log_metrics(
                    {k: getattr(r, k) for k in ("val_rmse", "val_mae", "test_rmse", "test_mae")}
                )
    out = results_docs_dir(cfg.dataset, "rating_prediction")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_csv(out / "results.csv", index=False)
    stats = split_obj.stats
    text = "\n".join(
        [
            f"# Rating prediction (experiment E7): {cfg.dataset}",
            "",
            f"Every rating counts (no 'liked' threshold). Split `{split_obj.config.name}` (fingerprint "
            f"`{stats['fingerprint']}`): train/val/test ratings = {stats['interactions']['train']:,} / "
            f"{stats['interactions']['val']:,} / {stats['interactions']['test']:,}, same per-user "
            "chronological protocol as the ranking experiments. Predictions are clipped to 1-5 stars. "
            "Lower is better; "
            "best test RMSE in **bold**.",
            "",
            results_markdown(results),
            "",
        ]
    )
    (out / "README.md").write_text(text, encoding="utf-8")
    typer.echo(text)


@app.command()
def tune(
    config: ConfigOpt = "ml-1m",
    models: Annotated[
        str, typer.Option(help="Comma-separated entries; default = all with a search space")
    ] = "",
    minutes: Annotated[float, typer.Option(help="Wall-clock budget per model")] = 25.0,
    max_trials: Annotated[int, typer.Option(help="Trial cap per model")] = 40,
) -> None:
    """Tune hyperparameters on the validation set (equal budget per model) and write <config>-tuned.yaml."""
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split
    from recsys.paths import results_docs_dir
    from recsys.tuning.report import tuning_markdown, write_tuned_config
    from recsys.tuning.spaces import has_space
    from recsys.tuning.tuner import tune_entry

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    split_obj = load_split(cfg)
    tunable = [e for e, p in cfg.models.items() if has_space(p.get("model", e))]
    names = [m.strip() for m in models.split(",") if m.strip()] or tunable
    bad = set(names) - set(tunable)
    if bad:
        raise typer.BadParameter(f"no search space / not in config: {sorted(bad)}")
    results = []
    for name in names:
        r = tune_entry(name, cfg.models[name], split_obj, cfg, minutes=minutes, max_trials=max_trials)
        counts = f"{r.n_complete} complete, {r.n_pruned} pruned"
        typer.echo(f"{name}: {counts}, default {r.default_value}, best {r.best_value}")
        results.append(r)
        path = write_tuned_config(
            cfg, [r], minutes, max_trials
        )  # saved after every model (safe to interrupt)
    out = results_docs_dir(cfg.dataset, "tuning")
    out.mkdir(parents=True, exist_ok=True)
    text = tuning_markdown(cfg, split_obj.stats["fingerprint"], tunable, minutes, max_trials)
    (out / "README.md").write_text(text, encoding="utf-8")
    typer.echo(f"\nTuned config: {path}\n\n{text}")


@app.command()
def export(
    config: ConfigOpt = "ml-1m-tuned",
    out: Annotated[str, typer.Option(help="Bundle folder (default: data/bundles/<dataset>)")] = "",
) -> None:
    """Export the serving bundle for the API: NumPy weights, the serving ranker, items, manifest."""
    from pathlib import Path

    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split
    from recsys.paths import data_dir
    from recsys.serving.bundle import export_bundle

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    if not cfg.serving:
        raise typer.BadParameter(f"{config} has no `serving:` section")
    target = Path(out) if out else data_dir() / "bundles" / cfg.dataset
    report = export_bundle(cfg, load_split(cfg), target)
    m = report.test_metrics
    typer.echo(
        f"Bundle: {report.path} ({report.seconds:.0f}s)\n"
        f"Serving hybrid on test: NDCG@10 {m['ndcg@10']:.4f}, HR@10 {m['hit@10']:.4f}, "
        f"coverage@10 {m['coverage@10']:.3f}"
    )
    if (processed_dir(cfg.dataset) / "content.parquet").exists():
        from recsys.content.build import add_content_to_bundle

        info = add_content_to_bundle(target, cfg.dataset)
        n_kw = info["items_with_keywords"]
        typer.echo(f"Content: {info['name']} embeddings added ({n_kw:,} with keywords)")
    else:
        typer.echo(f"(no semantic search: run `recsys content {cfg.dataset}` and export again)")
    from recsys.serving.forecast import add_forecast_to_bundle

    fc, _ = add_forecast_to_bundle(target)
    typer.echo(
        f"Forecast for trending and new users: {int(fc['n_viewers']):,} viewers ({float(fc['seconds']):.0f}s)"
    )


@app.command()
def forecast(
    config: ConfigOpt = "ml-32m",
    bundle: Annotated[str, typer.Option(help="Bundle folder (default: data/bundles/<dataset>)")] = "",
    viewers: Annotated[int, typer.Option(help="Dataset viewers to run the hybrid for")] = 4000,
    judges: Annotated[
        int, typer.Option(help="Other viewers to evaluate cold start on (0: build only, no evaluation)")
    ] = 5000,
) -> None:
    """Build the crowd forecast into a bundle (what the hybrid predicts viewers will watch next: the
    trending row and brand-new users) and compare it with popularity on the test split."""
    from pathlib import Path

    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split
    from recsys.paths import data_dir, results_docs_dir
    from recsys.serving.engine import Engine
    from recsys.serving.forecast import add_forecast_to_bundle, evaluate_forecast

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    target = Path(bundle) if bundle else data_dir() / "bundles" / cfg.dataset
    fc, others = add_forecast_to_bundle(target, n_viewers=viewers)
    typer.echo(f"Forecast from {int(fc['n_viewers']):,} viewers ({float(fc['seconds']):.0f}s) -> {target}")
    if judges <= 0:
        return
    table = evaluate_forecast(Engine(target), fc, load_split(cfg).targets("test"), others[:judges])
    out = results_docs_dir(cfg.dataset, "forecast")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "forecast.csv", index=False)
    typer.echo(table.round(4).to_string(index=False))
    typer.echo(f"-> {out / 'forecast.csv'}")


@app.command("serving-eval")
def serving_eval(
    config: ConfigOpt = "ml-32m",
    bundle: Annotated[str, typer.Option(help="Bundle folder (default: data/bundles/<dataset>)")] = "",
) -> None:
    """The served hybrid against each of its models on the same sampled test users as the export, with
    Wilcoxon tests (Holm) -> docs/results/<dataset>/serving/: test.csv (also the Results page's table),
    significance.csv and README.md. Uses the trained models and ranker as they are."""
    from pathlib import Path

    import pandas as pd

    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.report import COLUMNS, DISPLAY_NAMES, FAMILIES, to_markdown
    from recsys.experiments.runner import load_split
    from recsys.paths import data_dir, results_docs_dir
    from recsys.serving.bundle import evaluate_serving

    _setup_logging()
    cfg = ExperimentConfig.load(config)
    target = Path(bundle) if bundle else data_dir() / "bundles" / cfg.dataset
    results, tests = evaluate_serving(cfg, load_split(cfg), target)
    stem = cfg.source.stem if cfg.source else ""
    single = results_docs_dir(
        cfg.dataset, stem.removeprefix(cfg.dataset).strip("-") or "default_hparams", "test.csv"
    )
    fit = pd.read_csv(single).set_index("model")["fit (s)"].to_dict() if single.exists() else {}
    rows = []
    for name, r in results.items():
        cls = "hybrid" if name == "hybrid" else cfg.models[name].get("model", name)
        display = DISPLAY_NAMES.get(name, name)
        row = {"model": display, "family": FAMILIES.get(cls, FAMILIES.get(name, ""))}
        row |= {header: r.metrics.get(key) for key, header in COLUMNS}
        row["fit (s)"] = fit.get(display)
        rows.append(row)
    table = pd.DataFrame(rows)
    out = results_docs_dir(cfg.dataset, "serving")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "test.csv", index=False)
    tests.to_csv(out / "significance.csv", index=False)
    n_users = int(results["hybrid"].per_user["user_idx"].nunique())
    sig = [
        "| comparison | relative difference | users better / worse / tied | p-value (Holm) |",
        "|---|---:|---:|---:|",
    ]
    sig += [
        f"| {r['comparison']} | {r['relative difference']:+.1%} | {r['users better / worse / tied']} | "
        f"{r['p-value (Holm)']:.1e} |"
        for _, r in tests.iterrows()
    ]
    lines = [
        f"# Served hybrid vs its models: {cfg.dataset}",
        "",
        f"The hybrid the app serves and each of its candidate generators, evaluated on the same {n_users:,} "
        "sampled test users as the export (same seed). Full-catalogue ranking; the test set is used once. "
        f"Reproduce: `recsys serving-eval -c {stem or cfg.dataset}`.",
        "",
        to_markdown(table),
        "",
        "## Significance (Wilcoxon signed-rank on per-user NDCG@10, Holm-corrected)",
        "",
        *sig,
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    typer.echo(table[["model", "NDCG@10", "HR@10", "Coverage@10"]].round(4).to_string(index=False))
    typer.echo(tests[["comparison", "relative difference", "p-value (Holm)"]].to_string(index=False))
    typer.echo(f"-> {out}")


@app.command()
def newfilms(
    config: ConfigOpt = "ml-32m",
    language: Annotated[
        str,
        typer.Option(
            help="Original languages, comma-separated: English, Hindi, Telugu, Tamil, Kannada, Malayalam"
        ),
    ] = "English,Hindi,Telugu,Tamil,Kannada,Malayalam",
    since: Annotated[
        int, typer.Option(help="First release year (0: each language's own, see wikipedia.RELEASES)")
    ] = 0,
    until: Annotated[int, typer.Option(help="Last release year (0: this year)")] = 0,
    bundle: Annotated[str, typer.Option(help="Bundle folder (default: data/bundles/<dataset>)")] = "",
    evaluate: Annotated[
        bool,
        typer.Option(help="First measure the mapping, and how ratings warm it up, on films treated as new"),
    ] = True,
    refresh: Annotated[
        bool, typer.Option(help="Ask Wikidata again instead of reusing each language's saved table")
    ] = False,
    fetch_only: Annotated[
        bool, typer.Option(help="Only download the films' data; don't touch the bundle")
    ] = False,
) -> None:
    """New films nobody has rated: recent releases in six languages, from Wikidata and Wikipedia,
    recommended from their story (their iALS factors are predicted from their text; serving/newfilms.py).
    Replaces the bundle's new films with all the languages given; --evaluate first measures the idea on
    catalogue films treated as new (docs/results/<dataset>/newfilms/). The API adds films released later
    by itself, every day (backend/app/releases.py)."""
    from datetime import date
    from pathlib import Path

    import pandas as pd

    from recsys.data.wikipedia import LANGUAGES, RELEASES, language_films, save
    from recsys.experiments.config import ExperimentConfig
    from recsys.experiments.runner import load_split
    from recsys.paths import data_dir, results_docs_dir
    from recsys.serving.engine import Engine
    from recsys.serving.newfilms import build_new_films, evaluate_mapping, evaluate_warmup

    _setup_logging()
    languages = [x.strip().title() for x in language.split(",") if x.strip()]
    unknown = [x for x in languages if x not in LANGUAGES]
    if unknown:
        raise typer.BadParameter(f"unknown language {unknown}; choose from {', '.join(LANGUAGES)}")
    until = until or date.today().year
    folder = data_dir() / "processed" / "newfilms"
    tables = []
    for lang in languages:
        first, min_links = RELEASES[lang]
        first = since or first
        path = folder / f"{lang.lower()}_{first}-{until}.parquet"
        if path.exists() and not refresh:
            table = pd.read_parquet(path)
        else:
            # each language's articles are cached as they arrive, so an interrupted fetch resumes
            cache = folder / f"{lang.lower()}_pages.json"
            table = language_films(lang, first, until, cache=cache, min_links=min_links)
            save(table, path)
        typer.echo(f"{lang}: {len(table):,} films ({first}-{until})")
        tables.append(table)
    if fetch_only:
        return

    cfg = ExperimentConfig.load(config)
    target = Path(bundle) if bundle else data_dir() / "bundles" / cfg.dataset
    strength = None
    if evaluate:
        engine = Engine(target)
        split = load_split(cfg)
        out = results_docs_dir(cfg.dataset, "newfilms")
        out.mkdir(parents=True, exist_ok=True)
        table = evaluate_mapping(engine, split.targets("test"))
        table.to_csv(out / "cold_start.csv", index=False)
        typer.echo(
            f"{table.attrs['hidden films']} catalogue films hidden as new; ridge alpha {table.attrs['alpha']}"
        )
        typer.echo(table.round(4).to_string(index=False))
        warm = evaluate_warmup(engine, split.targets("val"), split.targets("test"))
        warm.to_csv(out / "warmup.csv", index=False)
        strength = warm.attrs["prior"]
        typer.echo(f"warm-up with ratings (prior strength {warm.attrs['prior']}, chosen on validation):")
        typer.echo(warm.round(4).to_string(index=False))
        del engine
    info = build_new_films(target, pd.concat(tables, ignore_index=True), strength=strength)
    typer.echo(f"added to the bundle: {info}")


@app.command("import-app-ratings")
def import_app_ratings(
    source: Annotated[str, typer.Argument(help="The API's /ops/ratings.csv URL, or a CSV file it produced")],
) -> None:
    """Bring the live app's ratings into the training data (data/raw/app/ratings.csv). From a URL, an
    operator's sign-in token is read from $REDBOX_TOKEN. Then retrain: train_app_models.sh --refresh."""
    import os
    from pathlib import Path

    import requests

    from recsys.data.movielens import APP_COLUMNS, app_ratings

    if source.startswith(("http://", "https://")):
        token = os.environ.get("REDBOX_TOKEN")
        if not token:
            raise typer.BadParameter("set REDBOX_TOKEN to an operator's sign-in token")
        r = requests.get(source, headers={"Authorization": f"Bearer {token}"}, timeout=120)
        r.raise_for_status()
        text = r.text
    else:
        text = Path(source).read_text(encoding="utf-8")
    if text.splitlines()[0].strip() != ",".join(APP_COLUMNS):
        raise typer.BadParameter(f"not an app ratings export: the header must be {','.join(APP_COLUMNS)}")
    target = raw_dir("app") / "ratings.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    rows = app_ratings(target)
    n_users = 0 if rows is None else rows["user_id"].nunique()
    typer.echo(f"{0 if rows is None else len(rows):,} app ratings from {n_users:,} users -> {target}")
    typer.echo("Next: bash ml/scripts/train_app_models.sh --refresh")


@app.command()
def wiki(
    name: DatasetArg = "ml-32m",
    refresh: Annotated[bool, typer.Option(help="Fetch every film again, not only the missing ones")] = False,
) -> None:
    """Posters and summaries from Wikipedia (no API key) for a dataset's films, matched through Wikidata by
    IMDb id -> data/processed/<dataset>/wiki.parquet. Then `recsys content` and export use them."""
    from recsys.data.wikipedia import fetch_wiki

    _setup_logging()
    table = fetch_wiki(name, refresh=refresh)
    typer.echo(
        f"{len(table):,} films: {table['wiki_title'].notna().sum():,} with a Wikipedia article, "
        f"{table['poster_url'].notna().sum():,} posters, {table['summary'].notna().sum():,} summaries"
    )


@app.command()
def content(
    name: DatasetArg = "ml-1m",
    bundle: Annotated[str, typer.Option(help="Also add the result to this exported bundle folder")] = "",
) -> None:
    """Build each movie's text (title, genres, ML-32M viewer keywords, TMDB plot when fetched) and its
    MiniLM sentence embedding, for semantic search and content similarity."""
    from pathlib import Path

    from recsys.content.build import add_content_to_bundle, build_content

    _setup_logging()
    table = build_content(name)
    with_kw = int((table["keywords"].map(len) > 0).sum())
    typer.echo(f"{len(table):,} movies embedded ({with_kw:,} with viewer keywords)")
    typer.echo(f"Example: {table['text'].iloc[0][:200]}")
    if bundle:
        info = add_content_to_bundle(Path(bundle), name)
        typer.echo(f"Added to {bundle}: {info}")


if __name__ == "__main__":
    app()
