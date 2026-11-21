# %% [markdown]
# # 03 · Results: all models, the hybrid, and Amazon (Weeks 2–3)
#
# **Setup**
# - Every model is evaluated with the same protocol: per-user chronological split, full-catalog
#   ranking, fit on train → early-stop on validation → refit on train + validation → test once.
# - Hyperparameters were tuned on validation only (`ml/configs/*-tuned.yaml`), with the same
#   time budget for every model.
# - SASRec and LightGCN keep their paper settings: a single training run takes more than an hour on this CPU,
#   longer than the whole tuning budget.
# - Every number below comes from the MLflow run with exactly these settings.
#
# Reproduce from `ml/`:
# `recsys tune`, `recsys benchmark -c ml-1m-tuned`, `recsys hybrid -c ml-1m-tuned`, and the same for
# `amazon-videogames`. Then run this notebook.

# %%
import numpy as np
import pandas as pd
from IPython.display import Image, display

from recsys.analysis import results as R
from recsys.experiments.config import ExperimentConfig
from recsys.experiments.report import DISPLAY_NAMES, per_user_from_mlflow, report_entries, rows_from_mlflow
from recsys.experiments.runner import find_run, load_split, params_hash
from recsys.paths import figures_dir, results_docs_dir

pd.set_option("display.width", 200)
FIG = figures_dir("results", "ml-1m")
cfg = ExperimentConfig.load("ml-1m-tuned")
split = load_split(cfg)
per_user = per_user_from_mlflow(cfg, split)
test = pd.read_csv(results_docs_dir("ml-1m", "tuned", "test.csv"))


def show(path) -> None:
    display(Image(filename=str(path), width=950))


def name(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


# %% [markdown]
# ## 1. Main results: MovieLens-1M (test set)

# %%
cols = ["model", "family", "NDCG@10", "Recall@10", "HR@10", "MRR@10", "Coverage@10", "Novelty@10", "ILD@10"]
test[cols].sort_values("NDCG@10", ascending=False).set_index("model").round(4)

# %% [markdown]
# - **The proposed Red Box hybrid is the most accurate system on every accuracy metric** (NDCG@10 0.1790).
# - **The best single model is SASRec with the cross-entropy loss** (0.1643), followed by the simple Markov
#   chain (0.1097).
# - **Every non-sequential model scores between 0.070 and 0.081**, with EASE best.
# - **SASRec also recommends from the widest part of the catalog** (64% coverage, highest novelty). The
#   hybrid trades a little of that coverage for accuracy.

# %% [markdown]
# ## 2. RQ1: do deep models beat well-tuned classic CF? Paired significance tests
#
# Wilcoxon signed-rank tests on per-user NDCG@10 (the same 6,034 users), with Holm correction.

# %%
pairs = [("hybrid", "sasrec_ce"), ("sasrec_ce", "ease"), ("markov", "ease"), ("ease", "lightgcn"),
         ("ease", "bpr"), ("ease", "ials"), ("ease", "neumf"), ("ease", "multvae")]  # fmt: skip
rows = [R.paired_tests(per_user[per_user.model.isin([a, b])], reference=a).iloc[0] for a, b in pairs]
tests = pd.DataFrame(rows).reset_index(drop=True)
tests = tests[["comparison", "relative difference", "users better / worse / tied", "p-value"]]
p = tests["p-value"].to_numpy()
order = np.argsort(p)
holm = np.empty_like(p)  # Holm step-down over these comparisons
holm[order] = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
tests["p (Holm)"] = holm
tests.style.format({"relative difference": "{:+.1%}", "p-value": "{:.1e}", "p (Holm)": "{:.1e}"})

# %% [markdown]
# **Among non-sequential models, no deep model beats tuned EASE.** EASE is significantly better than
# LightGCN (+10%), BPR-MF (+9%), iALS (+12%), NeuMF (+16%) and Mult-VAE (+16%). This matches the
# reproducibility studies of Ferrari Dacrema et al. (2019) and Rendle et al. (2020): a simple, well-tuned
# model is very hard to beat.
#
# **The model that does win, by a factor of two, is the sequential one (SASRec).** Section 3 shows why.

# %% [markdown]
# ## 3. Why sequential models win on MovieLens: the rating-session artifact

# %%
session = R.by_session(per_user, split, ["ease", "markov", "sasrec_ce", "hybrid"])
show(R.plot_by_session(session, ["ease", "markov", "sasrec_ce", "hybrid"], FIG / "results_by_session.png"))
session.rename(columns=name).round(4)

# %%
ablation_cfg = ExperimentConfig.load("ml-1m-ablations")
shuffled_hash = params_hash(ablation_cfg.models["sasrec_ce_shuffled"])
shuffled = find_run("ml-1m", split.stats["fingerprint"], "sasrec_ce_shuffled", shuffled_hash)
if shuffled is None:
    print("Order ablation not finished yet: re-run after `recsys benchmark -c ml-1m-ablations`.")
else:
    real = test.set_index("model").loc[name("sasrec_ce"), "NDCG@10"]
    print(f"SASRec (CE), real order:     test NDCG@10 {real:.4f}")
    print(f"SASRec (CE), shuffled order: test NDCG@10 {shuffled.data.metrics['test_ndcg_at_10']:.4f}")

# %% [markdown]
# **96% of MovieLens users (5,774 of 6,034) gave their first test rating less than an hour after their last
# known rating, in the same rating session.**
# - **For those users, SASRec is 2.1× as accurate as EASE** (0.169 vs 0.081).
# - **For the 219 users who came back more than a day later, the ranking flips:** EASE (0.085) beats
#   SASRec (0.067) and Markov (0.062).
#
# MovieLens users rated movies in bulk on pages the website arranged, so the *order* within a session mostly
# reflects how the site displayed movies, not taste (Fan et al., 2023). The sequential models learn that
# order, and the order ablation above measures how much of SASRec's score depends on it. The hybrid is the
# best system in the two large groups (it relies on the sequence signal within a session and on taste
# models otherwise). EASE beats it only in the small same-day group (41 users).

# %% [markdown]
# ## 4. RQ2: the Red Box hybrid

# %%
imp = pd.read_csv(results_docs_dir("ml-1m", "hybrid", "feature_importance.csv"), index_col=0).iloc[:, 0]
show(R.plot_feature_importance(imp, FIG / "hybrid_feature_importance.png"))

# %%
mmr = test[test.model.str.startswith("Red Box") | test.model.str.startswith("hybrid_mmr")].copy()
mmr["lam"] = [1.0 if m.startswith("Red Box") else float(m.replace("hybrid_mmr", "")) for m in mmr.model]
mmr = mmr.rename(columns={"NDCG@10": "ndcg", "ILD@10": "ild"}).sort_values("lam", ascending=False)
show(R.plot_mmr_tradeoff(mmr, FIG / "hybrid_mmr_tradeoff.png"))
mmr[["lam", "ndcg", "ild", "Coverage@10"]].round(4)

# %% [markdown]
# **The hybrid is significantly better than the best single model** (+9.0% NDCG@10 over SASRec, p ≈ 10⁻²⁶).
# - **The ranker leans heavily on SASRec's rank** (52% of its total gain), the strongest single signal. EASE's
#   taste score (7%), SASRec's score, the user's history length, release year, LightGCN, Markov,
#   similarity to the last movie and genre match account for the rest. It learns *when* to trust the
#   sequential signal and when to fall back on taste: this is why it beats SASRec for users who return
#   after more than a day (section 3).
# - **MMR diversity has a smooth, controllable cost.** With λ = 0.8, genre diversity within a list rises
#   from 0.64 to 0.74 for −4% NDCG. With λ = 0.5 it reaches 0.86 for −18%. In the app this is the diversity
#   slider.

# %% [markdown]
# ## 5. RQ3: accuracy vs. catalog coverage, and who is served well

# %%
show(
    R.plot_accuracy_vs_coverage(
        test[~test.model.str.startswith("hybrid_mmr")], FIG / "results_accuracy_vs_coverage.png"
    )
)

# %%
models = [
    "mostpop",
    "userknn",
    "itemknn",
    "ease",
    "bpr",
    "ials",
    "neumf",
    "multvae",
    "lightgcn",
    "markov",
    "sasrec_ce",
    "hybrid",
]
activity = R.by_activity(per_user[per_user.model.isin(models)])
show(R.plot_by_activity(activity, models, FIG / "results_ndcg_by_activity.png"))

# %% [markdown]
# ## 6. RQ4: does it hold for products? Amazon Video Games

# %%
acfg = ExperimentConfig.load("amazon-videogames-tuned")
asplit = load_split(acfg)
amazon = pd.read_csv(results_docs_dir("amazon-videogames", "tuned", "test.csv"))
amazon[["model", "family", "NDCG@10", "Recall@10", "HR@10", "Coverage@10", "Novelty@10"]].sort_values(
    "NDCG@10", ascending=False
).set_index("model").round(4)

# %%
a_user = per_user_from_mlflow(acfg, asplit)
a_session = R.by_session(a_user, asplit, ["ease", "markov", "sasrec_ce", "hybrid"])
show(
    R.plot_by_session(
        a_session,
        ["ease", "markov", "sasrec_ce", "hybrid"],
        figures_dir("results", "amazon-videogames", "results_by_session.png"),
    )
)
a_session.rename(columns=name).round(4)

# %%
both = pd.DataFrame(
    {
        "MovieLens-1M": test.set_index("model")["NDCG@10"],
        "Amazon Video Games": amazon.set_index("model")["NDCG@10"],
    }
).dropna()
both["MovieLens rank"] = both["MovieLens-1M"].rank(ascending=False).astype(int)
both["Amazon rank"] = both["Amazon Video Games"].rank(ascending=False).astype(int)
both.sort_values("MovieLens rank").round(4)

# %% [markdown]
# **Amazon has no bulk-rating artifact.** Only 0.03% of reviews share a second with another review by the
# same user, against 77% on MovieLens. The picture changes accordingly:
# - **The Markov chain drops below EASE** (−12%, significant). Its MovieLens advantage came from rating
#   sessions.
# - **SASRec still beats EASE (+20%), including for users who return days later** (0.052 vs 0.036). Product
#   purchases do have a real "what comes next" structure: accessories, sequels, platforms.
# - **EASE is again the best non-sequential model**, significantly ahead of LightGCN and iALS (+20% each).
# - **The Red Box hybrid is again the best system** (+8.3% over SASRec, p ≈ 10⁻²⁷).
#
# So the core conclusions hold on both datasets: tuned EASE ≥ deep non-sequential models; the hybrid
# beats every single model. How large the sequential advantage is depends on how the data was collected.

# %% [markdown]
# ## 7. What tuning changed (validation vs. test)

# %%
default_cfg = ExperimentConfig.load("ml-1m")
fp = split.stats["fingerprint"]
tuned_rows = {r.model: r for r in rows_from_mlflow("ml-1m", report_entries(cfg), fp)}
default_rows = {r.model: r for r in rows_from_mlflow("ml-1m", report_entries(default_cfg), fp)}
changes = []
for m, r in tuned_rows.items():
    d = default_rows.get(m)
    if (
        d is None
        or m.startswith("hybrid")
        or params_hash(cfg.models.get(m, {})) == params_hash(default_cfg.models.get(m, {}))
    ):
        continue
    changes.append({"model": name(m), "val default": d.val["ndcg@10"], "val tuned": r.val["ndcg@10"],
                    "test default": d.test["ndcg@10"], "test tuned": r.test["ndcg@10"]})  # fmt: skip
changes = pd.DataFrame(changes).set_index("model")
changes["test change"] = changes["test tuned"] / changes["test default"] - 1
changes.round(4)

# %% [markdown]
# - **BPR-MF gains most from tuning:** a larger dimension and stronger regularization give +8% on test.
#   NeuMF and iALS gain 2–3%.
# - **The neighbourhood models improve on validation but are slightly *worse* on test.** That's
#   overfitting to the validation period, a reminder of why the test set must be touched only once.
# - **EASE's paper default was already optimal.**

# %% [markdown]
# ## 8. Rating prediction (experiment E7) and training cost

# %%
pd.read_csv(results_docs_dir("ml-1m", "rating_prediction", "results.csv")).set_index("model").drop(
    columns="params"
).round(4)

# %%
R.training_costs("ml-1m", [m for m in models if m != "hybrid"], fp).round(1)

# %% [markdown]
# ## Summary for the report
#
# | Question | Answer |
# |---|---|
# | RQ1: do deep models beat well-tuned classic CF? | Not the non-sequential ones: tuned EASE significantly beats LightGCN, NeuMF, Mult-VAE, iALS and BPR-MF on both datasets. The sequential Transformer (SASRec) does win. |
# | RQ2: does the hybrid beat the best single model? | Yes, on both datasets: +9.0% (MovieLens) and +8.3% (Amazon) NDCG@10 over SASRec, highly significant |
# | RQ3: accuracy vs. diversity, and which users? | MMR gives a smooth trade-off (+16% diversity for −4% accuracy at λ = 0.8). On MovieLens, sequential models win inside rating sessions and taste models (EASE) for users who return later |
# | RQ4: does it hold for products? | The main conclusions hold. The Markov chain's MovieLens advantage disappears on Amazon, revealing a data-collection artifact |
# | Rating prediction (E7) | Biased MF: RMSE 0.863, against 0.927 for user + item biases and 1.162 for the global mean |
