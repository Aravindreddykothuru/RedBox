# %% [markdown]
# # 02 · Week 1 results: baseline models on MovieLens-1M
#
# Five models evaluated under the shared protocol (Methodology, Chapter 7):
# - per-user chronological split
# - full-catalog ranking
# - fit on train → early-stop on validation → refit on train + validation → test once
#
# **The hyperparameters are literature defaults, not tuned yet.** Every model gets the same Optuna
# tuning budget in Week 3, so these are first results, not final conclusions.
#
# Reproduce from `ml/`: `uv run recsys benchmark -c ml-1m`, then run this notebook.

# %%
import pandas as pd
from IPython.display import Image, display

from recsys.analysis import results as R
from recsys.experiments.config import ExperimentConfig
from recsys.experiments.report import DISPLAY_NAMES
from recsys.paths import figures_dir, results_docs_dir

cfg = ExperimentConfig.load("ml-1m")
MODELS = list(cfg.models)
FIG = figures_dir("results", "ml-1m")
test = pd.read_csv(results_docs_dir("ml-1m", "week1_baselines", "test.csv"))
per_user = R.load_per_user("ml-1m", cfg.split.name, MODELS)


def show(path) -> None:
    display(Image(filename=str(path), width=900))


# %% [markdown]
# ## 1. Main results (test set)

# %%
test.set_index("model").round(4)

# %% [markdown]
# **Findings**
# - **EASE is the most accurate model on every accuracy metric:** NDCG@10 0.0810 and HR@10 36.3%. For
#   about a third of users, at least one of the 10 recommended movies is one they actually liked next.
# - **The simple neighbourhood methods beat the learned factorization model at default settings:**
#   UserKNN 0.0727 and ItemKNN 0.0709 vs BPR-MF 0.0685. This echoes the reproducibility debate
#   (Ferrari Dacrema et al., 2019; Rendle et al., 2020). Week 3 tuning will show whether it holds.
# - **Personalization clearly pays off:** EASE improves NDCG@10 over MostPop by 74%.
# - **The absolute numbers look low compared with many papers.** Predicting each user's *future* likes,
#   ranked against the full catalog, is much harder than the random hold-out with ~100 sampled
#   candidates that is common in the literature.

# %% [markdown]
# ## 2. Are the differences real? Paired significance tests
#
# The Wilcoxon signed-rank test compares two models on **the same users** (per-user NDCG@10). Holm's
# correction keeps the overall false-positive rate at 5% across the several comparisons.

# %%
R.paired_tests(per_user, reference="ease")

# %%
R.paired_tests(per_user, reference="userknn")

# %%
wide = per_user.pivot_table(index="user_idx", columns="model", values="ndcg@10")
ties = pd.DataFrame(
    {
        other: {
            "tied users": int((wide["ease"] == wide[other]).sum()),
            "of which both score 0": int(((wide["ease"] == wide[other]) & (wide["ease"] == 0)).sum()),
        }
        for other in ["userknn", "itemknn", "bpr", "mostpop"]
    }
).T
print(f"Users for whom EASE finds none of their next liked movies: {(wide['ease'] == 0).mean():.1%}")
ties

# %% [markdown]
# - **EASE beats every other model significantly** (Holm-corrected p < 10⁻¹¹). For example, it improves
#   NDCG@10 by 11.5% over UserKNN.
# - **UserKNN vs ItemKNN is not significant** (p = 0.22): the two neighbourhood methods are statistically
#   tied.
# - **Most users are "tied" in each comparison** (about 3,400–3,650 of 6,034). In 95–99% of these cases,
#   *neither* model finds any of the user's next liked movies. Even EASE misses completely for 63.7% of
#   users, which shows how hard it is to predict what someone will like next.

# %% [markdown]
# ## 3. Accuracy vs. catalog coverage

# %%
show(R.plot_accuracy_vs_coverage(test, FIG / "week1_accuracy_vs_coverage.png"))

# %% [markdown]
# - **MostPop recommends only 3% of the catalog:** everyone gets the same blockbusters.
# - **BPR-MF covers the most of the catalog (41%) and has the highest novelty**, but is less accurate.
# - **EASE combines the best accuracy with moderate coverage (23%).**
# - **MostPop has the *highest* intra-list diversity**, because its single list mixes blockbusters from
#   many genres. *Diversity within one list* and *coverage across users* measure different things,
#   so the report shows both.

# %% [markdown]
# ## 4. Who is served well? Accuracy by user activity

# %%
activity = R.by_activity(per_user)
targets = (
    R.add_activity_bucket(per_user[per_user["model"] == "ease"])
    .groupby("bucket", observed=False)["n_targets"]
    .median()
    .rename("median test items")
)
activity.rename(columns=DISPLAY_NAMES).join(targets).round(4)

# %%
show(R.plot_by_activity(activity, MODELS, FIG / "week1_ndcg_by_activity.png"))

# %% [markdown]
# - **Compare models within a group, not groups with each other.** Users with short histories also have
#   very few test items (median 2, against 30 for the most active users), and NDCG behaves differently
#   with so few targets. That is why the bars are not monotone across groups.
# - **EASE leads by the widest margin for users with short histories:** +18% over the next-best model for
#   1–20 liked movies, and +17% for 21–50. Its lead shrinks to +4% for 51–200.
# - **For the most active users (>200 liked movies), UserKNN is best** (0.1177 vs 0.1124 for EASE).
# - **Different models win for different users.** This directly motivates the hybrid ranker in Week 3,
#   which can learn when to trust which model, using user activity as an input feature.

# %% [markdown]
# ## 5. BPR-MF training curve

# %%
curve = R.load_curve_from_mlflow("ml-1m", "bpr")
show(R.plot_training_curve(curve, FIG / "week1_bpr_training_curve.png"))

# %% [markdown]
# **The training loss keeps falling, but validation NDCG@10 stops improving.** It plateaus at about 0.070
# from epoch 25, peaks at epoch 65, and early stopping ends training at epoch 90. The model keeps fitting
# its training objective without generalizing further, which suggests that regularization and learning
# rate need tuning (Week 3).

# %% [markdown]
# ## Summary for the report
#
# | Question | Week 1 answer (untuned defaults) |
# |---|---|
# | Most accurate model | EASE, significantly better than every other model |
# | Do learned latent factors beat neighbourhood methods? | Not at default settings (BPR-MF < UserKNN ≈ ItemKNN) |
# | Is personalization worth it? | Yes: EASE improves NDCG@10 by 74% over MostPop |
# | Popularity bias | MostPop covers 3% of the catalog; BPR-MF covers 41% |
# | Which users gain most from EASE? | Users with short histories (+17–18% over the next-best model) |
# | Is one model best for everyone? | No: UserKNN wins for the most active users, which motivates the hybrid |
