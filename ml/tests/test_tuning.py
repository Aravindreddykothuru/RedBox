from __future__ import annotations

import optuna
import pytest

from recsys.models import get_model_class
from recsys.tuning.spaces import SPACES
from recsys.tuning.tuner import default_trial_params


@pytest.mark.parametrize("model_class", sorted(SPACES))
def test_default_trial_is_inside_the_search_space(model_class):
    """Trial 0 replays the paper defaults; every default value must be a legal point of the space."""
    params = default_trial_params(model_class, {})
    dist_trial = optuna.trial.FixedTrial(params)
    suggested = SPACES[model_class](dist_trial)  # raises if a value is outside a categorical's choices
    defaults = get_model_class(model_class).defaults
    for name, value in suggested.items():
        assert value == defaults[name], f"{model_class}.{name}: trial 0 would not reproduce the default"


def test_default_trial_respects_config_overrides():
    params = default_trial_params("lightgcn", {"layers": 2, "lr": 0.003})
    assert params["layers"] == 2 and params["lr"] == 0.003


def test_space_values_are_within_ranges():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(30):
        trial = study.ask()
        p = SPACES["ials"](trial)
        assert 0.5 <= p["alpha"] <= 100 and 1e-2 <= p["reg"] <= 1000 and p["dim"] in (32, 64, 128, 256)
        study.tell(trial, 0.0)
