"""
FedGA (baseline) and XFedGA must score the SAME fixed fitness subsample (same image ids, same
count = fitness.subsample) on both tasks: the old scripts scored the full fitness set for the
baseline and 40 images for XFedGA (audit weakness 6), a confound this test guards against.
Slow (GPU, real data): one tiny round per method and task.
"""
import json
import os

import pytest

from tests.conftest import require_path

TINY = ["train.rounds=1", "train.local_epochs=1", "ga.population=2", "ga.generations=1",
        "saliency.subsample=4", "evaluation.explain_samples=8"]


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("task", ["prostate", "breast"])
def test_fedga_and_xfedga_score_the_same_fitness_subsample(task, tmp_path, monkeypatch):
    require_path("raw data/PROMISE12" if task == "prostate" else "raw data/BUS-BRA")
    import experiments.run_federated as rf
    from experiments.config import load_config
    seen = {}
    real = rf.build_fitness_fn

    def spy(model, fixed, kind, *args, **kwargs):
        seen.setdefault(kind, []).append(list(fixed.indices))
        assert len(fixed) == len(fixed.images) == len(fixed.masks)
        return real(model, fixed, kind, *args, **kwargs)
    monkeypatch.setattr(rf, "build_fitness_fn", spy)

    ids = {}
    for method, kind in [("fedga", "dsc"), ("xfedga", "multi")]:
        res = rf.run_experiment(task, method, 1, TINY, out_root=str(tmp_path / method), evaluate_test=False)
        with open(os.path.join(res["out"], "split_log.json"), encoding="utf-8") as f:
            ids[method] = json.load(f)["fitness_subsample"]
        assert res["rounds"][0]["fitness_evals"] > 0

    n = load_config(task)["fitness"]["subsample"]
    assert seen["dsc"][0] == seen["multi"][0], "the two methods received different fitness subsamples"
    assert len(seen["dsc"][0]) == n == len(ids["fedga"])
    assert ids["fedga"] == ids["xfedga"]
    assert len(set(ids["fedga"])) == n
