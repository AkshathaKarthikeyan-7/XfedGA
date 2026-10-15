"""
Seed determinism of the custom loop, and the precondition for the elite-fitness cache (Q7).
Slow (GPU, real data): one tiny round per run.
"""
import pytest
import torch

from tests.conftest import require_path

TINY = ["train.rounds=1", "train.local_epochs=1", "ga.population=2", "ga.generations=1",
        "fitness.subsample=8", "saliency.subsample=4", "evaluation.explain_samples=8"]


def _run(tmp_path, method, seed):
    from experiments.run_federated import run_experiment
    return run_experiment("prostate", method, seed, TINY, out_root=str(tmp_path), evaluate_test=False)


@pytest.mark.data
@pytest.mark.slow
def test_same_seed_same_model_and_metrics(tmp_path):
    require_path("raw data/PROMISE12")
    a = _run(tmp_path / "a", "xfedga", 3)
    b = _run(tmp_path / "b", "xfedga", 3)
    assert all(torch.equal(a["final_state"][k], b["final_state"][k]) for k in a["final_state"])
    for key in ["val_dsc", "val_hd95_norm", "val_ec", "val_pg", "q_i", "alpha_i", "ga_best_per_gen"]:
        assert a["rounds"][0][key] == b["rounds"][0][key], key


@pytest.mark.data
@pytest.mark.slow
def test_different_seed_different_model(tmp_path):
    require_path("raw data/PROMISE12")
    a = _run(tmp_path / "a", "fedavg", 3)
    b = _run(tmp_path / "b", "fedavg", 4)
    assert not all(torch.equal(a["final_state"][k], b["final_state"][k]) for k in a["final_state"])


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("kind", ["dsc", "multi"])
def test_fitness_evaluation_is_repeatable(kind):
    """Q7: the elite cache is only valid if scoring the same candidate twice gives the same fitness."""
    require_path("raw data/Task05_Prostate/Task05_Prostate")
    from experiments.config import load_config, seed_everything
    from experiments.tasks import build_task
    from fedga.evaluation import FixedSet, build_fitness_fn
    from models.residual_unet import build_residual_unet
    seed_everything(0)
    cfg = load_config("prostate")
    data = build_task("prostate", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed = FixedSet(data.server["fitness"].dataset, data.fitness_subsample, device)
    model = build_residual_unet().to(device)
    fn = build_fitness_fn(model, fixed, kind, cfg["fitness"]["weights"], cfg["fitness"]["hd95_normalizer"],
                          cfg["fitness"]["cam_threshold"])
    state = {k: v.clone() for k, v in model.state_dict().items()}
    first = fn(state)
    assert all(fn(state) == first for _ in range(3))
