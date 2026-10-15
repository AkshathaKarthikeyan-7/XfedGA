"""Three-way server splits and fixed subsamples: disjointness, patient/case level, determinism."""
import os

import pytest

from data.splits import (ROLES, busbra_patient_split, decathlon_case_split, evenly_spaced_per_group,
                         group_split, one_per_group_subsample, random_subsample)
from tests.conftest import require_path

BUSBRA = "raw data/BUS-BRA"
DECATHLON = "raw data/Task05_Prostate/Task05_Prostate"


def _assert_partition(split, universe):
    sets = [set(split[r]) for r in ROLES]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not sets[i] & sets[j], f"{ROLES[i]} and {ROLES[j]} overlap"
    assert set().union(*sets) == set(universe)


def test_decathlon_split_blocks_synthetic():
    cases = [(f"c{i:02d}", "decathlon") for i in range(32)]
    sizes = {c: 5 + (i * 7) % 23 for i, (c, _) in enumerate(cases)}
    s1 = decathlon_case_split(cases, sizes, seed=0)
    assert [len(s1[r]) for r in ROLES] == [8, 8, 16]
    _assert_partition(s1, cases)
    assert s1 == decathlon_case_split(cases, sizes, seed=0)
    assert s1 != decathlon_case_split(cases, sizes, seed=1)


def test_group_split_keeps_groups_together():
    items = [f"P{p}_{k}" for p in range(10) for k in range(3)]
    train, val = group_split(items, lambda s: s.split("_")[0], val_fraction=0.2, seed=3)
    assert {s.split("_")[0] for s in train}.isdisjoint({s.split("_")[0] for s in val})
    assert len({s.split("_")[0] for s in val}) == 2 and len(train) + len(val) == len(items)


def test_subsamples_are_fixed():
    items = [f"P{p}_{k}" for p in range(50) for k in range(2)]
    a = one_per_group_subsample(items, lambda s: s.split("_")[0], 40, seed=7)
    assert a == one_per_group_subsample(items, lambda s: s.split("_")[0], 40, seed=7)
    assert len({items[i].split("_")[0] for i in a}) == 40
    assert random_subsample(500, 40, seed=1) == random_subsample(500, 40, seed=1)
    assert random_subsample(30, 40) == list(range(30))
    idx = evenly_spaced_per_group([15, 12, 20, 9, 14, 16, 11, 18], 40)
    assert len(idx) == len(set(idx)) == 40


@pytest.mark.data
def test_busbra_split_patient_level_and_proportions():
    require_path(BUSBRA)
    from data.breast_dataset import INDEXERS
    pairs = INDEXERS["BUS-BRA"](BUSBRA)
    split, patients, patient_of = busbra_patient_split(pairs, os.path.join(BUSBRA, "BUSBRA", "bus_data.csv"), seed=42)
    _assert_partition({r: [p[0] for p in split[r]] for r in ROLES}, [p[0] for p in pairs])
    _assert_partition(patients, set(patient_of.values()))
    n = len(set(patient_of.values()))
    for role, frac in [("fitness", 0.15), ("validation", 0.25), ("test", 0.60)]:
        assert abs(len(patients[role]) / n - frac) < 0.02, role
    for role in ROLES:  # every image of a patient sits in that patient's role
        pids = set(patients[role])
        assert all(patient_of[os.path.splitext(os.path.basename(p[0]))[0]] in pids for p in split[role])


@pytest.mark.data
def test_decathlon_split_real_cases():
    require_path(DECATHLON)
    from data.prostate_dataset import INDEXERS, prostate_cache
    cases = INDEXERS["Decathlon"](DECATHLON)
    cache = prostate_cache("Decathlon", DECATHLON)
    sizes = {c: len(cache.rows(c)) for c, _ in cases}
    split = decathlon_case_split(cases, sizes, seed=42)
    assert [len(split[r]) for r in ROLES] == [8, 8, 16]
    _assert_partition(split, cases)


@pytest.mark.data
@pytest.mark.parametrize("task", ["prostate", "breast"])
def test_split_summary_independent_of_subsample_settings(task):
    """results/splits/<task>_dataseed<k>.json must be identical for every run with the same data seed."""
    require_path("raw data/PROMISE12" if task == "prostate" else BUSBRA)
    import json
    from experiments.config import load_config
    from experiments.tasks import build_task, split_summary
    base = load_config(task)
    tiny = load_config(task, ["fitness.subsample=8", "saliency.subsample=4", "evaluation.explain_samples=8",
                              "aggregation.beta=1.0", "train.rounds=1"])
    a, b = split_summary(build_task(task, base)), split_summary(build_task(task, tiny))
    assert a == b
    with open(f"results/splits/{task}_dataseed{base['seed']}.json", encoding="utf-8") as f:
        assert json.load(f) == json.loads(json.dumps(a))
