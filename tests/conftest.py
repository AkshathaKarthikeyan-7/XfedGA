"""Shared test setup: repo root on sys.path and as working directory (configs use relative paths)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)


def require_path(path):
    if not os.path.exists(os.path.join(REPO, path)):
        pytest.skip(f"raw data not present: {path}")
