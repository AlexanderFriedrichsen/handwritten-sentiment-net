"""Pytest configuration shared by the whole test suite.

Ensures `handnet` and `scripts` are importable regardless of where
pytest is invoked from, and recompiles weights/weights.npz fresh once
per test session so tests never run against a stale compiled artifact.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def _compiled_weights():
    from scripts.compile_weights import main

    main()
