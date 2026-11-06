"""Tests for scripts/verify_freeze.py: confirms it passes against the
real, untouched freeze, and -- the actual point of this gate -- that it
actually detects tampering rather than rubber-stamping anything.
"""

import json
import os
import shutil

import pytest

from scripts.verify_freeze import FreezeViolation, verify_freeze

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def test_real_freeze_verifies_clean():
    manifest_hash = verify_freeze(quiet=True)
    expected = json.loads(open(os.path.join(ROOT, "freeze.json")).read())["manifest_sha256"]
    assert manifest_hash == expected


def test_detects_modified_frozen_file(tmp_path):
    _copy_repo_subset(tmp_path)
    target = tmp_path / "handnet" / "columns.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    with pytest.raises(FreezeViolation, match="hash mismatch"):
        verify_freeze(root=tmp_path, quiet=True)


def test_detects_missing_frozen_file(tmp_path):
    _copy_repo_subset(tmp_path)
    (tmp_path / "handnet" / "columns.py").unlink()
    with pytest.raises(FreezeViolation, match="MISSING"):
        verify_freeze(root=tmp_path, quiet=True)


def test_detects_tampered_manifest_hash(tmp_path):
    _copy_repo_subset(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    manifest = json.loads(manifest_path.read_text())
    real_hash = list(manifest["files"].values())[0]
    manifest["files"][next(iter(manifest["files"]))] = real_hash[:-1] + ("0" if real_hash[-1] != "0" else "1")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FreezeViolation, match="hash mismatch"):
        verify_freeze(root=tmp_path, quiet=True)


def _copy_repo_subset(tmp_path) -> None:
    """Copies freeze.json plus every frozen file (per freeze.json's own
    manifest) into tmp_path, preserving relative layout -- a disposable
    sandbox so these tests can tamper freely without touching the real
    frozen artifacts.
    """
    manifest = json.loads(open(os.path.join(ROOT, "freeze.json")).read())
    shutil.copy(os.path.join(ROOT, "freeze.json"), tmp_path / "freeze.json")
    for rel_path in manifest["files"]:
        src = os.path.join(ROOT, rel_path)
        dst = tmp_path / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
