"""Verifies every file in freeze.json's manifest against its recorded
SHA-256, and the manifest hash itself, exactly per freeze.json's own
documented `manifest_recipe`.

Every Phase 2 script (scripts/audit_labels.py, scripts/eval_handnet.py,
scripts/run_baselines.py) imports and calls verify_freeze() before
touching a label or running the net, so a silent drift in any frozen
file -- including an accidental re-write of weights/weights.npz by
conftest.py's session fixture -- is caught immediately rather than
laundered into a reported result.

Run directly: python scripts/verify_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_JSON = ROOT / "freeze.json"


class FreezeViolation(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(root: Path = ROOT, quiet: bool = False) -> str:
    """Raises FreezeViolation if any frozen file, or the manifest hash
    over all of them, doesn't match freeze.json. Returns the verified
    manifest hash on success.
    """
    manifest = json.loads((root / "freeze.json").read_text())
    files = manifest["files"]

    mismatches: list[str] = []
    lines: list[str] = []
    for rel_path, expected_hash in files.items():
        full_path = root / rel_path
        if not full_path.exists():
            mismatches.append(f"{rel_path}: MISSING (expected {expected_hash})")
            continue
        actual_hash = _sha256(full_path)
        lines.append(f"{rel_path} {actual_hash}")
        if actual_hash != expected_hash:
            mismatches.append(
                f"{rel_path}: hash mismatch\n"
                f"    expected {expected_hash}\n"
                f"    actual   {actual_hash}"
            )

    if mismatches:
        raise FreezeViolation(
            "FREEZE VIOLATION -- one or more frozen files do not match "
            "freeze.json:\n" + "\n".join(mismatches)
        )

    # Recompute the manifest hash per freeze.json's own "manifest_recipe":
    # sort "<path> <hex_digest>" lines lexicographically by path, join with
    # a single newline between lines and a trailing newline, sha256 that.
    sorted_lines = sorted(lines, key=lambda line: line.split(" ", 1)[0])
    manifest_bytes = ("\n".join(sorted_lines) + "\n").encode("utf-8")
    actual_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_hash = manifest["manifest_sha256"]
    if actual_manifest_hash != expected_manifest_hash:
        raise FreezeViolation(
            "FREEZE VIOLATION -- manifest hash mismatch "
            f"(expected {expected_manifest_hash}, got {actual_manifest_hash}) "
            "even though every individual file hash matched freeze.json. "
            "This should be impossible -- check this function against "
            "freeze.json's manifest_recipe for a documentation drift."
        )

    if not quiet:
        print(f"freeze verified OK: {len(files)}/{len(files)} files match, manifest {actual_manifest_hash}")
    return actual_manifest_hash


def main() -> int:
    try:
        verify_freeze()
    except FreezeViolation as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
