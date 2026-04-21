#!/usr/bin/env python3
"""Lightweight docs consistency checks.

Checks:
1) Claimed public API symbols in docs/roadmap_status.md exist in bayes_sysid.__all__.
2) Artifact index in examples/artifacts/README.md matches files on disk.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "docs" / "roadmap_status.md"
ARTIFACTS_README = ROOT / "examples" / "artifacts" / "README.md"
ARTIFACTS_DIR = ROOT / "examples" / "artifacts"


def _load_api_claims() -> list[str]:
    text = ROADMAP.read_text(encoding="utf-8")
    m = re.search(
        r"<!-- DOCS_API_CLAIMS_START -->(.*?)<!-- DOCS_API_CLAIMS_END -->",
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Missing DOCS_API_CLAIMS markers in docs/roadmap_status.md")
    return re.findall(r"-\s+`([^`]+)`", m.group(1))


def _load_exported_api() -> set[str]:
    sys.path.insert(0, str(ROOT / "src"))
    import bayes_sysid  # noqa: WPS433

    return set(bayes_sysid.__all__)


def _check_api_claims() -> list[str]:
    claims = _load_api_claims()
    exported = _load_exported_api()
    errors: list[str] = []
    for symbol in claims:
        if symbol not in exported:
            errors.append(
                f"Claimed API symbol `{symbol}` in docs/roadmap_status.md is not exported in bayes_sysid.__all__."
            )
    return errors


def _read_artifact_index() -> set[str]:
    text = ARTIFACTS_README.read_text(encoding="utf-8")
    entries = re.findall(r"-\s+`([^`]+)`\s+->", text)
    return {entry for entry in entries if entry != "README.md"}


def _read_artifact_files() -> set[str]:
    files: set[str] = set()
    for path in ARTIFACTS_DIR.rglob("*"):
        if path.is_file() and path.name != "README.md":
            files.add(str(path.relative_to(ARTIFACTS_DIR)).replace("\\\\", "/"))
    return files


def _check_artifact_index() -> list[str]:
    indexed = _read_artifact_index()
    actual = _read_artifact_files()
    errors: list[str] = []

    missing_from_index = sorted(actual - indexed)
    missing_from_disk = sorted(indexed - actual)

    for item in missing_from_index:
        errors.append(f"Artifact `{item}` exists on disk but is missing from examples/artifacts/README.md index.")
    for item in missing_from_disk:
        errors.append(f"Artifact `{item}` is indexed in README but not present on disk.")

    return errors


def main() -> int:
    errors = [*_check_api_claims(), *_check_artifact_index()]
    if errors:
        print("Docs consistency check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Docs consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
