"""Guards against publishing credentials.

These tests require a git work tree; they are skipped when one is not present
(for example when the source is unpacked from an archive).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def require_git_repo() -> None:
    probe = git("rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not a git work tree")


def test_secret_files_are_git_ignored() -> None:
    require_git_repo()

    for name in (".env", ".env.production", "secrets.json"):
        result = git("check-ignore", "--quiet", name)
        assert result.returncode == 0, f"{name} must be ignored by git"


def test_env_template_is_not_ignored() -> None:
    require_git_repo()

    assert git("check-ignore", "--quiet", ".env.example").returncode == 1


def test_repository_contains_no_tracked_secrets() -> None:
    require_git_repo()

    result = subprocess.run(
        [sys.executable, "scripts/check_no_secrets.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_secret_scanner_ignores_placeholders(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.check_no_secrets import load_secrets
    finally:
        sys.path.pop(0)

    fake_env = tmp_path / ".env"
    fake_env.write_text(
        "GROQ_API_KEY=changeme\nQDRANT_API_KEY=\nAPI_KEYS=short\nSOMETHING_ELSE=abcdefghijklmnop\n",
        encoding="utf-8",
    )

    assert load_secrets(fake_env) == {}
