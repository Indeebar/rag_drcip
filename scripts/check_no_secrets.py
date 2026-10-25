#!/usr/bin/env python
"""Pre-commit guard: fail if a credential is (or is about to be) tracked by git.

Checks, in order:

1. No secret-looking file (``.env``, ``.env.production``, ``*.pem``, ``*.key``) is tracked.
2. No tracked file contains the literal value of a secret found in ``.env``.

Only variable *names* and file paths are printed -- never secret values.
Exit code 0 = clean, 1 = problem found, 2 = could not inspect the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

SECRET_VARS = (
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "API_KEYS",
    "NEON_DATABASE_URL",
    "DATABASE_URL",
)

ALLOWED_DOTFILES = {".env.example"}

FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("not a git repository (git ls-files failed)")
    return [name for name in result.stdout.decode("utf-8", "replace").split("\0") if name]


def load_secrets(path: Path) -> dict[str, str]:
    """Read secret values from a local .env, skipping obvious placeholders."""
    if not path.is_file():
        return {}
    secrets: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name not in SECRET_VARS:
            continue
        if len(value) < 12 or value.lower().startswith(("changeme", "replace", "your-", "<")):
            continue
        secrets[name] = value
    return secrets


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def scan() -> list[str]:
    findings: list[str] = []
    files = tracked_files()

    for name in files:
        path = Path(name)
        if path.name in ALLOWED_DOTFILES:
            continue
        if path.name == ".env" or path.name.startswith(".env."):
            findings.append(f"secret file is tracked: {name}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"key material is tracked: {name}")

    secrets = load_secrets(ENV_FILE)
    if secrets:
        for name in files:
            path = REPO_ROOT / name
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if is_binary(data):
                continue
            text = data.decode("utf-8", "replace")
            for variable, value in secrets.items():
                if value in text:
                    findings.append(f"tracked file {name} contains the value of {variable}")

    return findings


def main() -> int:
    try:
        files = tracked_files()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"FAIL  cannot inspect repository: {exc}")
        return 2

    findings = scan()
    print(f"scanned {len(files)} tracked file(s); .env present: {ENV_FILE.is_file()}")
    if findings:
        print("FAIL  secret material would be published:")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print("PASS  no secret files or secret values are tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
