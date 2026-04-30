"""Refuse a pyproject.toml version bump when the matching git tag does not exist.

Wired into pre-commit on changes to pyproject.toml. Prevents the v1.1.x → v1.3.0
silent-publish-skip pattern where the version field was bumped repeatedly but no
git tag was ever pushed, so .github/workflows/publish.yml never fired.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def current_version() -> str:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    version = current_version()
    tag = f"v{version}"
    if tag_exists(tag):
        return 0
    print(
        f"ERROR: pyproject.toml is at {version!r} but git tag {tag!r} does not exist.\n"
        "Create and push the tag together with this commit, e.g.:\n"
        f"  git tag {tag}\n"
        f"  git push origin <branch> {tag}\n"
        "Or revert the version bump.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
