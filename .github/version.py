"""Read and check the package version.

The version lives in two files that have to agree, and PyPI will not let a
version be replaced once uploaded — so a disagreement cannot be corrected in
place. This exists so the agreement does not depend on remembering.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "pine_assistant" / "__init__.py"

PYPROJECT_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
INIT_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)
SOURCES = ((PYPROJECT, PYPROJECT_RE), (INIT, INIT_RE))


def read() -> dict[pathlib.Path, str]:
    found = {}
    for path, pattern in SOURCES:
        match = pattern.search(path.read_text())
        if match is None:
            sys.exit(f"no version found in {path.relative_to(ROOT)}")
        found[path] = match.group(2)
    return found


def check(expected: str) -> None:
    """Fail unless both files carry `expected`.

    Guards a tag build: a tag naming one version while the package declares
    another publishes something no one can find, and it cannot be corrected in
    place.
    """
    mismatched = {p: v for p, v in read().items() if v != expected}
    if mismatched:
        for path, version in mismatched.items():
            print(f"{path.relative_to(ROOT)}: {version} (tag says {expected})", file=sys.stderr)
        sys.exit("version mismatch")
    print(f"version {expected} agrees across {len(SOURCES)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", metavar="VERSION", help="fail unless both files say VERSION")
    group.add_argument("--show", action="store_true", help="print the current version")
    args = parser.parse_args()

    if args.check:
        check(args.check)
    else:
        print(next(iter(read().values())))


if __name__ == "__main__":
    main()
