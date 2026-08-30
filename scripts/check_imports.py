#!/usr/bin/env python
"""Verify that navi's declared dependencies cover everything lib/ imports.

Run this with navi's OWN interpreter, never a consumer's:

    python scripts/check_imports.py

navi-3.14.7 holds exactly the dependencies pyproject.toml declares and nothing
else. A consumer env is useless for this check -- alef-3.14.7 carries ~145
packages against navi's ~62, so an undeclared import sails through there and
only fails for whoever installs navi next.

Exits non-zero and names the offenders if any module fails to import.
"""

import importlib
import pkgutil
import sys
import warnings
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent / "lib"


def main() -> int:
    warnings.simplefilter("ignore")

    # walk_packages imports each package to recurse into it, so lib itself has to
    # resolve first. sys.path[0] is scripts/, not the repo root, which means this
    # deliberately exercises the INSTALLED package rather than the source tree.
    try:
        importlib.import_module("lib")
    except ModuleNotFoundError:
        print(f"navi is not installed for {sys.executable}.\n\n"
              "Run this with navi's own interpreter, which has it editable-installed:\n"
              "    pyenv exec python scripts/check_imports.py    # from the navi repo root\n"
              "or install it there first:  pip install -e .", file=sys.stderr)
        return 2

    names = sorted(m.name for m in pkgutil.walk_packages([str(LIB)], prefix="lib."))
    if not names:
        print(f"no modules found under {LIB}", file=sys.stderr)
        return 2

    failures = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as err:                      # noqa: BLE001 - report anything
            failures.append((name, f"{type(err).__name__}: {err}"))

    if failures:
        print(f"{len(failures)} of {len(names)} lib modules failed to import "
              f"using {sys.executable}\n")
        for name, err in failures:
            print(f"  {name}\n      {err}")
        print("\nEither the import is wrong, or pyproject.toml is missing a dependency.")
        return 1

    print(f"ok: all {len(names)} lib modules import under navi's declared dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
