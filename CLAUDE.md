# navi — shared library (package `lib`)

@../sefer/overview.md
@../sefer/conventions.md

The shared code library: data **clients** + **model code** (indicators,
strategies, estimators, stats) + plot/db utils. Consumers install it editable
(`-e ../navi`), so `import lib …` resolves here. Reference docs live in
`sefer/navi/` (as they arrive).

## Ownership

navi holds no tests of its own; each area is developed and tested by the repo
that owns it.

| Area | Owner | Tests live in |
| --- | --- | --- |
| `lib/data`, `lib/models`, `lib/plots`, `lib/trading` | alef | `alef/tests/` |
| `lib/clients` | meida | `meida/tests/` |

Loose modules (`config`, `env`, `logger`, `stats`, `utils`, `mcp_client`) are
shared infrastructure and belong to whoever is changing them — say so in the
commit rather than assuming.

Changes to an owned area are prototyped and validated in the owning repo first;
`yada` consumes the result and does not own library code.

## Environment & commands

- pyenv env `navi-3.14.7` (`.python-version`), also what `pyrightconfig.json`
  type-checks against. **Keep it minimal.** It holds exactly the dependencies
  `pyproject.toml` declares — ~62 packages against alef's ~145 — and that is the
  point: an import navi has not declared fails here while sailing through a
  consumer env, so do not pip-install extras into it.
- `python scripts/check_imports.py` imports every `lib` module and fails if one
  needs something `pyproject.toml` does not declare. Run it after adding an
  import or editing dependencies. Exit 0 pass, 1 drift, 2 could not run.
- navi pins nothing and has no lockfile; consumers pip-compile their own.

## Notes

- Packaged via `pyproject.toml`; the importable package is `lib` (not `navi`).
- **navi declares the shared third-party stack** (numpy, pandas, scipy,
  statsmodels, matplotlib, …) so consumers inherit it through `-e ../navi`
  instead of hand-listing it, which is how alef, yada and meida drifted onto
  different statsmodels and pandas versions. A new third-party import belongs in
  `pyproject.toml` here, not in a consumer's `requirements.in`.
- The MCP client is optional (`-e ../navi[mcp]`); `lib/__init__.py` imports
  `mcp_client` inside a try/except and sets it to `None` when absent.
