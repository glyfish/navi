# navi — shared library (package `lib`)

@../sefer/overview.md
@../sefer/conventions.md

The shared code library: data **clients** + **model code** (indicators,
strategies, estimators, stats) + plot/db utils. Consumers install it editable
(`-e ../navi`), so `import lib …` resolves here. Reference docs live in
`sefer/navi/` (as they arrive).

## Notes

- Packaged via `pyproject.toml`; the importable package is `lib` (not `navi`).
- Tests live with the consumers — client tests in meida, the rest in yada as the
  library comes into use. Keep that split.
