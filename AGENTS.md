# Oceananigans.jl

Julia package for ocean-flavored fluid dynamics on CPUs and GPUs (CUDA, AMD, Metal, OneAPI).
Every model runs the same source on every architecture through KernelAbstractions, so most of
what follows is about keeping code correct on a GPU that you are probably not testing on.

## Commands

```sh
# Run tests on CPU, selected by prefix of test/<group>/<name>.jl
CUDA_VISIBLE_DEVICES=-1 TEST_ARCHITECTURE=CPU julia --project -e 'using Pkg; Pkg.test("Oceananigans"; test_args=["unit/grids"])'

# Explicit imports and Aqua checks (run after any change to src/)
CUDA_VISIBLE_DEVICES=-1 TEST_ARCHITECTURE=CPU julia --project -e 'using Pkg; Pkg.test("Oceananigans"; test_args=["unit/quality_assurance"])'

# Trailing whitespace and blank lines at end of file (CI also requires exactly one final newline)
git diff --check origin/main
```

`Pkg.test(; test_args=["--list"])` lists every test with its last duration. The full suite is
large; run the files closest to the change (the `/run-tests` skill has a mapping). Docstring
examples are doctests and run in the documentation build.

## Before you change these, ask

- **Regression reference data and tolerances** (`test/regression/`, `test/setup/data_dependencies.jl`).
  A failing regression test is evidence of a behavior change. Find the cause; do not regenerate the
  data or loosen the tolerance to make it pass. When maintainers do regenerate it, the
  `regression_truth_data_vN` suffix must be bumped, because DataDeps never re-downloads a cached
  name and CI keeps a persistent depot.
- **`[deps]` and `[weakdeps]` in `Project.toml`**. They change load time and CI for every
  downstream package. Touch `[compat]` only when asked.
- **Exported names and keyword arguments of public constructors**. Downstream packages (ClimaOcean,
  Breeze, NumericalEarth) and user scripts depend on them.

## Verifying your work

- Read the current definition of anything you call (`@which`, `methods`, or the source under
  `src/`), including Oceananigans' own API. It changes quickly and remembered signatures go stale.
- A test that fails on your branch is yours until you reproduce the same failure on `main`.
- Report results by quoting the test summary line. An exit code alone is not a pass.
- If a fix makes a failing test run but you cannot explain why it was failing, the fix is
  probably wrong. Revisit the change that broke it.
- GPU "dynamic invocation error": rerun on CPU. If it passes there, the cause is almost always a
  type instability that the CPU tolerates.
- `UndefVarError` or load failures right after pulling are usually a stale `Manifest.toml`;
  re-resolve the environment before debugging code.

## Conventions that are not visible from the code

- Model constructors take `grid` positionally and everything else as keywords:
  `NonhydrostaticModel(grid; closure=nothing)`. Omit the `;` when there are no keywords.
  (`ShallowWaterModel(grid, gravitational_acceleration; ...)` is the one exception.)
- Source code uses explicit imports, checked by `unit/quality_assurance`. Examples, docs, and
  tests use `using Oceananigans`; if a common name is not exported, consider exporting it rather
  than importing it in the script.
- Backend-specific code goes in `ext/` and is selected by dispatch, not by `if` branches in `src/`.
  Move data between architectures with `on_architecture`, not `Array(...)` / `CuArray(...)`.
- Delete commented-out code and debugging leftovers; git keeps the history. Comments describe the
  code that is there, not how it got there.
- Never extend `getproperty` to make an undefined-property error go away; fix the caller.
- A "type is not callable" error usually means a local variable shadows a function name.
- Keep a PR to one concern. Unrelated cleanup goes in its own PR.

## Where to look

Rules in `.claude/rules/` load automatically in Claude Code when you edit matching files. Other
agents should read the one that matches the task:

| Task | Read |
|------|------|
| Writing or editing kernels, operators, or anything in `src/` | `.claude/rules/kernel-rules.md` |
| Naming, notation, comments | `.claude/rules/style-rules.md` |
| Docstrings | `.claude/rules/docstring-rules.md` |
| Tests | `.claude/rules/testing-rules.md` |
| Docs pages | `.claude/rules/docs-rules.md` |
| Examples | `.claude/rules/examples-rules.md` |

Skills (`.claude/skills/`): `/run-tests`, `/build-docs`, `/new-simulation`, `/babysit-ci`.
