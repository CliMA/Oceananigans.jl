---
name: babysit-ci
description: Monitor CI, auto-fix small issues, pause on bigger problems, retrigger flaky runs
user_invocable: true
---

# Babysit CI

Monitor GitHub Actions CI for the current branch/PR. Auto-fix small issues (typos, missing imports,
whitespace, formatting). Pause and describe anything that requires judgment. Retrigger flaky jobs
with empty commits.

## Step 1: Find the CI Run

```sh
# Get the current branch
git branch --show-current

# Find the latest CI run for this branch
gh run list --branch $(git branch --show-current) --limit 5

# Or for a specific PR
gh pr checks <PR_NUMBER>
```

## Step 2: Monitor Loop

Check CI status. For each failed job:

```sh
# View the run summary
gh run view <RUN_ID>

# Get logs for a failed job
gh run view <RUN_ID> --log-failed
```

## Step 3: Triage Each Failure

Classify the failure and act accordingly:

### Auto-fix (commit and push without asking)

These are mechanical issues with obvious fixes:

| Failure | How to fix |
|---------|-----------|
| **Whitespace check** | Run `.julia/contrib/check-whitespace.jl` logic: remove trailing whitespace, ensure final newline, no trailing blank lines |
| **Missing explicit import** | `ExplicitImports` error in `test_quality_assurance.jl` — add the missing `using`/`import` to the appropriate file |
| **Aqua.jl ambiguities** | Add the missing method disambiguation |
| **Doctest output mismatch** | Update the expected output in the docstring to match actual output |
| **Typo in error message or docstring** | Fix the typo |
| **Unused import warning** | Remove the unused import |

After fixing, commit with a descriptive message and push:

```sh
git add <specific files>
git commit -m "Fix <description of mechanical issue>"
git push
```

Then continue monitoring from Step 2.

### Retrigger (likely flaky)

These failures are often transient and not caused by the PR:

| Signal | Action |
|--------|--------|
| Test passed locally but fails in CI | Retrigger |
| Timeout with no test failure | Retrigger |
| Network/download error | Retrigger |
| `Pkg.instantiate` failure | Retrigger |
| CI infrastructure error | Retrigger |
| A job that failed but is unrelated to the changed files | Retrigger |

Retrigger with:

```sh
# Re-run just the failed jobs
gh run rerun <RUN_ID> --failed

# Or push an empty commit to retrigger all jobs
git commit --allow-empty -m "Retrigger CI"
git push
```

### Pause and describe (needs judgment)

Stop and explain the problem to the user for **any** of these:

- **Test logic failure**: a test assertion fails (expected vs actual mismatch) that isn't a doctest output update
- **Type instability or GPU error**: "dynamic invocation" or similar — may indicate a real bug
- **Regression test failure**: numerical results don't match stored references
- **New test failures unrelated to the PR changes**: could indicate a flaky test or upstream breakage — explain both possibilities
- **Build failure**: package won't precompile or load
- **Multiple interrelated failures**: several jobs failing in ways that suggest a common root cause

When pausing, report:
1. Which job(s) failed
2. The relevant error output (trimmed to the key lines)
3. Your assessment of the likely cause
4. Suggested fix options if you have ideas

## Step 4: Confirm All Green

Once all jobs pass:

```sh
gh pr checks <PR_NUMBER>
```

Report the final status to the user.

## Notes

- Always check `gh run view <RUN_ID> --log-failed` before acting — don't guess from job names alone
- After pushing a fix, wait for CI to pick it up before checking again (use `gh run list` to find the new run)
- The CI workflow has `cancel-in-progress` for PRs, so a new push cancels the old run automatically
- CI jobs: Whitespace, test groups (sharding, mpi_tripolar, distributed_output, turbulence_closures,
  makie, reactant, metal), each running on Julia 1.12
- Never force-push or rewrite history to fix CI — always add new commits
