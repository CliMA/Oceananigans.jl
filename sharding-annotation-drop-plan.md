# Plan: MWE for sharding-annotation drop on `reinstate-enum-bias`

## Context

PR #5450 (`reinstate-enum-bias`) reapplies #5318: WENO/upwind biases switched from
`struct LeftBias end / struct RightBias end` (singleton types) to `@enum Bias LeftBias RightBias`
(enum values). The job
[Sharding - Julia 1.11.9 - ubuntu-24.04-arm](https://github.com/CliMA/Oceananigans.jl/actions/runs/25348010780/job/74321260846)
fails: `test/test_sharded_lat_lon.jl` is OOM-killed on a 16 GB ARM Linux runner during
Reactant compilation.

Per [@giordano on PR #5450](https://github.com/CliMA/Oceananigans.jl/pull/5450#issuecomment-4380103554):
**high memory use is a side effect — the actual bug is sharding annotations
disappearing from the final XLA-compiled MLIR module, on Linux only.** Capping
address space with `prlimit --as=12884901888 julia ...` (12 GB) is the easiest
way to surface it.

## Confirmed bug fingerprint (from CI artifact `sharding-mlir-1.11.9-ubuntu-24.04-arm-default`)

The time-stepping compile dumps a sequence of `*_post_xla_compile.mlir` modules.
Comparing the second-to-last and last:

| Module                                            | Bytes   | `sdy.sharding_per_value` | `mhlo.sharding` | `<@mesh` |
| ------------------------------------------------- | ------- | ------------------------ | --------------- | -------- |
| `module_699_reactant_tokw_post_xla_compile.mlir`  | 145,732 | **539**                  | 592             | 568      |
| `module_719_reactant_tokw_post_xla_compile.mlir`  | 118,213 | **2**                    | 27              | 27       |

In `module_719` only the function argument/return shardings and one trailing
`xla.sdy.FuncResultSharding` custom call survive. The ~567 internal
`sdy.sharding_per_value` annotations on intermediate `stablehlo` ops — the ones
that tell XLA how to actually shard the body of the computation — are gone.

Workload at point of bug: `tensor<20x50x50xf64>` (the 40×40×10 grid with
halo=(5,5,5) → 50×50×20), mesh `<"x"=2, "y"=2>`, 4 virtual devices via
`--xla_force_host_platform_device_count=4`.

**Test signal we can use programmatically:**

```bash
grep -c 'sdy.sharding_per_value' <last *_post_xla_compile.mlir>
# healthy: hundreds       buggy: ~2
```

## Hypothesis

The enum-bias change moves bias dispatch from `bias isa LeftBias` (compile-time
type check, eliminated to a constant by the compiler) to `bias == LeftBias`
(runtime equality on an `@enum` value). Suspects, in order of likelihood:

1. **A type-stability or specialization regression** changes how Reactant traces
   the WENO bodies, producing MLIR with sharding annotations attached in a
   pattern (or on op kinds) that an XLA pass does not preserve through to the
   end of the pipeline. Linux-only because XLA pipelines / partitioner versions
   differ across platforms.
2. **The new `bias == LeftBias` form is traced as a runtime branch** rather than
   selected at trace time, producing strictly larger MLIR with annotations on
   `select`/`compare` ops that the partitioner then strips.
3. **Resource pressure** (more inference, more inlining) is itself the trigger —
   under tight memory the partitioner skips an annotation-propagating pass.
   `prlimit` alone surfaces the bug, supporting at least partial truth here.

We expect to learn which is correct from Phase B/C below.

## Reproduction environment

- **OS:** Linux (any distro, x86_64 or aarch64). The macOS dev box does **not**
  reproduce.
- **Memory cap:** `prlimit --as=12884901888` (12 GB address space).
- **Julia:** 1.11.9 (matches CI). Other 1.11.x is probably fine.
- **Single-process sharding:** `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  (already set inside `test/run_sharding_tests.jl`).
- No GPU needed.

The bug-detection signal is **MLIR content**, not whether the process OOMs, so
even a machine with plenty of RAM can reproduce it as long as `prlimit` caps
the address space (per giordano's comment).

## Plan

### Phase A — confirm we can reproduce on the chosen Linux box

1. Clone Oceananigans.jl, check out `reinstate-enum-bias`, instantiate
   `test/Project.toml`. The sharding test does not need GPU.
2. Run, with debug dumps and the memory cap:
   ```bash
   cd test
   prlimit --as=12884901888 julia --project=. -O0 run_sharding_tests.jl
   ```
   `run_sharding_tests.jl` already sets `Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true`,
   which writes per-pass MLIR into `$TMPDIR/reactant_*/`.
3. Find the last `*_post_xla_compile.mlir` of the `tokw` compile (the
   `time_step!` compile is the largest one) and:
   ```bash
   ls -S tmp/reactant_*/module_*_reactant_tokw_post_xla_compile.mlir | head
   grep -c 'sdy.sharding_per_value' <that file>
   ```
   Expect ~2 → confirms repro.
4. Run the same on `main` (after copying `run_sharding_tests.jl` /
   `sharding_test_utils.jl` if needed — they exist on main, just aren't wired
   into CI yet) and confirm it produces hundreds of `sdy.sharding_per_value`.

**Exit criterion:** one `grep -c` command produces a single integer that
distinguishes `main` from `reinstate-enum-bias` reproducibly.

### Phase B — narrow the workload to a minimal reproducing example

Starting from `sharding_test_model` in `test/sharding_test_utils.jl`, ablate
*one knob at a time* on the enum-bias branch. After each change, rerun
`run_sharding_tests.jl` and re-grep the annotation count. Order of removal
(easiest first):

1. Drop `coriolis = HydrostaticSphericalCoriolis()`.
2. Drop the tracer entirely (`tracers = ()`, no `tracer_advection`).
3. Replace `momentum_advection = WENOVectorInvariant(order=3)` with
   `momentum_advection = nothing` (or a centered scheme).
4. Replace `tracer_advection = WENO()` with `tracer_advection = Centered()`
   (do this in conjunction with #2 only if tracers stay).
5. Replace `SplitExplicitFreeSurface(grid; substeps=20)` with
   `ExplicitFreeSurface()` or `nothing`.
6. Shrink the grid: `size=(20, 20, 4)`, smaller halos where each scheme allows.
7. Reduce the partition (e.g. `Partition(2, 1)` → `Partition(1, 2)` → 1 device)
   to find the smallest mesh that still drops annotations. Note: this changes
   the bug surface — keep at least 2 devices.

Stop ablating as soon as a setup *no longer* reproduces (then back off one
step) or the model is irreducibly small.

**Hypothesis check:** if removing WENO (steps 3 & 4) makes the bug vanish, the
enum-bias change is implicated directly. If it survives without WENO, the
trigger is elsewhere — re-look at what enum-bias actually changed (`git diff
main..reinstate-enum-bias -- src/Advection`).

**Exit criterion:** smallest config (model + grid + partition) that still drops
annotations, captured as a self-contained `.jl` file.

### Phase C — locate the pass that strips the annotations

Once we have an MWE, walk the per-pass MLIR dumps in
`tmp/reactant_*/module_*_reactant_tokw_*.mlir`, sorted by module ID, and grep
the annotation count at each step:

```bash
for f in $(ls tmp/reactant_*/module_*_reactant_tokw_*.mlir | sort -V); do
    printf '%s\t%d\n' "$(basename $f)" "$(grep -c 'sdy.sharding_per_value' $f)"
done
```

The single transition where the count collapses (likely from a few hundred to a
handful) is the offending pass. The pass name is in the filename (Reactant
includes the pass name in pre/post dump filenames). Cross-reference against
`Reactant.jl/src/Compiler.jl` (or wherever the XLA pipeline is assembled) to
identify the pass and read its source.

**Exit criterion:** named pass and a one-paragraph hypothesis for why it strips
annotations on the enum-bias module shape but not the singleton-type one.

### Phase D — fix

Two possible directions, depending on Phase C:

1. **Fix in Oceananigans:** if the enum-bias code generates a tracing pattern
   the partitioner doesn't handle (e.g. extra `select`/`compare` ops, nested
   `ifelse`), tweak the bias dispatch to produce simpler MLIR — for example,
   replace `bias == LeftBias` with a generated function or an inlined
   `@inline` helper that returns a compile-time-known value, or revert the
   bias call sites that matter (the bounds-preserving WENO and
   `upwind_biased_reconstruction` were the ones touched).
2. **Fix in Reactant / file upstream:** if the responsible XLA pass is just
   buggy on the broader pattern, file an upstream issue with the MWE and the
   exact pass name. Optionally pin Reactant to the last good version while the
   fix lands.

In either case the verification loop is: rerun the MWE → grep
`sdy.sharding_per_value` count → expect hundreds.

## Useful artifacts already collected

- Failing CI MLIR artifact downloaded to:
  `/tmp/oc-mlir/tmp/reactant_XAiNJ3/` on the macOS dev box (758 .mlir files).
  Worth copying to the Linux machine for offline comparison.
- `module_699_..._post_xla_compile.mlir` (annotations still present) and
  `module_719_..._post_xla_compile.mlir` (annotations dropped) are the
  before/after pair for the actual bug.

## Open questions

- Does the bug reproduce on **non-aarch64** Linux (x86_64), or is it
  aarch64-only? CI only runs aarch64 currently. Worth checking on whichever
  Linux box we move to.
- Does it reproduce with newer Reactant versions, or only the pinned one in
  `test/Project.toml`?
- Is `WHILE_CONCAT[] = true` (set in `run_sharding_tests.jl`) involved? Try
  flipping it.
