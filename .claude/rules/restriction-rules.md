---
paths:
  - src/**/*.jl
  - test/**/*.jl
  - validation/**/*.jl
  - examples/**/*.jl
---

# Restriction Rules

Machine-written code fails in a characteristic way: it is **correct and unreadable**. Every branch is handled,
every failure anticipated, every invariant checked at three layers — and no human can hold the result in their
head. A reviewer facing 700 lines cannot tell which 40 are the feature.

These rules exist to keep the diff small, the mechanisms singular, and the prose scarce. They are **always in
effect**, and they outrank any instinct toward completeness. Names in the examples are illustrative.

> **The test behind every rule below:** could a reviewer who knows the physics read this hunk once, in order,
> and say yes?

**They apply to existing code too.** Every rule below is also a license to delete: when you are already editing
a file and a rule catches something that is *already there*, removing it is in scope, not scope creep. What is
out of scope is going looking — do not open unrelated files to hunt for violations, and keep a cleanup that
grows past the change it started from to its own PR.

## Rule 1 — The feature is the diff; everything else is scope creep

Before opening a PR, ask what the smallest diff is that delivers the stated change. That diff is the PR.
Anything that is not the feature — validation, fallbacks, gap-fillers, helper APIs — is a **separate concern**
and belongs in a separate PR, or nowhere.

- A generalization that **deletes** an abstraction is the best possible change. A change that ends net-negative
  needs no justification; celebrate it and stop there.
- If the feature commit is net-zero and the PR is +700, the PR is 95 % scaffolding. Drop the scaffolding.
- Adding lines to `src/` is a cost paid forever. The bar is not "is this useful?" but "is the codebase better
  off carrying this than not?"

**Small diffs may carry more than one concern.** Below **150 changed lines** — insertions plus deletions, counted
over the paths this file governs — a PR that bundles several independent changes is fine: the whole thing fits on
one screen, and splitting it costs more review time in branches and rebases than it saves. At or above 150, one
concern per PR. The threshold licenses *bundling*, not *hiding*: every concern still has to be named in the PR
description, and a silent change to an existing default — a timestepper, a drag coefficient, a substep count —
is called out on its own line whatever the diff size, because that is what a reviewer would otherwise miss.

**A PR must show only its own change.** Reviewability is a property of the diff GitHub renders, not of the
commits you know are yours. Never open a PR whose head has merged another feature branch into it: base the PR
on that branch instead, so the diff shows your commits alone. If the base has to be a feature branch, keep it
one level deep and rebase — never merge sideways.

❌ A 5-line forwarding method and an 11-line test, presented as **+6186 / −1926 across 119 files**, because the head
   merged another feature branch (50 commits of unrelated work) into itself.
✅ Base the PR on that feature branch directly. Diff renders 16 lines. Review takes one minute.

❌ Feature lands in commit 1 (+142/−143); commits 4 and 7 add +680 lines of validation nobody asked for.
✅ Feature lands in commit 1. PR ends. Validation, if genuinely needed, is proposed separately on its merits.

## Rule 2 — One invariant, one mechanism

An invariant is enforced in exactly one place. Enforcing it at setup *and* in the kernel *and* through a
fallback *and* in a gap-filler does not make it four times as safe — it makes it impossible to tell which
mechanism is load-bearing, and guarantees the four drift apart.

Before adding a check, find where the invariant is already enforced. If a runtime floor already exists,
setup-time validation is redundant; if setup-time validation is added, the runtime floor should be deleted.

❌ A positive diffusivity checked in `validate_diffusivity`, again in `local_diffusivity`, again in
   `guard_local_diffusivity`, again in `fill_diffusivity_gaps!` — while `face_diffusivity`'s `max(κ, κ₀)` already
   floored it before the PR.
✅ One check, at the layer that owns the contract. State in one line where that is.

## Rule 3 — No guards for states the code cannot reach

Never add machinery to defend against a state that is unreachable through the public API. "A user might mutate
this field after construction" and "a downstream package might pass a wrong type" are hypotheticals, not
requirements. Code written against hypotheticals is permanent; the hypothetical usually never arrives.

A **loud failure is better than a silent fallback.** A `NaN` that reaches the tendencies is visible in the
first output; a diffusivity silently floored at `1e-5` produces plausible wrong fluxes forever.

- If a guard's own docstring has to say *"this is a guard, not a physical parameter"* — delete the guard.
- A keyword argument must express **physics or numerics** — a rate, a length, a tolerance, an iteration
  cap, a search radius. What it must not express is a defense: a floor, a clamp, or a fallback whose
  only job is to keep malformed input from reaching the solver. Never grow a struct or a `show` line
  for one either.

❌ `minimum_diffusivity` — a new public keyword, struct field, `FT` conversion and `show` row, existing
   only so a field mutated after validation cannot produce `NaN`.
✅ Validate at construction, or let it fail loudly. No new API.

## Rule 4 — Don't invent an API to solve a problem the same PR created

If a change makes previously valid data invalid, and the PR then ships a new exported helper to repair that
data, the PR has manufactured its own problem. Fix the producer, reuse what exists, or loosen the new
constraint.

❌ New validation rejects the `NaN` cells of `load_bathymetry` → same PR adds `fill_bathymetry_gaps!`,
   two kernels and a 20-line docstring, to undo it — while noting that an existing helper already does this.
✅ `load_bathymetry` does not emit values its only consumer rejects, or the consumer tolerates them.

## Rule 5 — Minimize the number of methods; dispatch is for polymorphism, not for `if`

**Every method is a maintenance cost.** It has to be named, found, kept consistent with its siblings, kept
correct under refactoring, and understood by anyone reading any of its callers. If one method does the job,
do not define two or three. Prefer widening an existing method to introducing a new one, and prefer no new
name at all to a new name that only forwards.

- Adding a method is justified by **behavior that genuinely differs by type**, not by a value that differs.
  A value that differs is an argument.
- A method whose body is a single call to another method, with arguments reordered or a default filled in,
  should not exist — give the callee the default instead.
- Do not write families of one-line methods whose only purpose is to select a branch: a reader must not have
  to collect four scattered definitions to learn that a value is clamped when it comes from a `Field`.
- If a helper has exactly one call site and no name a physicist would recognize, inline it.
- Count the methods a hunk adds before pushing. Nine new names for one clamp is not thoroughness.

❌ `clamped_diffusivity` (3 methods) + `guard_local_diffusivity` (3 methods) +
   `guard_local_viscosity` (3 methods) — nine one-liners implementing one clamp.
❌ `local_wall_flux(f, i, j) = local_wall_flux(f, i, j, nothing)` — a forwarding method for a
   default; give `z` the default in the one real method.
✅ One `@inline` with an `ifelse`, or the clamp written where the value is used.

## Rule 6 — No table-driven generalization of a small fixed set

Four things are not a collection. Do not build a registry of `(name, slot, predicate, message)` tuples and loop
over it to save writing four explicit checks — the table costs more to read, hides the control flow, and its
promised "adding one is a single edit" is usually false the moment anything unpacks it positionally.

Reach for a table at roughly a dozen homogeneous entries, when they genuinely vary at runtime — not before.

❌ `closure_parameter_slots` returning 4 tuples of name/value/predicate/prose, walked by two functions, then
   destructured positionally as `νᵛ, κᵛ, Cᵛ, ℓᵛ = values` right afterward.
✅ Four explicit one-line checks, in order, readable top to bottom.

## Rule 7 — Comments describe this code, now — nothing else

Reinforces the comment rules in `style-rules.md`, which this codebase violates most often. A comment must not:

- explain what **another** function does, or what a **caller** or **kernel elsewhere** would do;
- justify a design decision, compare it to a rejected alternative, or narrate history;
- describe code that **is not there** ("the kernel would fail with…", "a bare `MethodError` on CPU…");
- teach the language (`==` vs `===`, how `@inbounds` works, why `Adapt` needs field order);
- carry a multi-line design memo or `TODO` about a different module's construction order.

If the reasoning genuinely cannot be dropped, it belongs in the PR description — which reviewers read once and
which does not age in the source tree.

❌ Five lines above an `Adapt.adapt_structure` method explaining that a positional rebuild
   elsewhere would mis-wire the struct and that a test pins the field order.
❌ `# Grids are compared with `==` … rather than `===`: two references to one grid are not guaranteed to be egal`
✅ `# ifelse, not ?:, so the kernel stays branch-free on GPU`

## Rule 8 — Docstrings describe, they do not argue

A docstring says what the function does, its arguments, and its units. It does not defend the function's
existence, quantify what would go wrong without it, or send the reader to three other functions to understand
it. A cluster of `@ref`s pointing at siblings means the abstraction is wrong, not that the docs are thorough.

- No "this is a backstop, not a supported regime" framing. No bulleted rationale lists.
- No tuning advice for a parameter the same docstring calls non-physical.
- Keep the argument list; drop the essay.

❌ 18 lines documenting `minimum_diffusivity`, of which 2 describe it and 16 argue about when it binds.
✅ `` `minimum_diffusivity`: Floor [m² s⁻¹] on the diffusivity used in the flux. Default: 1e-5. ``

## Rule 9 — Error messages are one sentence

Name the offending object, say what was required, stop. An `ArgumentError` is not a tutorial: it should not
explain the physics, list downstream consequences, quote numbers, or recommend a helper function. Six errors of
six lines each is fifty lines of string literal in `src/`.

❌ `"$name has $(count(...)) cells that are not $requirement (minimum $(minimum(values))). A bad cell either
   propagates NaN through the tendencies or silently zeroes every flux there.
   Fill the gaps first, e.g. with fill_diffusivity_gaps!."`
✅ `"diffusivity must be finite and positive, found $(minimum(values))"`

## Rule 10 — Test behavior, not scaffolding

Tests pin the physics and the public contract. A test that asserts an internal helper's return type, a struct's
field order, or the exact value a fallback produces makes the scaffolding **load-bearing**: it can no longer be
deleted without "breaking tests", which is precisely how machine-written scaffolding becomes permanent.

- Never write a test whose only purpose is to lock in a guard added by the same PR.
- No tautological assertions (`@test !isnothing(constructor(...))`) — a constructor returning `nothing` is a bug
  in itself, not something to assert.
- One clear analytical check (`c = exp(-κ k² t) sin(k x)`, cell by cell) beats twenty defensive ones.

## Rule 11 — Notation follows the codebase, or it is English

No inventing decorations. A superscript must mean what it means in the physics, not encode a code-level
distinction like "host-side copy" or "clamped contribution". If a value needs that kind of qualifier, it
needs a verbose English name.

Reread `style-rules.md` Rules 1–3 before naming anything: identifiers are **fully math or fully English**.

❌ `κᵛ`, `zᵛ`, `νᵛ`, `κᵐᵃˣᵛ`, `κuᶜ` (invented `ᵛ` = "values array", `ᶜ` = "clamped")
❌ `κmin` (math + truncated English in one identifier), `to_FT` (ad-hoc local closure)
✅ `diffusivity_values`, `minimum_diffusivity`

## Rule 12 — Use constructors as designed

This is broken most often in new doctests and examples. Never reach into a
constructor's result to prove it worked, and never write a doctest whose assertion is that `set!` assigned a
number.

❌ ```jldoctest``` that builds a full model to check
   `model.tracers.c[2, 1, 1] == 4.0`
✅ A doctest that shows the object, via `show`/`summary`, or nothing at all.

## Rule 13 — A definition that fits on one line is written on one line

Never break a single-expression definition across lines to stay under a column ceiling. 110 columns is a
*minimum* fill target, not a maximum, so a one-line body belongs on the same line as its signature — and a
returned tuple is not a reason to wrap. Wrap only when the whole definition genuinely will not fit in ~140
columns, and then break at an argument boundary, never straight after the `=`.

❌ ```julia
@inline interior_indices(i, j, grid) =
    wrap_index(i, grid.Nx, topology(grid, 1)), wrap_index(j, grid.Ny, topology(grid, 2))
```
✅ ```julia
@inline interior_indices(i, j, grid) = wrap_index(i, grid.Nx, topology(grid, 1)), wrap_index(j, grid.Ny, topology(grid, 2))
```

## Mandatory self-review

**Claude: run the checklist below over every change you make, before you present it.** Not at PR time — at
the end of every edit, every time. This is not optional and it is not proportional to how confident you feel.

The protocol:

1. **Read your own diff.** `git diff` (or `git diff --stat` first if it is large). Review what you actually
   wrote, not what you remember intending. The checklist is answered from the diff, never from memory.
2. **Answer all fourteen questions.** Every **yes** is a defect you introduced. There is no "yes, but it is
   justified here" — that judgment is exactly the one that produced the drift.
3. **Cut before presenting.** Removing your own scaffolding is the default action and needs no permission.
   Do not ask "should I remove the validation?" — remove it, and say that you did.
4. **Report the result in one line.** State that the checklist ran and what it caught, e.g.
   *"Self-review: cut the `minimum_diffusivity` guard (Q3) and two forwarding methods (Q5); 41 lines
   removed."* If it caught nothing, say *"Self-review: clean."* Never claim clean without having read the diff.
5. **Never delete a rule to pass the checklist.** If a rule seems wrong for the case at hand, say so to the user
   and let them decide. Editing `restriction-rules.md` to accommodate your own diff is the one forbidden move.

For a three-line edit this takes seconds. For a large one it is the difference between review and no review.

## The checklist

Any **yes** means cut before presenting.

1. Is the diff more than ~2× the size of the change actually described in the title, **and** 150 lines or more?
   (Under 150 changed lines, bundled concerns are fine — but is every one of them named in the description?)
2. Does any invariant get enforced in more than one place?
3. Does any new code defend against a state unreachable through the public API?
4. Does the PR add an API to repair damage the PR itself caused?
5. How many methods does this diff add, and could any of them be one method, an argument, or nothing?
   (≥3 one-liners implementing a single `ifelse`, or a method that only forwards to another, is a no.)
6. Is there a table, registry, or predicate list covering fewer than ~12 fixed entries?
7. Does any comment mention a function, module, kernel, or test other than the one it sits in?
8. Does any docstring argue for the function's existence or carry `@ref`s to siblings to be understood?
9. Is any error message longer than one sentence?
10. Does any test pin an internal helper, a field order, or a fallback value?
11. Does any identifier carry an invented sub/superscript, or mix math with English?
12. Is there a new public keyword argument that is neither physics nor numerics — a floor, clamp, or
    fallback guarding against malformed input?
13. Does the diff contain anything a reader would not expect from the **title or the description** —
    including another branch merged into the head, or a changed default that neither mentions?
14. Is any single-expression definition wrapped onto a second line when it would fit on one?
