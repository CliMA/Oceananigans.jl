---
paths:
  - src/**/*.jl
  - test/**/*.jl
  - validation/**/*.jl
  - examples/**/*.jl
---

# Style Rules

These rules are **ALWAYS** in effect. Apply them in every file, every scope — no exceptions.

## Variable Naming

### Rule 1 — Never mix math and verbose notation in the same identifier

A single identifier is *either* fully math notation *or* fully verbose English. Never half-and-half.

- ❌ `T_air`, `temp_a`, `Nz_max`, `dt_start`
- ✅ math: `Tₐ`, `Tᵃ`, `Ta`, `Nz`, `dt`
- ✅ verbose: `air_temperature`, `maximum_vertical_grid_size`, `start_timestep`

In the same scope, pick one style and stay consistent — don't write `t_start` alongside `start_time`, or `t_s` next to `time_series`.

Default by context:
- **Inside kernels / computational inner loops**: prefer math notation (`τ`, `δT`, `Nz`, `φ`, `λ`)
- **Outside kernels (scripts, function signatures, keyword arguments, user-facing APIs)**: prefer verbose notation

### Rule 2 — Verbose variables read in English, no truncation, qualifier-first

Verbose names must read as natural English noun phrases. Adjectives/qualifiers come *before* the noun, and no word is truncated.

- ❌ `temp_air` (truncated + adjective after noun)
- ❌ `temperature_air` (adjective after noun)
- ❌ `air_temp` (truncated)
- ✅ `air_temperature`
- ❌ `vel_horiz_max` → ✅ `maximum_horizontal_velocity`
- ❌ `coeff_diff` → ✅ `diffusion_coefficient`

### Rule 3 — No underscores in math notation

Math-style identifiers never use `_`. Use sub/superscript Unicode or concatenation instead.

- ❌ `d_K`, `T_a`, `u_star`
- ✅ `dK`, `dᴷ`, `Ta`, `Tₐ`, `Tᵃ`, `u★`, `uₛ`

(This rule combined with Rule 1 means: if you see `_` in a name, the whole name must be verbose English.)

### Rule 4 — Leading `_` is reserved for `@kernel` functions

A name starting with `_` is reserved by convention for the `@kernel` form of a launching function (e.g., `compute_tendencies!` launches `_compute_tendencies!`). Never use a leading `_` for ordinary helpers, internal utilities, or "private" functions — pick a real name instead.

- ❌ `_helper`, `_compute_internal`, `_validate` (for non-kernel helpers)
- ✅ `compute_internal`, `validate_inputs`, `apply_boundary`
- ✅ `_compute_tendencies!` (only because it is the `@kernel` paired with `compute_tendencies!`)

## Comments

Be synthetic with comments. Code should be self-documenting through clear names (see the variable naming rules above) and obvious structure — heavy commentary is a sign the code itself needs work, not more prose around it.

- Default to **no comment**. If a reader who knows the language and domain can follow the code from names and shape alone, the comment is noise.
- Add a comment only when the **logic is genuinely convoluted** and a one-line hint saves the reader real time: a non-obvious invariant, a subtle index trick, a workaround for a specific upstream bug, a numerical-stability detail, a sign convention that contradicts intuition.
- Keep such comments **surgical and minor**: one line, placed exactly at the confusing step. Never restate what the next line obviously does.
- Do not write block comments or multi-paragraph docstrings inside function bodies. Section banners that group a long function into labeled stages are fine. Do not narrate the task ("added for X", "fixes Y"), reference callers, or leave TODO-style commentary unless explicitly requested.
- ❌ `# increment counter` above `i += 1`
- ❌ `# loop over all cells` above `for cell in cells`
- ✅ `# offset by -1 because Fortran indexing is preserved in the on-disk layout`
- ✅ `# use Kahan summation here: naïve sum loses ~3 digits on long trajectories`
