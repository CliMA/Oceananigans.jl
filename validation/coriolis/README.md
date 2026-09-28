# C-grid Coriolis without null modes: the shear-signed oriented scheme

This folder holds the validation scripts of the JAMES manuscript on C-grid Coriolis discretizations (Overleaf project
`chiral_coriolis`) and the global experiment that compares the four-point average, the C-D scheme and the oriented
scheme. The attempt-by-attempt history is in `research_notes/C grid Coriolis scheme attempts/LOG.md` of the main
checkout, together with the linear-theory tools (`tools/`) that produced the theory figures.

## The scheme

The Coriolis term is the energy-conserving four-point average plus an increment written with the grid's own operators,

    ∂t (u, v) = EnergyConserving + grad A + curl* B,    A_c = Σ_z α_cz f_z Γ_z,    B_z = - Σ_c α_cz f_z D_c,

where `Γ = δx(Ax v) - δy(Ay u)` is the area-weighted circulation at the cell corners `z`, `D = δx(Ax u) + δy(Ay v)` the
horizontal volume-flux divergence at the cell centres `c`, and `α_cz` couples each cell to its four corners. Since
`grad = -divᵀ` and `curl* = -curlᵀ` in the `ΔxΔyΔz` inner product, the increment does no work for any coupling, metric or
land mask, and `curl(grad A) = 0` makes the curl of the increment a function of `D` only (potential vorticity
consistency). The coefficients may vary in space and time without breaking either property. Pairs whose corner touches
land or a wall, whose centre is inactive, or which cross the tripolar fold are excluded.

The weights on the corner `(i + a, j + b)` of the cell `(i, j)` are

    w = χ ε (a - b) + η (1 - 2 (a - b)²),    ε = 1/8,  η = 1/32,

so that `χ = +1` is the south-east stencil (`w_SE = ε - η`, `w_NW = -ε - η`, `w_SW = w_NE = η`) and `χ = -1` its 180°
rotation (north-west). On a uniform grid the potential vorticity interpolation becomes

    m = cos(kΔ/2) cos(lΔ/2) + 4η S² sin(kΔ/2) sin(lΔ/2) + 2iχε S² sin((k - l)Δ/2),    S² = 4 sin²(kΔ/2) + 4 sin²(lΔ/2),

and `|m(π, 0)| = |m(0, π)| = 8ε = 1`: the null modes of the four-point average feel the full `f`. Near the Nyquist
wavenumber a real interpolation cannot restore rotation (`Re m(π, 0) = 0`), so it is the phase `Im m` that removes the null
modes, and the phase has a sign.

The chirality `χ(x, y, z)` sets that sign. The chiral part of the scheme converts available potential energy at a rate
proportional to `χ` times the thermal-wind shear projected on the stencil diagonal; averaged over wave directions the
conversion is non-positive when

    χ = smooth(sign(smooth(∂z (u - v)))),    sign(0) = -1,

evaluated level by level, with the shear taken below the uppermost interface (surface Ekman layer), averaged over the two
interfaces adjacent to each level, and smoothed over 3 cells with masked (1, 2, 1) passes. `χ` is updated daily and
relaxed toward this target over 30 days; daily switching without relaxation produces grid-scale noise (LOG #30).

## Code

- `src/Coriolis/shear_signed_coriolis.jl`: `ShearSignedCoriolis(grid; ε, η, smoothing, update_interval, adjustment_time)`.
  The potentials `A` and `B` are stored fields computed once per stage by `update_coriolis!`, which also updates `χ`
  when the clock passes the next update time. The Coriolis term uses plain differences of `A` and `B` along the layer
  (the z⋆ slope correction of `∂x` would break the adjoint relations). `χ` and the next update time are part of the
  model's prognostic state, so checkpoints restart bit for bit. All operations are kernels (verified on Metal in Float32).
- `src/Models/HydrostaticFreeSurfaceModels/update_hydrostatic_free_surface_model_state.jl`: calls
  `update_coriolis!(model.coriolis, model)` before the `UpdateStateCallsite` callbacks (a no-op for other schemes).
- `src/Coriolis/cd_scheme.jl`: the C-D scheme of Adcroft, Hill and Marshall (1999), inherited from `ss/cd-coriolis-scheme`.
- `test/coriolis/schemes.jl`: the 2Δx mode feels `f v₀`, the Coriolis term does no work on an immersed grid with `χ` of both
  signs, and a checkpoint restart reproduces `u`, `v` and `χ`.

A fixed orientation is `ShearSignedCoriolis(grid; adjustment_time=Inf)` with `set!(scheme.chirality, ±1)`.

## Validation scripts (manuscript Sections 3 and 4)

These use the prototype `oriented_coriolis.jl` (16 corner offsets, fixed orientation) and run on the CPU.

| script | content |
|---|---|
| `oriented_coriolis.jl` | prototype scheme with arbitrary weights on a 4 × 4 corner stencil |
| `oriented_coriolis_linear_analysis.jl` | dispersion relations, null modes and phase of `m` |
| `oriented_coriolis_periodic_experiments.jl`, `periodic_figures.jl` | f-plane and β-plane periodic experiments |
| `null_mode_adjustment.jl`, `null_mode_adjustment_all.jl` | adjustment of the 2Δ modes, all schemes |
| `cd_scheme_linear_analysis.jl`, `cd_scheme_decaying_turbulence.jl` | C-D scheme analysis and decaying turbulence |
| `dispersion_figure.jl`, `schematic_figure.jl`, `spectra.jl` | manuscript figures |
| `momentum_budgets.jl` | kinetic energy and enstrophy budgets term by term (used by the global runs) |
| `test_coriolis_scheme_null_mode.jl`, `coriolis_immersed_stress_test.jl`, `jamart_basin.jl`, `enceladus_convection.jl` | older checks |

## Global experiment (`global/`)

Idealized ocean on the 1° tripolar grid (360 × 180 × 4, levels at 4000, 1500, 500, 100 m, z⋆), ETOPO bathymetry
(`bottom_height.jld2`, written by `bottom_height.jl`), zonal wind stress `τ = -0.15 sin(2φ) sin(6φ)`, SST restored to
`30 cos²φ` over 30 days, convective adjustment, `WENOVectorInvariant(order=5)` momentum and `WENO(order=7)` tracer
advection, split-explicit free surface, `SplitRungeKutta3`, Δt = 1 hour, from rest with `T = 10 exp(z / 1000)`. Use Float64:
Float32 overflows in the WENO smoothness indicators of the vector-invariant scheme (fix pending upstream).

    julia --project run_global.jl <FourPoint | CD | ShearSigned | FixedSE | FixedNW> [years = 5] [pickup]

Environment variables: `CORIOLIS_ARCHITECTURE=GPU` (loads CUDA), `CORIOLIS_FLOAT_TYPE` (default Float64),
`CORIOLIS_OUTPUT` (default `global/output`). The project needs Oceananigans developed from this branch, JLD2 and, on a
GPU, CUDA. Each run writes `<scheme>_<FT>.jld2` (surface ψ, ζ, speed, u, v, T, the level below the surface u3, v3, and
χ, χ3 for the oriented scheme, every 10 days), yearly checkpoints, the daily budget history `<scheme>_<FT>_budget.jld2`,
`coordinates.txt`, and `<scheme>_<FT>.done` at the end. On the CPU with 3 threads the four-point run takes about 4 hours per
simulated year and the oriented scheme 1.4 times more.

On tartarus (MIT VPN, `~/.juliaup/bin/julia`, `/home` nearly full): keep everything in one scratch directory, set
`JULIA_DEPOT_PATH=<scratch>/depot:$HOME/.julia`, pick an idle GPU with `CUDA_VISIBLE_DEVICES`, and launch with
`(setsid nohup ... &)`, for example

    CUDA_VISIBLE_DEVICES=1 CORIOLIS_ARCHITECTURE=GPU CORIOLIS_OUTPUT=<scratch>/output \
        julia --project=<scratch>/env validation/coriolis/global/run_global.jl ShearSigned 5

## Analysis (`global/analysis/`, Python with numpy, h5py, matplotlib, ffmpeg)

Run where the outputs are, with `CORIOLIS_OUTPUT` pointing at them.

- `band_table.py [days] [FT]`: surface v variance in the trade bands (8–22°, zonal 2.9–5Δ), near the Nyquist wavenumber at
  30–50°, and in the mesoscale band, at equal days for all available runs.
- `spin_up_figures.py <prefix> [FT]`: time series of the bands, barotropic streamfunction range and mean surface speed, and
  maps of the zonal 2Δ part of surface v.
- `fields_movie.py <output.mp4> [FT]`: surface speed, SST and surface v every 30 days.

## Results so far

Restart from the four-point state at day 1095 (Float64 except fixed SE), surface v variance in cm²/s² at day 1210:

| run | trades S / N | 30–50° S / N near-Nyquist | mesoscale 30–50N / 55–65S |
|---|---|---|---|
| four-point | 4.53 / 2.19 | 0.58 / 0.11 | 4.62 / 5.24 |
| fixed SE (χ = 1) | 58.1 / 19.7 | 0.11 / 0.01 | 4.48 / 3.81 |
| fixed NW (χ = -1) | 1.84 / 1.78 | 0.45 / 0.12 | 4.53 / 4.90 |
| shear-signed | 1.79 / 1.80 | 0.11 / 0.01 | 4.44 / 4.44 |

Spin-up from rest (Float64, χ relaxed over 30 days), day 300:

| band | four-point | C-D | shear-signed |
|---|---|---|---|
| 8–22°S, 2.9–5Δ | 0.121 | 0.050 | 0.322 |
| 8–22°N, 2.9–5Δ | 0.032 | 0.005 | 0.027 |
| 30–50°S, near-Nyquist | 0.175 | 0.005 | 0.075 |
| 30–50°N, near-Nyquist | 0.032 | 0.000 | 0.003 |
| 55–65°S mesoscale | 4.05 | 0.85 | 2.83 |

C-D is quiet at the grid scale but damps the mesoscale. In the four-point run the trade-band noise grows by an order of
magnitude between years 2 and 2.5 (8–22°S: 0.40 at day 730, 3.7 at day 900, 6.1 at day 1300); the comparison of the
southern trades, where the shear-signed run is noisier than the four-point during the first year, is decided in years 2–3.
