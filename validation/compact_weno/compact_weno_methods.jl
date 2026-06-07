# CompactWENO (CRWENO-type) vertical reconstruction — reference methods.
# Design: docs/plans/2026-06-05-compact-vertical-advection-design.md
# Reference: Ghosh & Baeder (2012), SIAM J. Sci. Comput. 34(3).
#
# This file is the complete, standalone (stdlib-only) reference implementation to
# port into src/Advection/. Everything here is plain serial Julia on one column.
#
# ## Conventions
#
# Face i sits at zᶠ[i], between cells i-1 and i (N cells, N+1 faces).
# Left-biased (w > 0) reconstruction at face i uses cells {i-2, i-1, i};
# right-biased (w < 0) uses cells {i-1, i, i+1}. Both couple the SAME three faces
# {i-1, i, i+1}, so per-row bias selection by sign(w) keeps the system tridiagonal.
#
# ## Algorithm (once per column, per advected field, per RK stage)
#
#   tables = biased_coefficient_tables(zᶠ)             # ONCE, at scheme construction
#   ĉ = compact_face_values(c̄, w, tables; closure_value)   # assemble + Thomas solve
#   flux[i] = Az * w[i] * ĉ[i]                          # pointwise flux lookup
#
# See demonstrate_usage() at the bottom for a worked example with mixed-sign w.
#
# ## Porting map (reference function → Oceananigans destination)
#
#   biased_coefficient_tables / stretched_coefficient_table
#       → materialize_advection(::CompactWENO, grid): per-face 1D z-arrays stored in
#         the scheme (on_architecture'd); uniform grids dispatch to compile-time
#         constants (uniform_coefficient_table values), like WENO's Nothing-coefficient
#         path. Solves are done in Float64 then converted to eltype(grid).
#
#   nonlinear_weights + the row-assembly bodies in assemble_compact_system
#       → one @kernel launched :xyz over (Center, Center, Face), writing the three
#         diagonals and the right-hand side; bias picked per face from sign(w[i,j,k]).
#         Boundary/immersed rows (the closure_value branches) become identity rows
#         selected with peripheral_node / immersed_peripheral_node, with the explicit
#         buffer_scheme reconstruction as right-hand side (here: first-order upwind;
#         in Oceananigans: WENO(order=3) biased_interpolate_zᵃᵃᶠ).
#
#   the Tridiagonal solve in compact_face_values
#       → solve!(ĉ, BatchedTridiagonalSolver(grid; tridiagonal_direction=ZDirection()), rhs)
#         batched over all (i, j) columns. ĉ is a (Center, Center, Face) scratch field.
#
#   the whole compact_face_values call
#       → precompute_advection!(scheme::CompactWENO, model, tracer), hooked into
#         update_state! after compute_auxiliaries! and halo fills.
#
#   flux lookup
#       → advective_tracer_flux_z(i, j, k, grid, ::CompactWENO, ::ExplicitTimeDiscretization, W, c) =
#             Azᶜᶜᶠ(i, j, k, grid) * W[i, j, k] * scheme.reconstructed_face_values[i, j, k]
#
# Interior-row validity (needed for the kernel's peripheral checks):
#   left bias  needs cells i-3 .. i+1  →  4 ≤ i ≤ N-1
#   right bias needs cells i-2 .. i+2  →  3 ≤ i ≤ N-2
# Rows outside the valid range for their bias get the explicit closure.

using LinearAlgebra

#####
##### Grid generators on [0, 1]
#####

uniform_faces(N) = collect(range(0.0, 1.0, length=N+1))

function geometric_faces(N, ratio)
    spacings = float(ratio) .^ (0:N-1)
    spacings ./= sum(spacings)
    return vcat(0.0, cumsum(spacings))
end

function exponential_faces(N, total_ratio)
    s = log(total_ratio)
    ξ = range(0.0, 1.0, length=N+1)
    return (exp.(s .* ξ) .- 1.0) ./ (exp(s) - 1.0)
end

#####
##### Cell averages of an arbitrary profile (5-point Gauss–Legendre per cell)
#####

const GAUSS_NODES = (-0.9061798459386640, -0.5384693101056831, 0.0,
                      0.5384693101056831,  0.9061798459386640)
const GAUSS_WEIGHTS = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                       0.4786286704993665, 0.2369268850561891)

function quadrature_cell_averages(f, zᶠ)
    N = length(zᶠ) - 1
    c̄ = zeros(N)
    for j in 1:N
        zL, zR = zᶠ[j], zᶠ[j+1]
        center, half = (zL + zR) / 2, (zR - zL) / 2
        c̄[j] = sum(GAUSS_WEIGHTS[q] * f(center + half * GAUSS_NODES[q]) for q in 1:5) / 2
    end
    return c̄
end

#####
##### Nonuniform compact candidates and optimal weights
#####

polynomial_cell_average(p, ζL, ζR) = (ζR^(p+1) - ζL^(p+1)) / ((p + 1) * (ζR - ζL))

# Candidate relation  a ĉ[fL] + (1-a) ĉ[fR] = γ c̄[cA] + (1-γ) c̄[cB],
# exact for cell averages of linear and quadratic polynomials.
# Coordinates are localized at face i and scaled by the left cell spacing
# so the error functionals are O(1).
function fitted_candidate(zᶠ, i, candidate_faces, candidate_cells)
    h = zᶠ[i] - zᶠ[i-1]
    ζ(z) = (z - zᶠ[i]) / h
    point(face, p) = ζ(zᶠ[face])^p
    average(cell, p) = polynomial_cell_average(p, ζ(zᶠ[cell]), ζ(zᶠ[cell+1]))

    fL, fR = candidate_faces
    cA, cB = candidate_cells

    M = zeros(2, 2)
    v = zeros(2)
    for (row, p) in enumerate((1, 2))
        M[row, 1] = point(fL, p) - point(fR, p)
        M[row, 2] = average(cB, p) - average(cA, p)
        v[row] = average(cB, p) - point(fR, p)
    end
    a, γ = M \ v

    error_functional(p) = a * point(fL, p) + (1 - a) * point(fR, p) -
                          γ * average(cA, p) - (1 - γ) * average(cB, p)

    return a, γ, error_functional
end

left_biased_stencils(i) = (((i-1, i  ), (i-2, i-1)),
                           ((i-1, i  ), (i-1, i  )),
                           ((i,   i+1), (i-1, i  )))

# Mirror image about face i: cell j ↦ cell 2i-j-1, face f ↦ face 2i-f.
right_biased_stencils(i) = (((i+1, i  ), (i+1, i  )),
                            ((i+1, i  ), (i,   i-1)),
                            ((i,   i-1), (i,   i-1)))

struct CoefficientTable
    a :: Vector{NTuple{3, Float64}}
    γ :: Vector{NTuple{3, Float64}}
    optimal :: Vector{NTuple{3, Float64}}
end

biased_stencils(i, bias) = bias == :left ? left_biased_stencils(i) : right_biased_stencils(i)

function stretched_coefficient_table(zᶠ; bias=:left)
    N = length(zᶠ) - 1
    a = fill((NaN, NaN, NaN), N + 1)
    γ = fill((NaN, NaN, NaN), N + 1)
    optimal = fill((NaN, NaN, NaN), N + 1)

    # candidate stencils exist for 3 ≤ i ≤ N (left) and 2 ≤ i ≤ N-1 (right)
    valid = bias == :left ? (3:N) : (2:N-1)
    for i in valid
        fits = [fitted_candidate(zᶠ, i, faces, cells) for (faces, cells) in biased_stencils(i, bias)]
        a[i] = ntuple(m -> fits[m][1], 3)
        γ[i] = ntuple(m -> fits[m][2], 3)

        E = [fits[m][3](p) for p in (3, 4), m in 1:3]
        weights = vcat([1.0 1.0 1.0], E) \ [1.0, 0.0, 0.0]
        optimal[i] = ntuple(m -> weights[m], 3)
    end

    return CoefficientTable(a, γ, optimal)
end

function uniform_coefficient_table(N)
    a = fill((2/3, 1/3, 2/3), N + 1)
    γ = fill((1/6, 5/6, 1/6), N + 1)
    optimal = fill((1/5, 1/2, 3/10), N + 1)
    return CoefficientTable(a, γ, optimal)
end

biased_coefficient_tables(zᶠ) = (left  = stretched_coefficient_table(zᶠ; bias=:left),
                                 right = stretched_coefficient_table(zᶠ; bias=:right))

#####
##### Smoothness indicators: uniform-grid Jiang–Shu, WENO-Z weights
#####

function nonlinear_weights(c̄, i, optimal; bias=:left, ε=1e-20)
    if bias == :left
        β₁ = 13/12 * (c̄[i-3] - 2c̄[i-2] +  c̄[i-1])^2 + 1/4 * ( c̄[i-3] - 4c̄[i-2] + 3c̄[i-1])^2
        β₂ = 13/12 * (c̄[i-2] - 2c̄[i-1] +  c̄[i  ])^2 + 1/4 * ( c̄[i-2]           -  c̄[i  ])^2
        β₃ = 13/12 * (c̄[i-1] - 2c̄[i  ] +  c̄[i+1])^2 + 1/4 * (3c̄[i-1] - 4c̄[i  ] +  c̄[i+1])^2
    else
        β₁ = 13/12 * (c̄[i+2] - 2c̄[i+1] +  c̄[i  ])^2 + 1/4 * ( c̄[i+2] - 4c̄[i+1] + 3c̄[i  ])^2
        β₂ = 13/12 * (c̄[i+1] - 2c̄[i  ] +  c̄[i-1])^2 + 1/4 * ( c̄[i+1]           -  c̄[i-1])^2
        β₃ = 13/12 * (c̄[i  ] - 2c̄[i-1] +  c̄[i-2])^2 + 1/4 * (3c̄[i  ] - 4c̄[i-1] +  c̄[i-2])^2
    end
    τ = abs(β₁ - β₃)
    α = optimal .* (1 .+ (τ ./ (ε .+ (β₁, β₂, β₃))) .^ 2)
    return α ./ sum(α)
end

#####
##### Row assembly
#####

# Returns (lower, diagonal, upper, rhs) for the weighted compact relation at face i.
# lower multiplies ĉ[i-1], upper multiplies ĉ[i+1].
function compact_row(c̄, i, table, bias; weights)
    ω = weights == :nonlinear ? nonlinear_weights(c̄, i, table.optimal[i]; bias) : table.optimal[i]
    a = table.a[i]
    γ = table.γ[i]

    if bias == :left
        lower    = ω[1] * a[1]       + ω[2] * a[2]
        diagonal = ω[1] * (1 - a[1]) + ω[2] * (1 - a[2]) + ω[3] * a[3]
        upper    =                                         ω[3] * (1 - a[3])
        rhs = ω[1] * (γ[1] * c̄[i-2] + (1 - γ[1]) * c̄[i-1]) +
              ω[2] * (γ[2] * c̄[i-1] + (1 - γ[2]) * c̄[i  ]) +
              ω[3] * (γ[3] * c̄[i-1] + (1 - γ[3]) * c̄[i  ])
    else
        upper    = ω[1] * a[1]       + ω[2] * a[2]
        diagonal = ω[1] * (1 - a[1]) + ω[2] * (1 - a[2]) + ω[3] * a[3]
        lower    =                                         ω[3] * (1 - a[3])
        rhs = ω[1] * (γ[1] * c̄[i+1] + (1 - γ[1]) * c̄[i  ]) +
              ω[2] * (γ[2] * c̄[i  ] + (1 - γ[2]) * c̄[i-1]) +
              ω[3] * (γ[3] * c̄[i  ] + (1 - γ[3]) * c̄[i-1])
    end

    return lower, diagonal, upper, rhs
end

interior_row(i, N, bias) = bias == :left ? 4 ≤ i ≤ N - 1 : 3 ≤ i ≤ N - 2

# Left-bias-only assembly (w > 0 everywhere), used by the prototype gate studies.
function assemble_compact_system(c̄, table::CoefficientTable; weights, closure_value)
    N = length(c̄)
    lower = zeros(N + 1)
    diagonal = ones(N + 1)
    upper = zeros(N + 1)
    rhs = zeros(N + 1)

    for i in 1:N+1
        if interior_row(i, N, :left)
            lower[i], diagonal[i], upper[i], rhs[i] = compact_row(c̄, i, table, :left; weights)
        else
            rhs[i] = closure_value(i)
        end
    end

    return Tridiagonal(lower[2:end], diagonal, upper[1:end-1]), rhs
end

# Mixed-sign assembly: bias selected per row from sign(w[i]); w is a face vector.
#
# WARNING (2026-06-06): DO NOT use this in production — adjacent rows with opposite
# biases can be exactly linearly dependent in smooth flow (observed in the baroclinic
# adjustment: a right-biased and a left-biased near-boundary row built from the same
# smooth data produced rows (0, x, y) and (x, y, 0) with identical entries → singular
# matrix → zero pivot → NaN). Kept only to study that failure mode. The production
# form is upwinded_face_values below: two fixed-bias solves + pointwise upwinding,
# which is what the Oceananigans implementation does.
function assemble_compact_system(c̄, w, tables; weights=:nonlinear, closure_value)
    N = length(c̄)
    lower = zeros(N + 1)
    diagonal = ones(N + 1)
    upper = zeros(N + 1)
    rhs = zeros(N + 1)

    for i in 1:N+1
        bias = w[i] ≥ 0 ? :left : :right
        if interior_row(i, N, bias)
            table = bias == :left ? tables.left : tables.right
            lower[i], diagonal[i], upper[i], rhs[i] = compact_row(c̄, i, table, bias; weights)
        else
            rhs[i] = closure_value(i)
        end
    end

    return Tridiagonal(lower[2:end], diagonal, upper[1:end-1]), rhs
end

function compact_face_values(c̄, table::CoefficientTable; weights, closure_value)
    T, rhs = assemble_compact_system(c̄, table; weights, closure_value)
    return T \ rhs
end

function compact_face_values(c̄, w, tables; weights=:nonlinear, closure_value)
    T, rhs = assemble_compact_system(c̄, w, tables; weights, closure_value)
    return T \ rhs
end

# Production form: two fixed-bias solves, then pointwise upwinding of the face values.
# Each fixed-bias system is the one validated by the prototype gates; biases are never
# mixed within one matrix (see WARNING above).
function upwinded_face_values(c̄, w, tables; weights=:nonlinear, closure_value)
    ĉᴸ = compact_face_values(c̄,  ones(length(w)), tables; weights, closure_value)
    ĉᴿ = compact_face_values(c̄, -ones(length(w)), tables; weights, closure_value)
    return [wᵢ ≥ 0 ? ĉᴸ[i] : ĉᴿ[i] for (i, wᵢ) in enumerate(w)]
end

#####
##### Explicit WENO5 with uniform-grid coefficients (current Oceananigans practice)
#####

function explicit_weno_face_values(c̄; weights, closure_value)
    N = length(c̄)
    ĉ = zeros(N + 1)
    for i in 1:N+1
        if interior_row(i, N, :left)
            q₁ = ( 2c̄[i-3] - 7c̄[i-2] + 11c̄[i-1]) / 6
            q₂ = (- c̄[i-2] + 5c̄[i-1] +  2c̄[i  ]) / 6
            q₃ = ( 2c̄[i-1] + 5c̄[i  ] -   c̄[i+1]) / 6
            ω = weights == :nonlinear ? nonlinear_weights(c̄, i, (1/10, 3/5, 3/10)) : (1/10, 3/5, 3/10)
            ĉ[i] = ω[1] * q₁ + ω[2] * q₂ + ω[3] * q₃
        else
            ĉ[i] = closure_value(i)
        end
    end
    return ĉ
end

#####
##### SSPRK3 flux-form column advection, w = 1 (Shu–Osher)
#####

function advect_column(c̄₀, Δ, face_values; stop_time, cfl=0.3, snapshots=nothing)
    c̄ = copy(c̄₀)
    N = length(c̄)
    tendency(c) = (ĉ = face_values(c); -(ĉ[2:N+1] - ĉ[1:N]) ./ Δ)

    Δt = cfl * minimum(Δ)
    t = 0.0
    snapshots !== nothing && push!(snapshots, (t, copy(c̄)))
    while t < stop_time - 1e-12
        δt = min(Δt, stop_time - t)
        c¹ = c̄ + δt * tendency(c̄)
        c² = 3/4 * c̄ + 1/4 * (c¹ + δt * tendency(c¹))
        c̄ =  1/3 * c̄ + 2/3 * (c² + δt * tendency(c²))
        t += δt
        snapshots !== nothing && push!(snapshots, (t, copy(c̄)))
    end
    return c̄
end

#####
##### Worked example: mixed-sign w, every step of the algorithm spelled out
#####

function demonstrate_usage(; N=64, total_ratio=10)
    zᶠ = exponential_faces(N, total_ratio)

    # 1. ONCE, at scheme construction (→ materialize_advection):
    tables = biased_coefficient_tables(zᶠ)

    # 2. EVERY RK stage, per advected field (→ precompute_advection!):
    #    cell averages with valid halos, face velocity, explicit boundary closure
    c̄ = quadrature_cell_averages(z -> exp(-((z - 0.5) / 0.1)^2), zᶠ)
    w = [sin(2π * z) for z in zᶠ]                       # mixed sign on purpose
    closure(i) = i ≤ 3 ? c̄[max(i - 1, 1)] : c̄[min(i - 1, length(c̄))]   # first-order upwind

    #    assemble (→ the :xyz kernel) and solve (→ BatchedTridiagonalSolver)
    ĉ = compact_face_values(c̄, w, tables; weights=:nonlinear, closure_value=closure)

    # 3. Pointwise flux lookup (→ advective_tracer_flux_z):
    flux = w .* ĉ                                        # × Azᶜᶜᶠ in Oceananigans

    return ĉ, flux
end
