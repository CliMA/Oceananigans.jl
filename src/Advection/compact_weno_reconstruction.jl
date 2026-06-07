#####
##### CompactWENO: fifth-order compact (CRWENO-type) reconstruction for vertical tracer advection
#####
##### Following Ghosh & Baeder (2012), SIAM J. Sci. Comput. 34(3). The biased face values along
##### each column satisfy a tridiagonal system whose coefficients are WENO-weighted combinations
##### of third-order compact candidates. Two fixed-bias systems are solved per tracer per stage
##### (`precompute_advection!`, called from the hydrostatic time stepping), and the vertical
##### advective flux upwinds pointwise between the two reconstructions,
#####
#####     F = Azᶜᶜᶠ (w⁺ ĉᴸ + w⁻ ĉᴿ).
#####
##### Biases must NOT be mixed within one matrix: adjacent opposite-bias rows can be linearly
##### dependent (exactly singular) in smooth flow.
#####
##### Face k sits between cells k-1 (below) and k (above). Left-biased reconstruction at face k
##### uses cells {k-2, k-1, k}; right-biased uses cells {k-1, k, k+1}. Near boundaries the rows
##### fall back to third-order compact WENO (candidates coupling only faces {k-1, k}), and to the
##### explicit `buffer_scheme` at the outermost faces.
#####
##### The tridiagonal coefficients are computed on the fly inside the solver's sweep through
##### `get_coefficient` extensions (the same pattern as the vertically-implicit diffusion solver),
##### so no coefficient arrays are stored.
#####
##### Scope: tracer advection in `HydrostaticFreeSurfaceModel` on grids with a `Bounded` vertical
##### topology. Horizontal reconstructions delegate to `horizontal_scheme`; vertical momentum
##### reconstructions delegate to `buffer_scheme` / `advecting_velocity_scheme`. MultiRegion and
##### `MutableVerticalDiscretization` (z-star) grids are untested.

using KernelAbstractions: @kernel, @index
using Oceananigans.Fields: ZFaceField
using Oceananigans.Grids: znodes, inactive_cell, topology, Bounded, ZDirection
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Utils: launch!

import Oceananigans.Solvers: get_row

struct CompactWENO{FT, WCT, TD, H, CA, SI, CT, S, R} <: AbstractUpwindBiasedAdvectionScheme{3, FT, TD}
    horizontal_scheme :: H
    buffer_scheme :: CA
    advecting_velocity_scheme :: SI
    time_discretization :: TD
    coefficients :: CT
    tridiagonal_solver :: S
    reconstructed_variable :: R

    CompactWENO{FT, WCT}(horizontal_scheme::H,
                         buffer_scheme::CA,
                         advecting_velocity_scheme::SI,
                         time_discretization::TD,
                         coefficients::CT,
                         tridiagonal_solver::S,
                         reconstructed_variable::R) where {FT, WCT, H, CA, SI, TD, CT, S, R} =
        new{FT, WCT, TD, H, CA, SI, CT, S, R}(horizontal_scheme, buffer_scheme, advecting_velocity_scheme,
                                              time_discretization, coefficients, tridiagonal_solver,
                                              reconstructed_variable)
end

"""
    CompactWENO([FT = Oceananigans.defaults.FloatType;]
                horizontal_scheme = WENO(FT; order=5),
                buffer_scheme = Centered(FT; order=2),
                weight_computation::DataType = Nothing,
                time_discretization = ExplicitTimeDiscretization())

Return a fifth-order compact weighted essentially non-oscillatory (CRWENO-type) reconstruction
scheme for _vertical_ tracer advection with precision `FT`, following Ghosh & Baeder (2012).
The compact candidate coefficients and optimal weights account for vertical grid stretching;
smoothness indicators use the uniform-grid Jiang–Shu form with WENO-Z weighting.

The compact reconstruction couples each vertical column globally: face values are obtained
from two fixed-bias tridiagonal solves per tracer per time step, precomputed before the
tendency computation. Currently supported as tracer advection in `HydrostaticFreeSurfaceModel`
on grids with `Bounded` vertical topology.

Keyword arguments
=================

- `horizontal_scheme`: reconstruction used in the horizontal directions.
                       Default: `WENO(FT; order=5)`.

- `buffer_scheme`: explicit reconstruction used for the outermost closure rows of the
                   compact system (and for vertical momentum reconstruction).
                   Default: `Centered(FT; order=2)`; pass `UpwindBiased(FT; order=1)`
                   for a dissipative closure.

- `weight_computation`: reserved for future weight-computation variants. Default: `Nothing`.

- `time_discretization`: only `ExplicitTimeDiscretization()` is supported.

Example
=======

```jldoctest
julia> using Oceananigans

julia> summary(CompactWENO())
"CompactWENO{Float64, Nothing}(order=5)"
```
"""
function CompactWENO(FT::DataType=Oceananigans.defaults.FloatType;
                     horizontal_scheme = WENO(FT; order=5),
                     buffer_scheme = Centered(FT; order=2),
                     weight_computation::DataType = Nothing,
                     time_discretization = ExplicitTimeDiscretization())

    advecting_velocity_scheme = Centered(FT; order=4)
    return CompactWENO{FT, weight_computation}(horizontal_scheme, buffer_scheme, advecting_velocity_scheme,
                                               time_discretization, nothing, nothing, nothing)
end

weno_order(::CompactWENO) = 5
Base.eltype(::CompactWENO{FT}) where FT = FT
Base.summary(::CompactWENO{FT, WCT}) where {FT, WCT} = string("CompactWENO{$FT, $WCT}(order=5)")

Base.show(io::IO, scheme::CompactWENO) =
    print(io, summary(scheme), " \n",
              "├── horizontal scheme: ", summary(scheme.horizontal_scheme), " \n",
              "├── buffer scheme: ", summary(scheme.buffer_scheme), " \n",
              "└── advecting velocity scheme: ", summary(scheme.advecting_velocity_scheme))

# `contains_compact_weno` lets models reject configurations CompactWENO does not support.
contains_compact_weno(scheme) = false
contains_compact_weno(::CompactWENO) = true
contains_compact_weno(scheme::FluxFormAdvection) = contains_compact_weno(scheme.x) |
                                                   contains_compact_weno(scheme.y) |
                                                   contains_compact_weno(scheme.z)

# Tendency kernels only need the reconstructed face values and the delegation schemes;
# the solver and the (CPU-built) coefficient tables stay on the host.
Adapt.adapt_structure(to, scheme::CompactWENO{FT, WCT}) where {FT, WCT} =
    CompactWENO{FT, WCT}(Adapt.adapt(to, scheme.horizontal_scheme),
                         Adapt.adapt(to, scheme.buffer_scheme),
                         Adapt.adapt(to, scheme.advecting_velocity_scheme),
                         Adapt.adapt(to, scheme.time_discretization),
                         nothing,
                         nothing,
                         Adapt.adapt(to, scheme.reconstructed_variable))

Architectures.on_architecture(to, scheme::CompactWENO{FT, WCT}) where {FT, WCT} =
    CompactWENO{FT, WCT}(on_architecture(to, scheme.horizontal_scheme),
                         on_architecture(to, scheme.buffer_scheme),
                         on_architecture(to, scheme.advecting_velocity_scheme),
                         on_architecture(to, scheme.time_discretization),
                         on_architecture(to, scheme.coefficients),
                         on_architecture(to, scheme.tridiagonal_solver),
                         on_architecture(to, scheme.reconstructed_variable))

#####
##### Interpolation interface: horizontal reconstructions delegate to `horizontal_scheme`,
##### vertical momentum reconstructions to the explicit sub-schemes; only the vertical
##### tracer flux uses the compact reconstruction.
#####

for interpolation_location in (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ)
    symmetric_interpolation = Symbol(:_symmetric_interpolate_, interpolation_location)
    biased_interpolation = Symbol(:_biased_interpolate_, interpolation_location)
    @eval begin
        @inline $symmetric_interpolation(i, j, k, grid, scheme::CompactWENO, args...) = $symmetric_interpolation(i, j, k, grid, scheme.horizontal_scheme, args...)
        @inline $biased_interpolation(i, j, k, grid, scheme::CompactWENO, args...) = $biased_interpolation(i, j, k, grid, scheme.horizontal_scheme, args...)
    end
end

for interpolation_location in (:zᵃᵃᶠ, :zᵃᵃᶜ)
    symmetric_interpolation = Symbol(:_symmetric_interpolate_, interpolation_location)
    biased_interpolation = Symbol(:_biased_interpolate_, interpolation_location)
    @eval begin
        @inline $symmetric_interpolation(i, j, k, grid, scheme::CompactWENO, args...) = $symmetric_interpolation(i, j, k, grid, scheme.advecting_velocity_scheme, args...)
        @inline $biased_interpolation(i, j, k, grid, scheme::CompactWENO, args...) = $biased_interpolation(i, j, k, grid, scheme.buffer_scheme, args...)
    end
end

@inline advective_tracer_flux_z(i, j, k, grid, scheme::CompactWENO, ::ExplicitTimeDiscretization, W, c) =
    @inbounds Azᶜᶜᶠ(i, j, k, grid) * upwind_biased_product(W[i, j, k],
                                                           scheme.reconstructed_variable.left[i, j, k],
                                                           scheme.reconstructed_variable.right[i, j, k])

#####
##### Nonuniform compact candidate coefficients and optimal weights (built on the CPU,
##### in Float64, once per scheme materialization)
#####

compact_polynomial_cell_average(p, ζL, ζR) = (ζR^(p+1) - ζL^(p+1)) / ((p + 1) * (ζR - ζL))

# Candidate relation  a ĉ[fL] + (1-a) ĉ[fR] = γ c̄[cA] + (1-γ) c̄[cB], exact for cell averages
# of linear and quadratic polynomials. Coordinates are localized at face k and scaled by the
# spacing below it so the error functionals are O(1).
function fitted_compact_candidate(zᶠ, k, candidate_faces, candidate_cells)
    h = zᶠ[k] - zᶠ[k-1]
    ζ(z) = (z - zᶠ[k]) / h
    point(face, p) = ζ(zᶠ[face])^p
    average(cell, p) = compact_polynomial_cell_average(p, ζ(zᶠ[cell]), ζ(zᶠ[cell+1]))

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

left_biased_compact_stencils(k) = (((k-1, k  ), (k-2, k-1)),
                                   ((k-1, k  ), (k-1, k  )),
                                   ((k,   k+1), (k-1, k  )))

# Mirror image about face k: cell j ↦ cell 2k-j-1, face f ↦ face 2k-f
right_biased_compact_stencils(k) = (((k+1, k  ), (k+1, k  )),
                                    ((k+1, k  ), (k,   k-1)),
                                    ((k,   k-1), (k,   k-1)))

# Tables are stored as plain (Nz+1, m) matrices of FT: every backend can allocate and
# copy those, while arrays of tuples are not portable across GPU backends.
function compact_coefficient_table(zᶠ, biased_stencils, valid_face_range, FT)
    Nz = length(zᶠ) - 1
    a = fill(FT(NaN), Nz + 1, 3)
    γ = fill(FT(NaN), Nz + 1, 3)
    optimal5 = fill(FT(NaN), Nz + 1, 3)
    optimal3 = fill(FT(NaN), Nz + 1, 2)

    for k in valid_face_range
        fits = [fitted_compact_candidate(zᶠ, k, faces, cells) for (faces, cells) in biased_stencils(k)]

        E = [fits[m][3](p) for p in (3, 4), m in 1:3]
        weights5 = vcat([1.0 1.0 1.0], E) \ [1.0, 0.0, 0.0]

        # third-order pair (candidates 1, 2 couple only faces {k-1, k}): fourth-order
        # accurate combination used for the near-boundary compact closure rows
        weights3 = [1.0 1.0; E[1, 1] E[1, 2]] \ [1.0, 0.0]

        for m in 1:3
            a[k, m] = FT(fits[m][1])
            γ[k, m] = FT(fits[m][2])
            optimal5[k, m] = FT(weights5[m])
        end
        optimal3[k, 1] = FT(weights3[1])
        optimal3[k, 2] = FT(weights3[2])
    end

    return (; a, γ, optimal5, optimal3)
end

function device_table(arch, host_table::NamedTuple)
    move(host) = (device = zeros(arch, eltype(host), size(host)...); copyto!(device, host); device)
    return map(move, host_table)
end

function compact_weno_coefficient_tables(grid, FT)
    zᶠ = Float64.(collect(znodes(grid, Face())))
    Nz = length(zᶠ) - 1
    arch = architecture(grid)
    left  = device_table(arch, compact_coefficient_table(zᶠ, left_biased_compact_stencils,  3:Nz,   FT))
    right = device_table(arch, compact_coefficient_table(zᶠ, right_biased_compact_stencils, 2:Nz-1, FT))
    return (; left, right)
end

#####
##### Tridiagonal coefficients, computed on the fly inside the solver sweep
#####

# Marker for the solver's row-based coefficient interface: each row of the compact
# system is computed once per sweep through `get_row`.
struct CompactWENORow end

function materialize_advection(scheme::CompactWENO{FT, WCT}, grid) where {FT, WCT}
    topology(grid, 3) === Bounded ||
        throw(ArgumentError("CompactWENO requires a `Bounded` vertical topology; " *
                            "the grid has $(topology(grid, 3)) in z."))

    coefficients = compact_weno_coefficient_tables(grid, FT)

    tridiagonal_solver = BatchedTridiagonalSolver(grid; lower_diagonal = CompactWENORow(),
                                                        diagonal = CompactWENORow(),
                                                        upper_diagonal = CompactWENORow())

    reconstructed_variable = (left = ZFaceField(grid), right = ZFaceField(grid))

    return CompactWENO{FT, WCT}(materialize_advection(scheme.horizontal_scheme, grid),
                                materialize_advection(scheme.buffer_scheme, grid),
                                materialize_advection(scheme.advecting_velocity_scheme, grid),
                                scheme.time_discretization,
                                coefficients,
                                tridiagonal_solver,
                                reconstructed_variable)
end

#####
##### Row computation: fifth-order compact WENO rows in the interior, third-order compact WENO
##### rows near boundaries, explicit buffer-scheme rows at the outermost faces. Each row is
##### evaluated in registers during the solver sweep; nothing is stored.
#####

@inline compact_bias(::Val{:left}) = LeftBias
@inline compact_bias(::Val{:right}) = RightBias

@inline function compact_weno5_left_row(i, j, k, FT, table, c)
    @inbounds begin
        c̄₁ = c[i, j, k-3]
        c̄₂ = c[i, j, k-2]
        c̄₃ = c[i, j, k-1]
        c̄₄ = c[i, j, k]
        c̄₅ = c[i, j, k+1]
        a = (table.a[k, 1], table.a[k, 2], table.a[k, 3])
        γ = (table.γ[k, 1], table.γ[k, 2], table.γ[k, 3])
        optimal = (table.optimal5[k, 1], table.optimal5[k, 2], table.optimal5[k, 3])
    end

    β₁ = FT(13/12) * (c̄₁ - 2c̄₂ + c̄₃)^2 + FT(1/4) * (c̄₁ - 4c̄₂ + 3c̄₃)^2
    β₂ = FT(13/12) * (c̄₂ - 2c̄₃ + c̄₄)^2 + FT(1/4) * (c̄₂ - c̄₄)^2
    β₃ = FT(13/12) * (c̄₃ - 2c̄₄ + c̄₅)^2 + FT(1/4) * (3c̄₃ - 4c̄₄ + c̄₅)^2

    τ = abs(β₁ - β₃)
    ε = convert(FT, 1e-8)
    α₁ = optimal[1] * (1 + (τ / (β₁ + ε))^2)
    α₂ = optimal[2] * (1 + (τ / (β₂ + ε))^2)
    α₃ = optimal[3] * (1 + (τ / (β₃ + ε))^2)
    Σα = α₁ + α₂ + α₃
    ω₁ = α₁ / Σα
    ω₂ = α₂ / Σα
    ω₃ = α₃ / Σα

    lower = ω₁ * a[1] + ω₂ * a[2]
    diag  = ω₁ * (1 - a[1]) + ω₂ * (1 - a[2]) + ω₃ * a[3]
    upper = ω₃ * (1 - a[3])
    rhs = ω₁ * (γ[1] * c̄₂ + (1 - γ[1]) * c̄₃) +
          ω₂ * (γ[2] * c̄₃ + (1 - γ[2]) * c̄₄) +
          ω₃ * (γ[3] * c̄₃ + (1 - γ[3]) * c̄₄)

    return lower, diag, upper, rhs
end

@inline function compact_weno5_right_row(i, j, k, FT, table, c)
    @inbounds begin
        c̄₁ = c[i, j, k+2]
        c̄₂ = c[i, j, k+1]
        c̄₃ = c[i, j, k]
        c̄₄ = c[i, j, k-1]
        c̄₅ = c[i, j, k-2]
        a = (table.a[k, 1], table.a[k, 2], table.a[k, 3])
        γ = (table.γ[k, 1], table.γ[k, 2], table.γ[k, 3])
        optimal = (table.optimal5[k, 1], table.optimal5[k, 2], table.optimal5[k, 3])
    end

    β₁ = FT(13/12) * (c̄₁ - 2c̄₂ + c̄₃)^2 + FT(1/4) * (c̄₁ - 4c̄₂ + 3c̄₃)^2
    β₂ = FT(13/12) * (c̄₂ - 2c̄₃ + c̄₄)^2 + FT(1/4) * (c̄₂ - c̄₄)^2
    β₃ = FT(13/12) * (c̄₃ - 2c̄₄ + c̄₅)^2 + FT(1/4) * (3c̄₃ - 4c̄₄ + c̄₅)^2

    τ = abs(β₁ - β₃)
    ε = convert(FT, 1e-8)
    α₁ = optimal[1] * (1 + (τ / (β₁ + ε))^2)
    α₂ = optimal[2] * (1 + (τ / (β₂ + ε))^2)
    α₃ = optimal[3] * (1 + (τ / (β₃ + ε))^2)
    Σα = α₁ + α₂ + α₃
    ω₁ = α₁ / Σα
    ω₂ = α₂ / Σα
    ω₃ = α₃ / Σα

    upper = ω₁ * a[1] + ω₂ * a[2]
    diag  = ω₁ * (1 - a[1]) + ω₂ * (1 - a[2]) + ω₃ * a[3]
    lower = ω₃ * (1 - a[3])
    rhs = ω₁ * (γ[1] * c̄₂ + (1 - γ[1]) * c̄₃) +
          ω₂ * (γ[2] * c̄₃ + (1 - γ[2]) * c̄₄) +
          ω₃ * (γ[3] * c̄₃ + (1 - γ[3]) * c̄₄)

    return lower, diag, upper, rhs
end

@inline function compact_weno3_left_row(i, j, k, FT, table, c)
    @inbounds begin
        c̄₁ = c[i, j, k-2]
        c̄₂ = c[i, j, k-1]
        c̄₃ = c[i, j, k]
        a = (table.a[k, 1], table.a[k, 2], table.a[k, 3])
        γ = (table.γ[k, 1], table.γ[k, 2], table.γ[k, 3])
        optimal = (table.optimal3[k, 1], table.optimal3[k, 2])
    end

    β₁ = (c̄₁ - c̄₂)^2
    β₂ = (c̄₂ - c̄₃)^2

    τ = abs(β₁ - β₂)
    ε = convert(FT, 1e-8)
    α₁ = optimal[1] * (1 + (τ / (β₁ + ε))^2)
    α₂ = optimal[2] * (1 + (τ / (β₂ + ε))^2)
    Σα = α₁ + α₂
    ω₁ = α₁ / Σα
    ω₂ = α₂ / Σα

    lower = ω₁ * a[1] + ω₂ * a[2]
    diag  = ω₁ * (1 - a[1]) + ω₂ * (1 - a[2])
    upper = zero(FT)
    rhs = ω₁ * (γ[1] * c̄₁ + (1 - γ[1]) * c̄₂) +
          ω₂ * (γ[2] * c̄₂ + (1 - γ[2]) * c̄₃)

    return lower, diag, upper, rhs
end

@inline function compact_weno3_right_row(i, j, k, FT, table, c)
    @inbounds begin
        c̄₁ = c[i, j, k+1]
        c̄₂ = c[i, j, k]
        c̄₃ = c[i, j, k-1]
        a = (table.a[k, 1], table.a[k, 2], table.a[k, 3])
        γ = (table.γ[k, 1], table.γ[k, 2], table.γ[k, 3])
        optimal = (table.optimal3[k, 1], table.optimal3[k, 2])
    end

    β₁ = (c̄₁ - c̄₂)^2
    β₂ = (c̄₂ - c̄₃)^2

    τ = abs(β₁ - β₂)
    ε = convert(FT, 1e-8)
    α₁ = optimal[1] * (1 + (τ / (β₁ + ε))^2)
    α₂ = optimal[2] * (1 + (τ / (β₂ + ε))^2)
    Σα = α₁ + α₂
    ω₁ = α₁ / Σα
    ω₂ = α₂ / Σα

    upper = ω₁ * a[1] + ω₂ * a[2]
    diag  = ω₁ * (1 - a[1]) + ω₂ * (1 - a[2])
    lower = zero(FT)
    rhs = ω₁ * (γ[1] * c̄₁ + (1 - γ[1]) * c̄₂) +
          ω₂ * (γ[2] * c̄₂ + (1 - γ[2]) * c̄₃)

    return lower, diag, upper, rhs
end

@inline function compact_cells_active(i, j, grid, kmin, kmax)
    active = true
    for k in kmin:kmax
        active = active & !inactive_cell(i, j, k, grid)
    end
    return active
end

@inline function compact_row(i, j, k, grid, table, buffer_scheme, c, side)
    Nz = size(grid, 3)
    FT = eltype(grid)

    if side == LeftBias
        if (4 ≤ k ≤ Nz - 1) & compact_cells_active(i, j, grid, k-3, k+1)
            return compact_weno5_left_row(i, j, k, FT, table, c)
        elseif (3 ≤ k ≤ Nz) & compact_cells_active(i, j, grid, k-2, k)
            return compact_weno3_left_row(i, j, k, FT, table, c)
        else
            return zero(FT), one(FT), zero(FT), _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, buffer_scheme, LeftBias, c)
        end
    else
        if (3 ≤ k ≤ Nz - 2) & compact_cells_active(i, j, grid, k-2, k+2)
            return compact_weno5_right_row(i, j, k, FT, table, c)
        elseif (2 ≤ k ≤ Nz - 1) & compact_cells_active(i, j, grid, k-1, k+1)
            return compact_weno3_right_row(i, j, k, FT, table, c)
        else
            return zero(FT), one(FT), zero(FT), _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, buffer_scheme, RightBias, c)
        end
    end
end

const CRW = CompactWENORow

@inline get_row(i, j, k, grid, ::CRW, ::CRW, ::CRW, ::CRW, p, ::ZDirection, table, buffer_scheme, c, side) =  
    compact_row(i, j, k, grid, table, buffer_scheme, c, side)

# the top face is not part of the k = 1..Nz tridiagonal system
@kernel function _set_top_face_reconstruction!(ĉ, grid, buffer_scheme, c, side)
    i, j = @index(Global, NTuple)
    kᴺ   = size(grid, 3)
    @inbounds ĉ[i, j, kᴺ+1] = _biased_interpolate_zᵃᵃᶠ(i, j, kᴺ+1, grid, buffer_scheme, side, c)
end

#####
##### precompute_advection!: called from the hydrostatic time stepping once per stage,
##### right before the tracer tendency computation.
#####

precompute_advection!(scheme, velocities, c) = nothing
precompute_advection!(scheme::FluxFormAdvection, velocities, c) = precompute_advection!(scheme.z, velocities, c)

function precompute_advection!(advection::NamedTuple, velocities, tracers::NamedTuple)
    for name in keys(tracers)
        precompute_advection!(advection[name], velocities, tracers[name])
    end
    return nothing
end

precompute_advection!(scheme::CompactWENO, velocities, c) = compute_compact_reconstruction!(scheme, c)

function compute_compact_reconstruction!(scheme::CompactWENO, c)
    grid = c.grid
    arch = architecture(grid)

    for (side, table, ĉ) in ((LeftBias,  scheme.coefficients.left,  scheme.reconstructed_variable.left),
                             (RightBias, scheme.coefficients.right, scheme.reconstructed_variable.right))

        solve!(ĉ, scheme.tridiagonal_solver, CompactWENORow(), table, scheme.buffer_scheme, c, side)
        launch!(arch, grid, :xy, _set_top_face_reconstruction!, ĉ, grid, scheme.buffer_scheme, c, side)
    end

    return nothing
end
