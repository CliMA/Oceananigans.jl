using Oceananigans.Advection: implicit_advection_upper_diagonal,
                              implicit_advection_lower_diagonal,
                              implicit_advection_diagonal
using Oceananigans.BoundaryConditions: implicit_flux_coefficient, immersed_implicit_flux_coefficient, needs_implicit_solver
using Oceananigans.Fields: location
using Oceananigans.Grids: Periodic, ZDirection, topology
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, ImmersedBoundaryCondition, immersed_inactive_node
using Oceananigans.Operators: Δz
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!

import Oceananigans.Solvers: get_coefficient
import Oceananigans.TimeSteppers: implicit_step!

const IBG = ImmersedBoundaryGrid

#####
##### implicit_step! interface
#####
##### Closures with `VerticallyImplicitTimeDiscretization` can define
#####
##### 1. "Coefficient extractors" `νz` and `κz` to support vertically-implicit
#####    treatment of a diffusive term iwth the form `∂z κz ∂z ϕ` for a variable `ϕ`.
#####    There are three extractors for momentum (`νz`) and one for tracers (`κz`)
#####    relevant to implicit vertical diffusion.
#####
##### 2. `implicit_linear_coefficient` to support the implicit treament of a _linear_ term.
#####

const c = Center()
const f = Face()
const C = Center
const F = Face

# Fallbacks: extend these function for `closure` to support.
@inline implicit_linear_coefficient(i, j, k, grid, args...) = zero(grid)

# General implementation
@inline νzᶠᶜᶠ(i, j, k, grid, args...) = zero(grid)
@inline νzᶜᶠᶠ(i, j, k, grid, args...) = zero(grid)
@inline νzᶜᶜᶜ(i, j, k, grid, args...) = zero(grid)
@inline κzᶜᶜᶠ(i, j, k, grid, args...) = zero(grid)

# Vertical momentum diffusivities: u, v, w
@inline ivd_diffusivity(i, j, k, grid, ::F, ::C, ::F, clo, K, id, clk, fields) = νzᶠᶜᶠ(i, j, k, grid, clo, K, id, clk, fields) * !inactive_node(i, j, k, grid, f, c, f)
@inline ivd_diffusivity(i, j, k, grid, ::C, ::F, ::F, clo, K, id, clk, fields) = νzᶜᶠᶠ(i, j, k, grid, clo, K, id, clk, fields) * !inactive_node(i, j, k, grid, c, f, f)
@inline ivd_diffusivity(i, j, k, grid, ::C, ::C, ::C, clo, K, id, clk, fields) = νzᶜᶜᶜ(i, j, k, grid, clo, K, id, clk, fields) * !inactive_node(i, j, k, grid, c, c, c)

# Tracer diffusivity
@inline ivd_diffusivity(i, j, k, grid, ::C, ::C, ::F, clo, K, id, clk, fields) = κzᶜᶜᶠ(i, j, k, grid, clo, K, id, clk, fields) * !inactive_node(i, j, k, grid, c, c, f)

#####
##### Batched Tridiagonal solver for implicit diffusion
#####

implicit_diffusion_solver(::ExplicitTimeDiscretization, args...; kwargs...) = nothing

#####
##### Solver kernel functions for tracers / horizontal velocities and for vertical velocities
##### Note: "ivd" stands for implicit vertical diffusion.
#####

# local definition of the generic reciprocal function Δz⁻¹
@inline Δz⁻¹(i, j, k, grid, ℓx, ℓy, ℓz) = 1 / Δz(i, j, k, grid, ℓx, ℓy, ℓz)

# Tracers and horizontal velocities at cell centers in z
@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Center, Δt, clock, fields)
    closure_ij = getclosure(i, j, closure)
    κᵏ⁺¹     = ivd_diffusivity(i, j, k+1, grid, ℓx, ℓy, f, closure_ij, K, id, clock, fields)
    Δz⁻¹ᶜₖ   = Δz⁻¹(i, j, k,   grid, ℓx, ℓy, c)
    Δz⁻¹ᶠₖ₊₁ = Δz⁻¹(i, j, k+1, grid, ℓx, ℓy, f)
    du       = - Δt * κᵏ⁺¹ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ₊₁)
    # This conditional ensures the diagonal is correct
    return du * !peripheral_node(i, j, k+1, grid, ℓx, ℓy, f)
end

@inline function ivd_lower_diagonal(i, j, k′, grid, closure, K, id, ℓx, ℓy, ::Center, Δt, clock, fields)
    k = k′ + 1 # Shift index to match LinearAlgebra.Tridiagonal indexing convenction
    closure_ij = getclosure(i, j, closure)
    κᵏ     = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, f, closure_ij, K, id, clock, fields)
    Δz⁻¹ᶜₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, c)
    Δz⁻¹ᶠₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, f)
    dl     = - Δt * κᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)

    # This conditional ensures the diagonal is correct. (Note we use LinearAlgebra.Tridiagonal
    # indexing convention, so that lower_diagonal should be defined for k′ = 1 ⋯ N-1.)
    return dl * !peripheral_node(i, j, k′, grid, ℓx, ℓy, c)
end

#####
##### Vertical velocity kernel functions (at cell interfaces in z)
#####
##### Note: these coefficients are specific to vertically-bounded grids (and so is
##### the BatchedTridiagonalSolver).

@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Face, Δt, clock, fields)
    closure_ij = getclosure(i, j, closure)
    νᵏ     = ivd_diffusivity(i, j, k, grid, ℓx, ℓy, c, closure_ij, K, id, clock, fields)
    Δz⁻¹ᶜₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, c)
    Δz⁻¹ᶠₖ = Δz⁻¹(i, j, k, grid, ℓx, ℓy, f)
    du     = - Δt * νᵏ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ)
    return du * !peripheral_node(i, j, k, grid, ℓx, ℓy, c)
end

# `dl(m)` multiplies `ϕ[m]` in row `m + 1`, and the viscous flux between faces `m` and `m + 1` sits at center `m`
@inline function ivd_lower_diagonal(i, j, m, grid, closure, K, id, ℓx, ℓy, ::Face, Δt, clock, fields)
    closure_ij = getclosure(i, j, closure)
    νᵐ       = ivd_diffusivity(i, j, m,   grid, ℓx, ℓy, c, closure_ij, K, id, clock, fields)
    Δz⁻¹ᶜₘ   = Δz⁻¹(i, j, m,   grid, ℓx, ℓy, c)
    Δz⁻¹ᶠₘ₊₁ = Δz⁻¹(i, j, m+1, grid, ℓx, ℓy, f)
    dl       = - Δt * νᵐ * (Δz⁻¹ᶜₘ * Δz⁻¹ᶠₘ₊₁)
    return dl * !peripheral_node(i, j, m, grid, ℓx, ℓy, c)
end

### Diagonal terms
@inline ivd_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) =
    one(grid) - Δt * _implicit_linear_coefficient(i, j, k,   grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) -
                              _ivd_upper_diagonal(i, j, k,   grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) -
                              _ivd_lower_diagonal(i, j, k-1, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields)


# Fallback for single closure. These coefficients are extended for tupled closures in `closure_tuples.jl`
@inline _implicit_linear_coefficient(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) =
    implicit_linear_coefficient(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields)

@inline _ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) =
    ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields)

@inline _ivd_lower_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields) =
    ivd_lower_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, Δt, clock, fields)

# When closure is `nothing`, diffusion contributions are zero.
@inline _implicit_linear_coefficient(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _ivd_upper_diagonal(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _ivd_lower_diagonal(i, j, k, grid, ::Nothing, args...) = zero(grid)

#####
##### Implicit-explicit flux boundary conditions: the linear flux coefficient λ is embedded in the
##### boundary-cell diagonal (top: k = Nz, bottom: k = 1).
#####

@inline function boundary_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, top_bc, bottom_bc, immersed_bc)
    # Constant-folds away unless a boundary condition is implicit-explicit
    if !(needs_implicit_solver(top_bc) | needs_implicit_solver(bottom_bc) | needs_implicit_solver(immersed_bc))
        return zero(grid)
    end

    Nz  = size(grid, 3)
    Δzᵏ = Δz(i, j, k, grid, ℓx, ℓy, ℓz)
    λᵗ  = implicit_flux_coefficient(top_bc,    i, j, grid, clk, fields)
    λᵇ  = implicit_flux_coefficient(bottom_bc, i, j, grid, clk, fields)
    dᵗ  = ifelse(k == Nz,  Δt * λᵗ / Δzᵏ, zero(grid))  # top flux:     tendency −J/Δz
    dᵇ  = ifelse(k == 1,  -Δt * λᵇ / Δzᵏ, zero(grid))  # bottom flux:  tendency +J/Δz
    dⁱ  = immersed_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, immersed_bc)
    return dᵗ + dᵇ + dⁱ
end

@inline immersed_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, immersed_bc) = zero(grid)

# Immersed fluxes point along the inward-facing normal on every facet, so both immersed faces contribute
# `+J/Δz` to the tendency, unlike the domain top which contributes `-J/Δz`.
@inline function immersed_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, immersed_bc::ImmersedBoundaryCondition)
    Δzᵏ = Δz(i, j, k, grid, ℓx, ℓy, ℓz)
    active = !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

    # `immersed_inactive_node` is false outside the domain, so a column wet to `k = 1` or `k = Nz`
    # sees no immersed face there and the domain boundary condition is counted once.
    on_bottom = active & immersed_inactive_node(i, j, k-1, grid, ℓx, ℓy, ℓz)
    on_top    = active & immersed_inactive_node(i, j, k+1, grid, ℓx, ℓy, ℓz)

    λᵇ = immersed_implicit_flux_coefficient(immersed_bc.bottom, i, j, k, grid, clk, fields)
    λᵗ = immersed_implicit_flux_coefficient(immersed_bc.top,    i, j, k, grid, clk, fields)

    return ifelse(on_bottom, -Δt * λᵇ / Δzᵏ, zero(grid)) +
           ifelse(on_top,    -Δt * λᵗ / Δzᵏ, zero(grid))
end

#####
##### Solver constructor
#####

struct VerticallyImplicitDiffusionLowerDiagonal end
struct VerticallyImplicitDiffusionDiagonal end
struct VerticallyImplicitDiffusionUpperDiagonal end

"""
$(TYPEDSIGNATURES)

Build tridiagonal solvers for the elliptic equations

```math
(1 - Δt ∂_z κ_z ∂_z - Δt L) cⁿ⁺¹ = c_★
```

and

```math
(1 - Δt ∂_z ν_z ∂_z - Δt L) wⁿ⁺¹ = w_★
```

where ``cⁿ⁺¹`` and ``c_★`` live at cell `Center`s in the vertical,
and ``wⁿ⁺¹`` and ``w_★`` live at cell `Face`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, grid)
    topo = topology(grid)

    topo[3] == Periodic && error("VerticallyImplicitTimeDiscretization can only be specified on " *
                                 "grids that are Bounded in the z-direction.")

    z_solver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = VerticallyImplicitDiffusionLowerDiagonal(),
                                        diagonal       = VerticallyImplicitDiffusionDiagonal(),
                                        upper_diagonal = VerticallyImplicitDiffusionUpperDiagonal())

    return z_solver
end

# Extend `get_coefficient` to retrieve `ivd_diagonal`, `_ivd_lower_diagonal` and `_ivd_upper_diagonal`.
# Note that we use the "periphery-aware" upper and lower diagonals. The trailing arguments are supplied
# by `implicit_step!` below. `density` selects volume-conserving (Boussinesq) versus
# density-weighted (mass-flux) advection coefficients.
@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection, w, density, top_bc, bottom_bc, immersed_bc)
    duκ = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    duw = implicit_advection_upper_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy, ℓz, density)
    return duκ + duw
end

@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection, w, density, top_bc, bottom_bc, immersed_bc)
    dlκ = _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dlw = implicit_advection_lower_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy, ℓz, density)
    return dlκ + dlw
end

@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection, w, density, top_bc, bottom_bc, immersed_bc)
    dκ  = ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dw  = implicit_advection_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy, ℓz, density)
    dbc = boundary_flux_diagonal(i, j, k, grid, ℓx, ℓy, ℓz, Δt, clk, fields, top_bc, bottom_bc, immersed_bc)
    return dκ + dw + dbc
end

#####
##### Implicit step functions
#####

is_vertically_implicit(closure) = TimeSteppers.time_discretization(closure) isa VerticallyImplicitTimeDiscretization

"""
$(TYPEDSIGNATURES)

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments into the coefficient
functions that return coefficients of the lower diagonal, diagonal, and upper diagonal of the
resulting tridiagonal system.

`advection` and `velocities` add the implicit vertical-advection contribution of an adaptive-implicit
scheme; `density` selects density-weighted coefficients for mass-flux (anelastic / compressible)
models, while `nothing` keeps the volume-conserving Boussinesq behavior.
"""
function implicit_step!(field::Field,
                        implicit_solver::BatchedTridiagonalSolver,
                        closure, closure_fields, tracer_index,
                        clock, fields, Δt,
                        advection=nothing, velocities=nothing, density=nothing)

    if closure isa Tuple
        N = length(closure)
        vi_closure        = Tuple(closure[n]        for n = 1:N if is_vertically_implicit(closure[n]))
        vi_closure_fields = Tuple(closure_fields[n] for n = 1:N if is_vertically_implicit(closure[n]))
    elseif closure isa Nothing || !is_vertically_implicit(closure)
        vi_closure = nothing
        vi_closure_fields = nothing
    else
        vi_closure = closure
        vi_closure_fields = closure_fields
    end

    bcs = field.boundary_conditions
    isnothing(vi_closure) && !needs_implicit_solver(advection) && !needs_implicit_solver(bcs) && return nothing

    LX, LY, LZ = location(field)
    w = isnothing(velocities) ? nothing : velocities.w

    return solve!(field, implicit_solver, field,
                  vi_closure, vi_closure_fields, tracer_index,
                  LX(), LY(), LZ(), Δt, clock, fields,
                  advection, w, density, bcs.top, bcs.bottom, bcs.immersed)
end
