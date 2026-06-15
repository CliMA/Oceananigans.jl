using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection,
                              implicit_advection_upper_diagonal,
                              implicit_advection_lower_diagonal,
                              implicit_advection_diagonal
using Oceananigans.Fields: location
using Oceananigans.Grids: Periodic, ZDirection, topology
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
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

@inline function ivd_lower_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Face, Δt, clock, fields)
    k′ = k + 2 # Shift to adjust for Tridiagonal indexing convention
    closure_ij = getclosure(i, j, closure)
    νᵏ⁻¹     = ivd_diffusivity(i, j, k′-1, grid, ℓx, ℓy, c, closure_ij, K, id, clock, fields)
    Δz⁻¹ᶜₖ   = Δz⁻¹(i, j, k′,   grid, ℓx, ℓy, c)
    Δz⁻¹ᶠₖ₋₁ = Δz⁻¹(i, j, k′-1, grid, ℓx, ℓy, f)
    dl       = - Δt * νᵏ⁻¹ * (Δz⁻¹ᶜₖ * Δz⁻¹ᶠₖ₋₁)
    return dl * !peripheral_node(i, j, k, grid, ℓx, ℓy, c)
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

# When closure is `nothing` (e.g. AIVA without turbulence closure), diffusion contributions are zero
@inline _implicit_linear_coefficient(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _ivd_upper_diagonal(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _ivd_lower_diagonal(i, j, k, grid, ::Nothing, args...) = zero(grid)

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
# Note that we use the "periphery-aware" upper and lower diagonals
@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields) =
    _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields) =
    _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields) =
    ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

#####
##### Implicit step functions
#####

is_vertically_implicit(closure) = TimeSteppers.time_discretization(closure) isa VerticallyImplicitTimeDiscretization

# When closure is nothing but solver exists (e.g., created for AIVA),
# the pure-diffusion implicit step is a no-op (no diffusion to solve).
"""
$(TYPEDSIGNATURES)

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments into the coefficient
functions that return coefficients of the lower diagonal, diagonal, and upper diagonal of the
resulting tridiagonal system.
"""
implicit_step!(::Field, ::BatchedTridiagonalSolver, ::Nothing, closure_fields, tracer_index, clock, fields, Δt) = nothing

function implicit_step!(field::Field,
                        implicit_solver::BatchedTridiagonalSolver,
                        closure::Union{AbstractTurbulenceClosure, AbstractArray{<:AbstractTurbulenceClosure}, Tuple},
                        closure_fields,
                        tracer_index,
                        clock,
                        fields,
                        Δt)

    # Filter explicit closures for closure tuples
    if closure isa Tuple
        closure_tuple = closure
        N = length(closure_tuple)
        vi_closure        = Tuple(closure[n]        for n = 1:N if is_vertically_implicit(closure[n]))
        vi_closure_fields = Tuple(closure_fields[n] for n = 1:N if is_vertically_implicit(closure[n]))
    else
        vi_closure = closure
        vi_closure_fields = closure_fields
    end

    LX, LY, LZ = location(field)
    return solve!(field, implicit_solver, field,
                  # ivd_*_diagonal gets called with these args after (i, j, k, grid):
                  vi_closure, vi_closure_fields, tracer_index, LX(), LY(), LZ(), Δt, clock, fields)
end

#####
##### Extended get_coefficient methods for combined implicit diffusion + advection
#####
##### When `advection` and `velocities` are passed as extra args (from the extended implicit_step!),
##### the advection contribution is added to the diffusion coefficients.
#####

const AIVA = AdaptiveImplicitVerticalAdvection

# With AdaptiveImplicitVerticalAdvection: add advection contribution
@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection::AIVA, w)
    du_diff = _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    du_adv = implicit_advection_upper_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy)
    return du_diff + du_adv
end

@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection::AIVA, w)
    dl_diff = _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    dl_adv = implicit_advection_lower_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy)
    return dl_diff + dl_adv
end

@inline function get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal, p, ::ZDirection,
                                 clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields,
                                 advection::AIVA, w)
    d_diff = ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)
    d_adv = implicit_advection_diagonal(i, j, k, grid, advection, w, Δt, ℓx, ℓy)
    return d_diff + d_adv
end

# Fallback: non-adaptive advection schemes contribute nothing to the implicit system
@inline get_coefficient(i, j, k, grid, d::VerticallyImplicitDiffusionUpperDiagonal, p, dir::ZDirection,
                        clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection, w) =
    _ivd_upper_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

@inline get_coefficient(i, j, k, grid, d::VerticallyImplicitDiffusionLowerDiagonal, p, dir::ZDirection,
                        clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection, w) =
    _ivd_lower_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

@inline get_coefficient(i, j, k, grid, d::VerticallyImplicitDiffusionDiagonal, p, dir::ZDirection,
                        clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields, advection, w) =
    ivd_diagonal(i, j, k, grid, clo, K, id, ℓx, ℓy, ℓz, Δt, clk, fields)

#####
##### Extended implicit_step! that passes advection and vertical velocity through
#####

function implicit_step!(field::Field,
                        implicit_solver::BatchedTridiagonalSolver,
                        closure, closure_fields, tracer_index,
                        clock, fields, Δt,
                        advection, velocities)

    if closure isa Tuple
        N = length(closure)
        vi_closure        = Tuple(closure[n]        for n = 1:N if is_vertically_implicit(closure[n]))
        vi_closure_fields = Tuple(closure_fields[n] for n = 1:N if is_vertically_implicit(closure[n]))
    else
        vi_closure = closure
        vi_closure_fields = closure_fields
    end

    LX, LY, LZ = location(field)
    return solve!(field, implicit_solver, field,
                  vi_closure, vi_closure_fields, tracer_index,
                  LX(), LY(), LZ(), Δt, clock, fields,
                  advection, velocities.w)
end
