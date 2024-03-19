using Oceananigans.Operators: Δz
using Oceananigans.AbstractOperations: flip
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!
using Oceananigans.Grids: ZDirection

import Oceananigans.Solvers: get_coefficient

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

# Fallbacks: extend these function for `closure` to support.
# TODO: docstring
@inline implicit_linear_coefficient(i, j, k, grid, closure, diffusivity_fields, tracer_index, ℓx, ℓy, ℓz, clock, Δt, κz) =
    zero(grid)

@inline νzᶠᶜᶠ(i, j, k, grid, closure, diffusivity_fields, clock, args...) = zero(grid) # u
@inline νzᶜᶠᶠ(i, j, k, grid, closure, diffusivity_fields, clock, args...) = zero(grid) # v
@inline νzᶜᶜᶜ(i, j, k, grid, closure, diffusivity_fields, clock, args...) = zero(grid) # w
@inline κzᶜᶜᶠ(i, j, k, grid, closure, diffusivity_fields, tracer_index, clock, args...) = zero(grid) # tracers

#####
##### Batched Tridiagonal solver for implicit diffusion
#####

implicit_step!(field, ::Nothing, args...; kwargs...) = nothing
implicit_diffusion_solver(::ExplicitTimeDiscretization, args...; kwargs...) = nothing

#####
##### Solver kernel functions for tracers / horizontal velocities and for vertical velocities
##### Note: "ivd" stands for implicit vertical diffusion.
#####

const c = Center()
const f = Face()

# Tracers and horizontal velocities at cell centers in z
@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Center, clock, Δt, κz)
    closure_ij = getclosure(i, j, closure)
    κᵏ⁺¹   = κz(i, j, k+1, grid, closure_ij, K, id, clock)
    Δzᶜₖ   = Δz(i, j, k,   grid, ℓx, ℓy, c)
    Δzᶠₖ₊₁ = Δz(i, j, k+1, grid, ℓx, ℓy, f)
    du     = - Δt * κᵏ⁺¹ / (Δzᶜₖ * Δzᶠₖ₊₁)

    # This conditional ensures the diagonal is correct
    return ifelse(k > grid.Nz-1, zero(grid), du)
end

@inline function ivd_lower_diagonal(i, j, k′, grid, closure, K, id, ℓx, ℓy, ::Center, clock, Δt, κz)
    k = k′ + 1 # Shift index to match LinearAlgebra.Tridiagonal indexing convenction
    closure_ij = getclosure(i, j, closure)  
    κᵏ   = κz(i, j, k, grid, closure_ij, K, id, clock)
    Δzᶜₖ = Δz(i, j, k, grid, ℓx, ℓy, c)
    Δzᶠₖ = Δz(i, j, k, grid, ℓx, ℓy, f)
    dl   = - Δt * κᵏ / (Δzᶜₖ * Δzᶠₖ)

    # This conditional ensures the diagonal is correct: the lower diagonal does not
    # exist for k′ = 0. (Note we use LinearAlgebra.Tridiagonal indexing convention,
    # so that lower_diagonal should be defined for k′ = 1 ⋯ N-1).
    return ifelse(k′ < 1, zero(grid), dl)
end

#####
##### Vertical velocity kernel functions (at cell interfaces in z)
#####
##### Note: these coefficients are specific to vertically-bounded grids (and so is
##### the BatchedTridiagonalSolver).

@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Face, clock, Δt, νzᶜᶜᶜ) 
    closure_ij = getclosure(i, j, closure)  
    νᵏ = νzᶜᶜᶜ(i, j, k, grid, closure_ij, K, clock)
    Δzᶜₖ = Δz(i, j, k, grid, ℓx, ℓy, c)
    Δzᶠₖ = Δz(i, j, k, grid, ℓx, ℓy, f)
    du   = - Δt * νᵏ / (Δzᶜₖ * Δzᶠₖ)
    return ifelse(k < 1, zero(grid), du)
end

@inline function ivd_lower_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ::Face, clock, Δt, νzᶜᶜᶜ)
    k′ = k + 2 # Shift to adjust for Tridiagonal indexing convention
    closure_ij = getclosure(i, j, closure)  
    νᵏ⁻¹   = νzᶜᶜᶜ(i, j, k′-1, grid, closure_ij, K, clock)
    Δzᶜₖ   = Δz(i, j, k′,   grid, ℓx, ℓy, c)
    Δzᶠₖ₋₁ = Δz(i, j, k′-1, grid, ℓx, ℓy, f)
    dl     = - Δt * νᵏ⁻¹ / (Δzᶜₖ * Δzᶠₖ₋₁)
    return ifelse(k < 1, zero(grid), dl)
end

### Diagonal terms

@inline ivd_diagonal(i, j, k, grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz) =
    one(grid) - Δt * _implicit_linear_coefficient(i, j, k,   grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz) -
                              _ivd_upper_diagonal(i, j, k,   grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz) -
                              _ivd_lower_diagonal(i, j, k-1, grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)

@inline _implicit_linear_coefficient(args...) = implicit_linear_coefficient(args...)
@inline _ivd_upper_diagonal(args...) = ivd_upper_diagonal(args...)
@inline _ivd_lower_diagonal(args...) = ivd_lower_diagonal(args...)

#####
##### Solver constructor
#####

struct VerticallyImplicitDiffusionLowerDiagonal end
struct VerticallyImplicitDiffusionDiagonal end
struct VerticallyImplicitDiffusionUpperDiagonal end

"""
    implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, grid)

Build tridiagonal solvers for the elliptic equations

```math
(1 - Δt ∂z κz ∂z - Δt L) cⁿ⁺¹ = c★
```

and

```math
(1 - Δt ∂z νz ∂z - Δt L) wⁿ⁺¹ = w★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical,
and `wⁿ⁺¹` and `w★` lives at cell `Face`s in the vertical.
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
@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionLowerDiagonal, p, ::ZDirection, args...) = _ivd_lower_diagonal(i, j, k, grid, args...)
@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionUpperDiagonal, p, ::ZDirection, args...) = _ivd_upper_diagonal(i, j, k, grid, args...)
@inline get_coefficient(i, j, k, grid, ::VerticallyImplicitDiffusionDiagonal,      p, ::ZDirection, args...) = ivd_diagonal(i, j, k, grid, args...)

#####
##### Implicit step functions
#####

# Special viscosity extractors with tracer_index === nothing
@inline νzᶠᶜᶠ(i, j, k, grid, closure, K, ::Nothing, clock, args...) = νzᶠᶜᶠ(i, j, k, grid, closure, K, clock, args...)
@inline νzᶜᶠᶠ(i, j, k, grid, closure, K, ::Nothing, clock, args...) = νzᶜᶠᶠ(i, j, k, grid, closure, K, clock, args...)

is_vertically_implicit(closure) = time_discretization(closure) isa VerticallyImplicitTimeDiscretization

"""
    implicit_step!(field, implicit_solver::BatchedTridiagonalSolver,
                   closure, diffusivity_fields, tracer_index, clock, Δt)

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments
`clock, Δt, κ⁻⁻ᶠ, κ` into the coefficient functions that return coefficients of the
lower diagonal, diagonal, and upper diagonal of the resulting tridiagonal system.

`args...` are passed into `z_diffusivity` and `z_viscosity` appropriately for the purpose of retrieving
the diffusivities / viscosities associated with `closure`.
"""
function implicit_step!(field::Field,
                        implicit_solver::BatchedTridiagonalSolver,
                        closure::Union{AbstractTurbulenceClosure, AbstractArray{<:AbstractTurbulenceClosure}, Tuple},
                        diffusivity_fields,
                        tracer_index,
                        clock,
                        Δt; 
                        kwargs...)
    
   loc = location(field)

   # << Look at all these assumptions >>
   # Or put another way, `location(field)` serves to identify velocity components.
   # Change this if `location(field)` does not uniquely identify velocity components.
   κz = # "Extractor function
       loc === (Center, Center, Center) ? κzᶜᶜᶠ :
       loc === (Face, Center, Center)   ? νzᶠᶜᶠ :
       loc === (Center, Face, Center)   ? νzᶜᶠᶠ :
       loc === (Center, Center, Face)   ? νzᶜᶜᶜ :
       error("Cannot take an implicit_step! for a field at $location")

    # Nullify tracer_index if `field` is not a tracer   
    κz === κzᶜᶜᶠ || (tracer_index = nothing)

    # Filter explicit closures for closure tuples
    if closure isa Tuple
        closure_tuple = closure
        N = length(closure_tuple)
        vi_closure            = Tuple(closure[n]            for n = 1:N if is_vertically_implicit(closure[n]))
        vi_diffusivity_fields = Tuple(diffusivity_fields[n] for n = 1:N if is_vertically_implicit(closure[n]))
    else
        vi_closure = closure
        vi_diffusivity_fields = diffusivity_fields
    end

    return solve!(field, implicit_solver, field,
                  # ivd_*_diagonal gets called with these args after (i, j, k, grid):
                  vi_closure, vi_diffusivity_fields, tracer_index, map(ℓ -> ℓ(), loc)..., clock, Δt, κz; kwargs...)
end

