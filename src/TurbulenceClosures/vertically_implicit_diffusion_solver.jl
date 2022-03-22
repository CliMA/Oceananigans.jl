using Oceananigans.Operators: Δzᵃᵃᶜ, Δzᵃᵃᶠ
using Oceananigans.AbstractOperations: flip
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!

#####
##### Batched Tridiagonal solver for implicit diffusion
#####

implicit_step!(field, ::Nothing, args...; kwargs...) = NoneEvent()
implicit_diffusion_solver(::ExplicitTimeDiscretization, args...; kwargs...) = nothing

#####
##### Solver kernel functions for tracers / horizontal velocities and for vertical velocities
##### Note: "ivd" stands for implicit vertical diffusion.
#####

@inline κ_Δz²(i, j, kᶜ, kᶠ, grid, κ) = κ / Δzᵃᵃᶜ(i, j, kᶜ, grid) / Δzᵃᵃᶠ(i, j, kᶠ, grid)

instantiate(X) = X()

# Tracers and horizontal velocities at cell centers in z

@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, LX, LY, ::Center, clock, Δt, κz)
    closure_ij = getclosure(i, j, closure)  
    κᵏ⁺¹ = κz(i, j, k+1, grid, closure_ij, K, id, clock)

    return ifelse(k > grid.Nz-1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k, k+1, grid, κᵏ⁺¹))
end

@inline function ivd_lower_diagonal(i, j, k, grid, closure, K, id, LX, LY, ::Center, clock, Δt, κz)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    closure_ij = getclosure(i, j, closure)  
    κᵏ = κz(i, j, k′, grid, closure_ij, K, id, clock)

    return ifelse(k < 1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k′, k′, grid, κᵏ))
end

# Vertical velocity kernel functions (at cell interfaces in z)
#
# Note: these coefficients are specific to vertically-bounded grids (and so is
# the BatchedTridiagonalSolver).
@inline function ivd_upper_diagonal(i, j, k, grid, closure, K, id, LX, LY, ::Face, clock, Δt, νzᶜᶜᶜ) 
    closure_ij = getclosure(i, j, closure)  
    νᵏ = νzᶜᶜᶜ(i, j, k, grid, closure_ij, K, clock)

    return ifelse(k < 1, # should this be k < 2? #should this be grid.Nz - 1?
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k, k, grid, νᵏ))
end

@inline function ivd_lower_diagonal(i, j, k, grid, closure, K, id, LX, LY, ::Face, clock, Δt, νzᶜᶜᶜ)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    closure_ij = getclosure(i, j, closure)  
    νᵏ⁻¹ = νzᶜᶜᶜ(i, j, k′-1, grid, closure_ij, K, clock)
    return ifelse(k < 1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k′, k′-1, grid, νᵏ⁻¹))
end

### Diagonal terms

@inline ivd_diagonal(i, j, k, grid, closure, K, id, LX, LY, LZ, clock, Δt, κz) =
    one(eltype(grid)) - _ivd_upper_diagonal(i, j, k,   grid, closure, K, id, LX, LY, LZ, clock, Δt, κz) -
                        _ivd_lower_diagonal(i, j, k-1, grid, closure, K, id, LX, LY, LZ, clock, Δt, κz)

@inline _ivd_upper_diagonal(args...) = ivd_upper_diagonal(args...)
@inline _ivd_lower_diagonal(args...) = ivd_lower_diagonal(args...)

#####
##### Solver constructor
#####

"""
    implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, grid)

Build tridiagonal solvers for the elliptic equations

```math
(1 - Δt ∂z κz ∂z) cⁿ⁺¹ = c★
```

and

```math
(1 - Δt ∂z νz ∂z) wⁿ⁺¹ = w★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical,
and `wⁿ⁺¹` and `w★` lives at cell `Face`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, grid)
    topo = topology(grid)

    topo[3] == Periodic && error("VerticallyImplicitTimeDiscretization can only be specified on " *
                                 "grids that are Bounded in the z-direction.")

    z_solver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = _ivd_lower_diagonal,
                                        diagonal = ivd_diagonal,
                                        upper_diagonal = _ivd_upper_diagonal)

    return z_solver
end

#####
##### Implicit step functions
#####

# Special viscosity extractors with tracer_index === nothing
@inline νzᶠᶜᶠ(i, j, k, grid, closure, K, ::Nothing, clock) = νzᶠᶜᶠ(i, j, k, grid, closure, K, clock)
@inline νzᶜᶠᶠ(i, j, k, grid, closure, K, ::Nothing, clock) = νzᶜᶠᶠ(i, j, k, grid, closure, K, clock)

is_vertically_implicit(closure) = time_discretization(closure) isa VerticallyImplicitTimeDiscretization

"""
    implicit_step!(field, implicit_solver::BatchedTridiagonalSolver,
                   closure, diffusivity_fields, tracer_index, clock, Δt;
                   dependencies)

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
                        Δt; dependencies)
    
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
        vi_closure = Tuple(closure[n] for n = 1:N if is_vertically_implicit(closure[n]))
        vi_diffusivity_fields = Tuple(diffusivity_fields[n] for n = 1:N if is_vertically_implicit(closure[n]))
    else
        vi_closure = closure
        vi_diffusivity_fields = diffusivity_fields
    end

    return solve!(field, implicit_solver, field,
                  # ivd_*_diagonal gets called with these args after (i, j, k, grid):
                  vi_closure, vi_diffusivity_fields, tracer_index, instantiate.(loc)..., clock, Δt, κz;
                  dependencies)
end

