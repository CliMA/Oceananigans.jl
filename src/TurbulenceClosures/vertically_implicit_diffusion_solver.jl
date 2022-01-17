using Oceananigans.Operators: interpolation_code
using Oceananigans.AbstractOperations: flip
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!

#####
##### Vertically implicit solver
#####

struct VerticallyImplicitDiffusionSolver{A, Z}
    architecture :: A
    z_solver :: Z
end

"""
    z_viscosity(closure, args...)

Returns the "vertical" (z-direction) viscosity associated with `closure`.
"""
function z_viscosity end

"""
    z_diffusivity(closure, ::Val{tracer_index}, args...) where tracer_index

Returns the "vertical" (z-direction) diffusivity associated with `closure` and `tracer_index`.
"""
function z_diffusivity end

implicit_step!(field, ::Nothing, args...; kwargs...) = NoneEvent()
implicit_diffusion_solver(::ExplicitTimeDiscretization, args...; kwargs...) = nothing

#####
##### Solver kernel functions for tracers / horizontal velocities and for vertical velocities
##### Note: "ivd" stands for implicit vertical diffusion.
#####

@inline κ_Δz²(i, j, kᶜ, kᶠ, grid, κ) = κ / Δzᵃᵃᶜ(i, j, kᶜ, grid) / Δzᵃᵃᶠ(i, j, kᶠ, grid)

# Tracers and horizontal velocities at cell centers in z


@inline function ivd_upper_diagonal(LX, LY, LZ::Center, i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    κᵏ⁺¹ = κ⁻⁻ᶠ(i, j, k+1, grid, clock, κ)

    return ifelse(k > grid.Nz-1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k, k+1, grid, κᵏ⁺¹))
end

@inline function ivd_lower_diagonal(LX, LY, LZ::Center, i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    κᵏ = κ⁻⁻ᶠ(i, j, k′, grid, clock, κ)

    return ifelse(k < 1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k′, k′, grid, κᵏ))
end

# Vertical velocity kernel functions (at cell interfaces in z)
#
# Note: these coefficients are specific to vertically-bounded grids (and so is
# the BatchedTridiagonalSolver).

@inline function ivd_upper_diagonal(LX, LY, LZ::Face, i, j, k, grid, clock, Δt, νᶜᶜᶜ, ν)
    νᵏ = νᶜᶜᶜ(i, j, k, grid, clock, ν)

    return ifelse(k < 1, # should this be k < 2?
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k, k, grid, νᵏ))
end

@inline function ivd_lower_diagonal(LX, LY, LZ::Face, i, j, k, grid, clock, Δt, νᶜᶜᶜ, ν)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    νᵏ⁻¹ = νᶜᶜᶜ(i, j, k′-1, grid, clock, ν)
    return ifelse(k < 1,
                  zero(eltype(grid)),
                  - Δt * κ_Δz²(i, j, k′, k′-1, grid, νᵏ⁻¹))
end

### Diagonal terms

@inline ivd_diagonal(LX, LY, i, j, k, grid, clock, Δt, interp_κ, κ) =
    one(eltype(grid)) - ivd_upper_diagonal(LX, LY, LZ, i, j, k, grid, clock, Δt, interp_κ, κ) -
                        ivd_lower_diagonal(LX, LY, LZ, i, j, k-1, grid, clock, Δt, interp_κ, κ)

#####
##### Solver constructor
#####

"""
    implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid)

Build tridiagonal solvers for the elliptic equations

```math
(1 - Δt ∂z κ ∂z) cⁿ⁺¹ = c★
```

and

```math
(1 - Δt ∂z ν ∂z) wⁿ⁺¹ = w★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical,
and `wⁿ⁺¹` and `w★` lives at cell `Face`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid)

    topo = topology(grid)

    topo[3] == Periodic && error("VerticallyImplicitTimeDiscretization can only be specified on " *
                                 "grids that are Bounded in the z-direction.")

    z_solver = BatchedTridiagonalSolver(grid;
                                        lower_diagonal = ivd_lower_diagonal,
                                        diagonal = ivd_diagonal,
                                        upper_diagonal = ivd_upper_diagonal)

    return VerticallyImplicitDiffusionSolver(arch, z_solver)
end

#####
##### Implicit step functions
#####

is_c_location(loc) = loc === (Center, Center, Center)
is_u_location(loc) = loc === (Face, Center, Center)
is_v_location(loc) = loc === (Center, Face, Center)
is_w_location(loc) = loc === (Center, Center, Face)

"""
    implicit_step!(field, solver::VerticallyImplicitDiffusionSolver, clock, Δt,
                   closure, tracer_index, args...; dependencies = Event

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments
`clock, Δt, κ⁻⁻ᶠ, κ` into the coefficient functions that return coefficients of the
lower diagonal, diagonal, and upper diagonal of the resulting tridiagonal system.

`args...` are passed into `z_diffusivity` and `z_viscosity` appropriately for the purpose of retrieving
the diffusivities / viscosities associated with `closure`.
"""
function implicit_step!(field::AbstractField{LX, LY, LZ},
                        implicit_solver::VerticallyImplicitDiffusionSolver,
                        clock,
                        Δt,
                        closure,
                        tracer_index,
                        args...;
                        dependencies) where {LX, LY, LZ}
                        
    if is_c_location((LX, LY, LZ))

        locate_coeff = κᶜᶜᶠ
        coeff = z_diffusivity(closure, Val(tracer_index), args...)

    elseif is_u_location((LX, LY, LZ))

        locate_coeff = νᶠᶜᶠ
        coeff = z_viscosity(closure, args...)

    elseif is_v_location((LX, LY, LZ))

        locate_coeff = νᶜᶠᶠ
        coeff = z_viscosity(closure, args...)

    elseif is_w_location((LX, LY, LZ))

        locate_coeff = νᶜᶜᶜ
        coeff = z_viscosity(closure, args...)

    else
        error("Cannot take an implicit_step! for a field at $field_location")
    end

    solver = implicit_solver.z_solver

    return solve!(field, solver, field,
                  clock, Δt, locate_coeff, coeff; dependencies = dependencies)
end

