using Oceananigans.Operators: interpolation_code
using Oceananigans.AbstractOperations: flip
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve_batched_tridiagonal_system!

#####
##### Vertically implicit solver
#####

struct VerticallyImplicitDiffusionSolver{H, Z}
    tracer_solver :: H # also used for horizontal velocities
    w_velocity_solver :: Z
end

"""
    z_viscosity(closure, diffusivities)

Returns the "vertical" (z-direction) viscosity associated with `closure`.
"""
function z_viscosity end

"""
    z_diffusivity(closure, diffusivities, ::Val{tracer_index}) where tracer_index

Returns the "vertical" (z-direction) diffusivity associated with `closure` and `tracer_index`.
"""
function z_diffusivity end

implicit_velocity_step!(u, ::Nothing, args...; kwargs...) = NoneEvent()
implicit_tracer_step!(c, ::Nothing, args...; kwargs...) = NoneEvent()

implicit_diffusion_solver(closure::AbstractTurbulenceClosure, args...) = implicit_solver(time_discretization(closure), closure, args...)
implicit_diffusion_solver(::ExplicitTimeDiscretization, args...) = nothing

#####
##### Solver kernel functions for tracers / horizontal velocities and for vertical velocities
##### Note: "ivd" stands for implicit vertical diffusion.

@inline κ_Δz²(i, j, kᶠ, kᶜ, grid, κ) = κ / Δzᵃᵃᶜ(i, j, kᶜ, grid) / Δzᵃᵃᶠ(i, j, kᶠ, grid)

# Tracers and horizontal velocities at cell centers in z

@inline function ivd_upper_diagonalᵃᵃᶜ(i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    κᵏ⁺¹ = κ⁻⁻ᶠ(i, j, k+1, grid, clock, κ)
    return - Δt * κ_Δz²(i, j, k, k+1, grid, κᵏ⁺¹)
end

@inline function ivd_lower_diagonalᵃᵃᶜ(i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    κᵏ = κ⁻⁻ᶠ(i, j, k′, grid, clock, κ)
    return - Δt * κ_Δz²(i, j, k′, k′, grid, κᵏ)
end

@inline function ivd_diagonalᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT}, clock, Δt, κ⁻⁻ᶠ, κ) where FT
    κᵏ   = κ⁻⁻ᶠ(i, j, k,   grid, clock, κ)
    κᵏ⁺¹ = κ⁻⁻ᶠ(i, j, k+1, grid, clock, κ)

    return ifelse(k == 1,      one(FT) + Δt *  κ_Δz²(i, j, k, k+1, grid, κᵏ⁺¹),
           ifelse(k < grid.Nz, one(FT) + Δt * (κ_Δz²(i, j, k, k+1, grid, κᵏ⁺¹) + κ_Δz²(i, j, k, k, grid, κᵏ)),
                               one(FT) + Δt *                                    κ_Δz²(i, j, k, k, grid, κᵏ)))
end

# Vertical velocity kernel functions (at cell interfaces in z)

@inline function ivd_upper_diagonalᵃᵃᶠ(i, j, k, grid::AbstractGrid{FT}, clock, Δt, νᶜᶜᶜ, ν) where FT
    return ifelse(k == 1,
                  zero(FT), 
                  - Δt * κ_Δz²(i, j, k, k, grid, νᶜᶜᶜ(i, j, k, grid, clock, ν)))
end

@inline function ivd_lower_diagonalᵃᵃᶠ(i, j, k, grid, clock, Δt, νᶜᶜᶜ, ν)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    νᵏ⁻¹ = νᶜᶜᶜ(i, j, k′ - 1, grid, clock, κ)
    return - Δt * κ_Δz²(i, j, k′, k′-1, grid, νᵏ⁻¹)
end

@inline function ivd_diagonalᵃᵃᶜ(i, j, k, grid::AbstractGrid{FT}, clock, Δt, νᶜᶜᶜ, ν) where FT
    νᵏ⁻¹ = νᶜᶜᶜ(i, j, k-1, grid, clock, ν)
    νᵏ   = νᶜᶜᶜ(i, j, k,   grid, clock, ν)

    return ifelse(k == 1,
                  one(FT),
                  1 + Δt * (κ_Δz²(i, j, k, k, grid, νᵏ) + κ_Δz²(i, j, k, k-1, grid, νᵏ⁻¹)))
end

#####
##### Solver constructor
#####

"""
    implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid)

Build a tridiagonal solver for the elliptic equation

```math
(1 + Δt ∂z κ ∂z) cⁿ⁺¹ = c★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid; with_w_velocity_solver)

    topo = topology(grid)

    topo[3] == Periodic && error("VerticallyImplicitTimeDiscretization can only be specified on " *
                                 "grids that are Bounded in the z-direction.")

    # Scratch memory for right_hand_side
    right_hand_side = arch_array(arch, zeros(grid.Nx, grid.Ny, grid.Nz))

    tracer_solver = BatchedTridiagonalSolver(arch, grid;
                                             lower_diagonal = ivd_lower_diagonalᵃᵃᶜ,
                                             diagonal = ivd_diagonalᵃᵃᶜ,
                                             upper_diagonal = ivd_upper_diagonalᵃᵃᶜ,
                                             right_hand_side = right_hand_side)

    if with_w_velocity_solver
        w_velocity_solver = BatchedTridiagonalSolver(arch, grid;
                                                     lower_diagonal = ivd_lower_diagonalᵃᵃᶠ,
                                                     diagonal = ivd_diagonalᵃᵃᶠ,
                                                     upper_diagonal = ivd_upper_diagonalᵃᵃᶠ,
                                                     right_hand_side = right_hand_side)
    else
        w_velocity_solver = nothing
    end

    return VerticallyImplicitDiffusionSolver(tracer_solver, w_velocity_solver)
end

#####
##### Implicit step functions
#####

is_c_location(loc) = loc === (Center, Center, Center)
is_u_location(loc) = loc === (Face, Center, Center)
is_v_location(loc) = loc === (Center, Face, Center)
is_w_location(loc) = loc === (Center, Center, Face)

"""
    implicit_step!(field, solver::VerticallyImplicitDiffusionSolver, clock, Δt, κ⁻⁻ᶠ, κ; dependencies)

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments
`clock, Δt, κ⁻⁻ᶠ, κ` into the coefficient functions that return coefficients of the
lower diagonal, diagonal, and upper diagonal of the resulting tridiagonal system.
"""
function implicit_step!(velocity_field,
                        implicit_solver::VerticallyImplicitDiffusionSolver,
                        clock,
                        Δt,
                        field_location,
                        closure,
                        diffusivities;
                        dependencies = Event(device(model.architecture)))

    if is_c_location(field_location)

        coeff = z_diffusivity(closure, diffusivities, Val(tracer_index))
        locate_coeff = κᶜᶜᶠ
        solver = implicit_solver.tracer_solver

    elseif is_u_location(field_location)

        locate_coeff = νᶠᶜᶠ
        coeff = z_viscosity(closure, diffusivities)
        solver = implicit_solver.tracer_solver

    elseif is_v_location(field_location)

        locate_coeff = νᶜᶠᶠ
        coeff = z_viscosity(closure, diffusivities)
        solver = implicit_solver.tracer_solver

    elseif is_w_location(field_location)

        locate_coeff = νᶜᶜᶜ
        coeff = z_viscosity(closure, diffusivities)
        solver = implicit_solver.w_velocity_solver

    else
        error("Cannot take an implicit_step! for a field at $field_location")
    end

    field_interior = interior(field)
    solver.f .= field_interior

    return solve_batched_tridiagonal_system!(field,
                                             solver, clock, Δt, locate_coeff, coeff;
                                             dependencies = dependencies)
end
