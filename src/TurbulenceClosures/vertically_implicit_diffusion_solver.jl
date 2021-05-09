using Oceananigans.Operators: interpolation_code
using Oceananigans.AbstractOperations: flip
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve_batched_tridiagonal_system!

#####
##### Vertically implicit solver
#####

struct VerticallyImplicitDiffusionSolver{B}
    batched_tridiagonal_solver :: B
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

@inline κ_Δz²_ᵏ(i, j, k, grid, κᵏ)     = κᵏ   / Δzᵃᵃᶜ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k,   grid)
@inline κ_Δz²_ᵏ⁺¹(i, j, k, grid, κᵏ⁺¹) = κᵏ⁺¹ / Δzᵃᵃᶜ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k+1, grid)

# "ivd" stands for implicit vertical diffusion:

@inline function ivd_upper_diagonal(i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    k′ = k
    κᵏ⁺¹ = κ⁻⁻ᶠ(i, j, k′ + 1, grid, clock, κ)
    return - Δt * κ_Δz²_ᵏ⁺¹(i, j, k′, grid, κᵏ⁺¹)
end

@inline function ivd_lower_diagonal(i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    k′ = k + 1 # Shift to adjust for Tridiagonal indexing convenction
    κᵏ = κ⁻⁻ᶠ(i, j, k′, grid, clock, κ)
    return - Δt * κ_Δz²_ᵏ(i, j, k′, grid, κᵏ)
end

@inline function ivd_diagonal(i, j, k, grid, clock, Δt, κ⁻⁻ᶠ, κ)
    κᵏ   = κ⁻⁻ᶠ(i, j, k,   grid, clock, κ)
    κᵏ⁺¹ = κ⁻⁻ᶠ(i, j, k+1, grid, clock, κ)

    return ifelse(k == 1,      1 + Δt *  κ_Δz²_ᵏ⁺¹(i, j, k, grid, κᵏ⁺¹),
           ifelse(k < grid.Nz, 1 + Δt * (κ_Δz²_ᵏ⁺¹(i, j, k, grid, κᵏ⁺¹) + κ_Δz²_ᵏ(i, j, k, grid, κᵏ)),
                               1 + Δt *                                   κ_Δz²_ᵏ(i, j, k, grid, κᵏ)))
end

"""
    implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid)

Build a tridiagonal solver for the elliptic equation

```math
(1 + Δt ∂z κ ∂z) cⁿ⁺¹ = c★
```

where `cⁿ⁺¹` and `c★` live at cell `Center`s in the vertical.
"""
function implicit_diffusion_solver(::VerticallyImplicitTimeDiscretization, arch, grid)

    right_hand_side = arch_array(arch, zeros(grid.Nx, grid.Ny, grid.Nz))

    batched_tridiagonal_solver = BatchedTridiagonalSolver(arch, grid;
                                                           lower_diagonal = ivd_lower_diagonal,
                                                                 diagonal = ivd_diagonal,
                                                           upper_diagonal = ivd_upper_diagonal,
                                                          right_hand_side = right_hand_side)

    return VerticallyImplicitDiffusionSolver(batched_tridiagonal_solver)
end

"""
    implicit_step!(field, solver::VerticallyImplicitDiffusionSolver, clock, Δt, κ⁻⁻ᶠ, κ; dependencies)

Initialize the right hand side array `solver.batched_tridiagonal_solver.f`, and then solve the
tridiagonal system for vertically-implicit diffusion, passing the arguments
`clock, Δt, κ⁻⁻ᶠ, κ` into the coefficient functions that return coefficients of the
lower diagonal, diagonal, and upper diagonal of the resulting tridiagonal system.
"""
function implicit_step!(field, solver::VerticallyImplicitDiffusionSolver,
                        clock, Δt, κ⁻⁻ᶠ, κ;
                        dependencies = Event(device(model.architecture)))

    field_interior = interior(field)
    solver.batched_tridiagonal_solver.f .= field_interior

    return solve_batched_tridiagonal_system!(field, solver.batched_tridiagonal_solver,
                                             clock, Δt, κ⁻⁻ᶠ, κ;
                                             dependencies = dependencies)
end
                                             
function implicit_velocity_step!(velocity_field, solver::VerticallyImplicitDiffusionSolver, clock, Δt,
                                 field_location, closure, diffusivities; dependencies = Event(device(model.architecture)))

    vertical_viscous_flux_location = (field_location[1], field_location[2], flip(field_location[3]))
    ν⁻⁻ᶠ = eval(Symbol(:ν, interpolation_code.(vertical_viscous_flux_location)...)) 
    ν = z_viscosity(closure, diffusivities)

    return implicit_step!(velocity_field, solver,
                          clock, Δt, ν⁻⁻ᶠ, ν;
                          dependencies = dependencies)
end

function implicit_tracer_step!(tracer_field, solver::VerticallyImplicitDiffusionSolver, clock, Δt,
                               closure, diffusivities, tracer_index; dependencies = Event(device(model.architecture)))

    κ = z_diffusivity(closure, diffusivities, Val(tracer_index))

    return implicit_step!(tracer_field, solver,
                          clock, Δt, κᶜᶜᶠ, κ;
                          dependencies = dependencies)
end

