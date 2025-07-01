import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!
using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: Field
using Oceananigans.BoundaryConditions: PerturbationAdvection, FlatExtrapolation
using Oceananigans.ImmersedBoundaries: immersed_inactive_node

const MatchingScheme = Union{FlatExtrapolation, PerturbationAdvection}
const OBC  = BoundaryCondition{<:Open} # OpenBoundaryCondition (with no matching scheme)
const FOBC = BoundaryCondition{<:Open{<:Nothing}} # "Fixed-velocity" OpenBoundaryCondition (with no matching scheme)
const ROBC = BoundaryCondition{<:Open{<:MatchingScheme}} # Radiation OpenBoundaryCondition

# Left boundary averages for normal velocity components
west_average(u)   = Field(Average(view(u, 1, :, :), dims=(2, 3)))[]
south_average(v)  = Field(Average(view(v, :, 1, :), dims=(1, 3)))[]
bottom_average(w) = Field(Average(view(w, :, :, 1), dims=(1, 2)))[]

# Right boundary averages for normal velocity components
east_average(u)   = Field(Average(view(u, u.grid.Nx + 1, :, :), dims=(2, 3)))[]
north_average(v)  = Field(Average(view(v, :, v.grid.Ny + 1, :), dims=(1, 3)))[]
top_average(w)    = Field(Average(view(w, :, :, w.grid.Nz + 1), dims=(1, 2)))[]

function gather_boundary_fluxes(model::NonhydrostaticModel)

    velocities = model.velocities
    grid = model.grid

    # Get the boundary conditions for the velocities
    u_bcs = velocities.u.boundary_conditions
    v_bcs = velocities.v.boundary_conditions
    w_bcs = velocities.w.boundary_conditions

    # Collect left and right ROBC boundary conditions into separate lists
    left_ROBCs = Symbol[]
    right_ROBCs = Symbol[]

    # Initialize fluxes to zero
    left_flux = zero(grid)
    right_flux = zero(grid)

    # Calculate flux through left boundaries
    if u_bcs.west isa OBC
        left_flux += west_average(velocities.u)
        u_bcs.west isa ROBC && push!(left_ROBCs, :west)
    end
    if v_bcs.south isa OBC
        left_flux += south_average(velocities.v)
        v_bcs.south isa ROBC && push!(left_ROBCs, :south)
    end
    if w_bcs.bottom isa OBC
        left_flux += bottom_average(velocities.w)
        w_bcs.bottom isa ROBC && push!(left_ROBCs, :bottom)
    end

    # Calculate flux through right boundaries
    if u_bcs.east isa OBC
        right_flux += east_average(velocities.u)
        u_bcs.east isa ROBC && push!(right_ROBCs, :east)
    end
    if v_bcs.north isa OBC
        right_flux += north_average(velocities.v)
        v_bcs.north isa ROBC && push!(right_ROBCs, :north)
    end
    if w_bcs.top isa OBC
        right_flux += top_average(velocities.w)
        w_bcs.top isa ROBC && push!(right_ROBCs, :top)
    end

    # Calculate total flux (positive means net inflow)
    total_flux = left_flux - right_flux

    return total_flux, left_ROBCs, right_ROBCs
end


"""
correct_boundary_mass_flux!(model::NonhydrostaticModel)

Correct boundary mass fluxes for perturbation advection boundary conditions to ensure
zero net mass flux through each boundary.
"""
function correct_boundary_mass_flux!(model::NonhydrostaticModel)
    velocities = model.velocities
    grid = model.grid

    total_flux, left_ROBCs, right_ROBCs = gather_boundary_fluxes(model)

    # Calculate flux correction per boundary
    extra_flux_per_boundary = total_flux / (length(left_ROBCs) + length(right_ROBCs))

    # Subtract extra flux from left boundaries to reduce inflow
    for bc in left_ROBCs
        if bc == :west
            velocities.u[1, :, :] = velocities.u[1, :, :] .- extra_flux_per_boundary
        elseif bc == :south  
            velocities.v[:, 1, :] = velocities.v[:, 1, :] .- extra_flux_per_boundary
        elseif bc == :bottom
            velocities.w[:, :, 1] = velocities.w[:, :, 1] .- extra_flux_per_boundary
        end
    end

    # Add extra flux to right boundaries to increase outflow
    for bc in right_ROBCs
        if bc == :east
            velocities.u[grid.Nx + 1, :, :] = velocities.u[grid.Nx + 1, :, :] .+ extra_flux_per_boundary
        elseif bc == :north
            velocities.v[:, grid.Ny + 1, :] = velocities.v[:, grid.Ny + 1, :] .+ extra_flux_per_boundary
        elseif bc == :top
            velocities.w[:, :, grid.Nz + 1] = velocities.w[:, :, grid.Nz + 1] .+ extra_flux_per_boundary
        end
    end
end



"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    #correct_boundary_mass_flux!(model)

    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, Δt, model.velocities)
    fill_halo_regions!(model.pressures.pNHS)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure multiplied by the timestep via

    `u^{n+1} = u^n - δₓp_{NH} * Δt / Δx`
"""
@kernel function _make_pressure_correction!(U, grid, pNHSΔt)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pNHSΔt)
    @inbounds U.v[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pNHSΔt)
    @inbounds U.w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHSΔt)
end

"Update the solution variables (velocities and tracers)."
function make_pressure_correction!(model::NonhydrostaticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _make_pressure_correction!,
            model.velocities,
            model.grid,
            model.pressures.pNHS)
    
    ϵ = eps(eltype(model.pressures.pNHS))
    Δt⁺ = max(ϵ, Δt)
    model.pressures.pNHS ./= Δt⁺

    return nothing
end
