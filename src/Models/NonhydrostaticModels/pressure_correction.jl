import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))
    enforce_open_boundary_mass_conservation!(model, model.boundary_mass_fluxes)
    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, model.free_surface, model.velocities, Δt)
    set_top_pressure_boundary_condition!(model.pressures.pNHS, model.free_surface, model.velocities.w, Δt)
    fill_halo_regions!(model.pressures.pNHS)

    return nothing
end

set_top_pressure_boundary_condition!(p, ::Nothing, w̃, Δt) = nothing

function set_top_pressure_boundary_condition!(p, free_surface, w̃, Δt)
    top_bc = p.boundary_conditions.top
    g = free_surface.gravitational_acceleration

    # Set the "coefficient" of the MixedBoundaryCondition
    top_bc.condition.coefficient[] = 1 / (g * Δt^2)

    # Set the "combination" of the MixedBoundaryCondition
    η = free_surface.η
    combo = top_bc.condition.combination
    grid = p.grid
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_top_pressure_boundary_condition!, combo, grid, w̃, Δt, η)
    return nothing
end

@kernel function _set_top_pressure_boundary_condition!(combo, grid, w̃, Δt, η)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz
    @inbounds combo[i, j, 1] = η[i, j, Nz+1] / Δt^2 * w̃[i, j, Nz+1] / Δt
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
