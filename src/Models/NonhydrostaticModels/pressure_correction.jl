import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!

"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))
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
