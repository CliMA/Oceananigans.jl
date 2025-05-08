import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: step_free_surface!

"""
    calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(model::NonhydrostaticModel, Δt)
    #saves current w before correction?
    if hasfield(typeof(model.auxiliary_fields), :w_star)
        copyto!(model.auxiliary_fields.w_star.data, model.velocities.w.data)
    end

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, Δt, model.velocities, model.free_surface.η)
    fill_halo_regions!(model.pressures.pNHS)

    if !isnothing(model.free_surface)
        # "First" barotropic pressure correction
        pressure_correct_velocities!(model, Δt)
    end
    
    #fill_halo_regions!(model.free_surface.η, model.clock, fields(model))

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""

# puts in free surface correction after velocity correction
@kernel function _pressure_correct_velocities!(U, grid, Δt, pNHS, η)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pNHS) * Δt
    @inbounds U.v[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pNHS) * Δt
    if k <= grid.Nz
        @inbounds U.w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHS) * Δt
    end

    g = 10.0
    
    if k == grid.Nz
        pNHS[i,j,k+1] = ((1/Δzᶜᶜᶜ(i, j, k, grid) - 1/(2*g*Δt^2))/(1/Δzᶜᶜᶜ(i, j, k, grid) + 1/(2*g*Δt^2))) * pNHS[i,j,k] + (1/(Δt^2*(1/Δzᶜᶜᶜ(i, j, k, grid) + 1/(2*g*Δt^2)))) * (η[i,j] + Δt*U.w[i,j,k+1])
        U.w[i, j, k+1] -= (pNHS[i,j,k+1] - pNHS[i,j,k]) * Δt/Δzᶜᶜᶜ(i, j, k, grid) # TODO: use the right dz
        η[i,j] = (pNHS[i,j,k] + pNHS[i,j,k+1]) / (2g)

    end
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(model::NonhydrostaticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_velocities!,
            model.velocities,
            model.grid,
            Δt,
            model.pressures.pNHS, model.free_surface.η)  

    return nothing
end
