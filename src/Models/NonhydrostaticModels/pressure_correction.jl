import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!
using Oceananigans.AbstractOperations: ∂x, ∂y, ∂z

# using Oceananigans.Models.HydrostaticFreeSurfaceModels: step_free_surface!

"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)
    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))

    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, Δt, model.velocities, model.free_surface.η)
    fill_halo_regions!(model.pressures.pNHS)

    if !isnothing(model.free_surface)
        # "First" barotropic pressure correction
        make_pressure_correction!(model, Δt)

        u, v, w = model.velocities
        if haskey(model.auxiliary_fields, :divergence)
            model.auxiliary_fields.divergence .= (∂x(u) + ∂y(v) + ∂z(w))
        end
    end
    
    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure multiplied by the timestep via

    `u^{n+1} = u^n - δₓp_{NH} * Δt / Δx`
"""

@kernel function _make_pressure_correction!(U, grid, Δt, pNHSΔt, η)
    i, j, k = @index(Global, NTuple)

    g = 10.0

    @inbounds U.u[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pNHSΔt)
    @inbounds U.v[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pNHSΔt)
    if k <= grid.Nz
        @inbounds U.w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHSΔt)
    end
    
    # velocity correction at the surface and free surface correction
    if k == grid.Nz
        pNHSΔt[i,j,k+1] = ((1/Δzᶜᶜᶜ(i, j, k, grid) - 1/(2*g*Δt^2))/(1/Δzᶜᶜᶜ(i, j, k, grid) + 1/(2*g*Δt^2))) * pNHSΔt[i,j,k] + (1/(Δt*(1/Δzᶜᶜᶜ(i, j, k, grid) + 1/(2*g*Δt^2)))) * (η[i,j] + Δt*U.w[i,j,k+1])
        U.w[i, j, k+1] -= (pNHSΔt[i,j,k+1] - pNHSΔt[i,j,k])/Δzᶜᶜᶜ(i, j, k, grid)                       
        η[i,j] = (pNHSΔt[i,j,k] + pNHSΔt[i,j,k+1]) / (2g*Δt)
    end
end

"Update the solution variables (velocities and tracers)."
function make_pressure_correction!(model::NonhydrostaticModel, Δt)
    launch!(model.architecture, model.grid, :xyz,
            _make_pressure_correction!,
            model.velocities,
            model.grid,
            Δt,
            model.pressures.pNHS, model.free_surface.η)
    
    ϵ = eps(eltype(model.pressures.pNHS))
    Δt⁺ = max(ϵ, Δt)
    # model.pressures.pNHS ./= Δt⁺

    return nothing
end
