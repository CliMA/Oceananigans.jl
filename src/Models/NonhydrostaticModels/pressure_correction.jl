import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

const c = Center()
const f = Face()

using Oceananigans.Grids: node
using Oceananigans.StokesDrifts: StokesDrift, parameters_tuple

function modulate_by_stokes_drift!(model, sgn, t=time(model))
    stokes_drift = model.stokes_drift
    if stokes_drift isa StokesDrift
        grid = model.grid
        arch = architecture(grid)
        u, v, w = model.velocities
        launch!(arch, grid, :xyz, _modulate_by_stokes_drift, u, v, w, sgn, grid, t, stokes_drift)
    end
    return nothing
end

@kernel function _modulate_by_stokes_drift(u, v, w, sgn, grid, time, sd)
    i, j, k = @index(Global, NTuple)

    pt = parameters_tuple(sd)
    Xu = node(i, j, k, grid, f, c, c)
    Xv = node(i, j, k, grid, c, f, c)
    Xw = node(i, j, k, grid, c, c, f)

    @inbounds begin
        u[i, j, k] = u[i, j, k] + sgn * sd.uˢ(Xu..., time, pt...)
        v[i, j, k] = v[i, j, k] + sgn * sd.vˢ(Xv..., time, pt...)
        w[i, j, k] = w[i, j, k] + sgn * sd.wˢ(Xw..., time, pt...)
    end
end

subtract_stokes_drift!(model, t=time(model)) = modulate_by_stokes_drift!(model, -1, t)
add_stokes_drift!(model, t=time(model)) = modulate_by_stokes_drift!(model, +1, t)

"""
    calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function calculate_pressure_correction!(model::NonhydrostaticModel, Δt)

    subtract_stokes_drift!(model)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))
    solve_for_pressure!(model.pressures.pNHS, model.pressure_solver, Δt, model.velocities)
    fill_halo_regions!(model.pressures.pNHS)

    add_stokes_drift!(model, time(model) + Δt)

    return nothing
end

#####
##### Fractional and time stepping
#####

"""
Update the predictor velocities u, v, and w with the non-hydrostatic pressure via

    `u^{n+1} = u^n - δₓp_{NH} / Δx * Δt`
"""
@kernel function _pressure_correct_velocities!(U, grid, Δt, pNHS)
    i, j, k = @index(Global, NTuple)

    @inbounds U.u[i, j, k] -= ∂xᶠᶜᶜ(i, j, k, grid, pNHS) * Δt
    @inbounds U.v[i, j, k] -= ∂yᶜᶠᶜ(i, j, k, grid, pNHS) * Δt
    @inbounds U.w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHS) * Δt
end

"Update the solution variables (velocities and tracers)."
function pressure_correct_velocities!(model::NonhydrostaticModel, Δt)

    launch!(model.architecture, model.grid, :xyz,
            _pressure_correct_velocities!,
            model.velocities,
            model.grid,
            Δt,
            model.pressures.pNHS)

    return nothing
end
