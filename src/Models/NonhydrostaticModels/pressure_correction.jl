"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)

    # Mask immersed velocities
    foreach(mask_immersed_field!, model.velocities)
    fill_halo_regions!(model.velocities, model.clock, fields(model))
    enforce_open_boundary_mass_conservation!(model, model.boundary_mass_fluxes)

    p_Δt = model.pressures.pNHS
    solve_for_pressure!(p_Δt, model.pressure_solver, model.free_surface, model.velocities, Δt)

    set_top_pressure_boundary_condition!(p_Δt, model.free_surface, model.velocities.w, Δt)
    fill_halo_regions!(p_Δt)

    return nothing
end

# TODO: make these fallbacks
# Routines for rigid lid
set_top_pressure_boundary_condition!(p_Δt, ::Nothing, w̃, Δt) = nothing

function set_top_pressure_boundary_condition!(p_Δt, free_surface, w̃, Δt)
    top_bc = p_Δt.boundary_conditions.top
    g = free_surface.gravitational_acceleration

    # Set the "coefficient" of the MixedBoundaryCondition
    top_bc.condition.coefficient[] = 1 / (g * Δt^2)

    # Set the "combination" of the MixedBoundaryCondition
    η = free_surface.η
    bc_rhs = top_bc.condition.inhomogeneity
    grid = p_Δt.grid
    arch = grid.architecture
    launch!(arch, grid, :xy, _set_top_pressure_boundary_condition!, bc_rhs, grid, w̃, Δt, η, g, p_Δt)
    return nothing
end

@kernel function _set_top_pressure_boundary_condition!(bc_rhs, grid, w̃, Δt, η, g, p_Δt)
    i, j = @index(Global, NTuple)
    Nz = grid.Nz

    @inbounds bc_rhs[i, j, 1] = η[i, j, Nz+1] / Δt + w̃[i, j, Nz+1]

    # Compute ηⁿ⁺¹
    Δzᶠ = Δzᵃᵃᶠ(i, j, Nz+1, grid)
    a = 1 / Δzᶠ - 1 / (2g * Δt^2)
    b = 1 / Δzᶠ + 1 / (2g * Δt^2)
    @inbounds begin
        η★ = η[i, j, Nz+1] + Δt * w̃[i, j, Nz+1]
        η[i, j, Nz+1] = (p_Δt[i, j, Nz] / Δt + (p_Δt[i, j, Nz] * a / (b * Δt) + η★ / (b * Δt^2))) / 2g
    end
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

@kernel function _compute_surface_vertical_velocity!(w, grid, pNHSΔt)
    i, j = @index(Global, NTuple)
    k = grid.Nz + 1
    @inbounds w[i, j, k] -= ∂zᶜᶜᶠ(i, j, k, grid, pNHSΔt)
end

"Update the solution variables (velocities and tracers)."
function make_pressure_correction!(model::NonhydrostaticModel, Δt)

    grid = model.grid
    arch = grid.architecture

    launch!(arch, grid, :xyz,
            _make_pressure_correction!,
            model.velocities,
            model.grid,
            model.pressures.pNHS)

    if !isnothing(model.free_surface)
        launch!(arch, grid, :xy,
                _compute_surface_vertical_velocity!,
                model.velocities.w,
                model.grid,
                model.pressures.pNHS)
    end

    ϵ = eps(eltype(model.pressures.pNHS))
    Δt⁺ = max(ϵ, Δt)
    model.pressures.pNHS ./= Δt⁺

    return nothing
end
