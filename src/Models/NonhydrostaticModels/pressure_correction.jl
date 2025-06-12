import Oceananigans.TimeSteppers: compute_pressure_correction!, make_pressure_correction!
using Oceananigans.AbstractOperations: Average
using Oceananigans.Fields: Field
using Oceananigans.BoundaryConditions: PerturbationAdvection, PerturbationAdvectionOpenBoundaryCondition

const PAOBC = BoundaryCondition{<:Open{<:PerturbationAdvection}}

"""
correct_boundary_mass_flux!(model::NonhydrostaticModel)

Correct boundary mass fluxes for perturbation advection boundary conditions to ensure
zero net mass flux through each boundary.
"""
function correct_boundary_mass_flux!(model::NonhydrostaticModel)
    velocities = model.velocities
    grid = model.grid

    # Apply mass flux corrections for perturbation advection BCs
    for field in (:u, :v, :w)
        velocity = getproperty(velocities, field)
        bc = getproperty(velocity, :boundary_conditions)

        # West boundary
        if bc.west isa PAOBC
            _correct_west_boundary_mass_flux!(grid, velocity, bc.west)
        end

        # East boundary  
        if bc.east isa PAOBC
            _correct_east_boundary_mass_flux!(grid, velocity, bc.east)
        end

        # South boundary
        if bc.south isa PAOBC
            _correct_south_boundary_mass_flux!(grid, velocity, bc.south, zero(eltype(velocity)))
        end

        # North boundary
        if bc.north isa PAOBC
            _correct_north_boundary_mass_flux!(grid, velocity, bc.north, zero(eltype(velocity)))
        end

        # Bottom boundary
        if bc.bottom isa PAOBC
            _correct_bottom_boundary_mass_flux!(grid, velocity, bc.bottom, zero(eltype(velocity)))
        end

        # Top boundary
        if bc.top isa PAOBC
            _correct_top_boundary_mass_flux!(grid, velocity, bc.top, zero(eltype(velocity)))
        end
    end
    return nothing
end

@inline function _correct_west_boundary_mass_flux!(grid, u, bc)
    i = 1
    current_flux = zero(eltype(u))
    target_flux = bc.classification.matching_scheme.average_mass_flux

    # Calculate current mass flux and total area
    current_flux = Field(Average(view(u, i, :, :)))[]

    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for j in 1:grid.Ny, k in 1:grid.Nz
        @inbounds u[i, j, k] += flux_correction
    end

    return nothing
end

@inline function _correct_east_boundary_mass_flux!(grid, u, bc)
    i = grid.Nx + 1
    current_flux = zero(eltype(u))
    target_flux = bc.classification.matching_scheme.average_mass_flux
    
    # Calculate current mass flux and total area
    current_flux = Field(Average(view(u, i, :, :)))[]
    
    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for j in 1:grid.Ny, k in 1:grid.Nz
        @inbounds u[i, j, k] += flux_correction
    end
    
    return nothing
end

@inline function _correct_south_boundary_mass_flux!(grid, v, bc)
    j = 1
    current_flux = zero(eltype(v))
    target_flux = bc.classification.matching_scheme.average_mass_flux

    # Calculate current mass flux and total area
    current_flux = Field(Average(view(v, :, j, :)))[]

    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for i in 1:grid.Nx, k in 1:grid.Nz
        @inbounds v[i, j, k] += flux_correction
    end

    return nothing
end

@inline function _correct_north_boundary_mass_flux!(grid, v, bc)
    j = grid.Ny + 1
    current_flux = zero(eltype(v))
    target_flux = bc.classification.matching_scheme.average_mass_flux
    
    # Calculate current mass flux and total area
    current_flux = Field(Average(view(v, :, j, :)))[]
    
    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for i in 1:grid.Nx, k in 1:grid.Nz
        @inbounds v[i, j, k] += flux_correction
    end
    
    return nothing
end

@inline function _correct_bottom_boundary_mass_flux!(grid, w, bc)
    k = 1
    current_flux = zero(eltype(w))
    target_flux = bc.classification.matching_scheme.average_mass_flux

    # Calculate current mass flux and total area
    current_flux = Field(Average(view(w, :, :, k)))[]

    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for i in 1:grid.Nx, j in 1:grid.Ny
        @inbounds w[i, j, k] += flux_correction
    end

    return nothing
end

@inline function _correct_top_boundary_mass_flux!(grid, w, bc)
    k = grid.Nz + 1
    current_flux = zero(eltype(w))
    target_flux = bc.classification.matching_scheme.average_mass_flux
    
    # Calculate current mass flux and total area
    current_flux = Field(Average(view(w, :, :, k)))[]
    
    # Calculate and apply correction
    flux_correction = (target_flux - current_flux)
    for i in 1:grid.Nx, j in 1:grid.Ny
        @inbounds w[i, j, k] += flux_correction
    end
    
    return nothing
end




"""
    compute_pressure_correction!(model::NonhydrostaticModel, Δt)

Calculate the (nonhydrostatic) pressure correction associated `tendencies`, `velocities`, and step size `Δt`.
"""
function compute_pressure_correction!(model::NonhydrostaticModel, Δt)

    correct_boundary_mass_flux!(model)

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
