using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SSPRK3TimeStepper

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, ::Nothing, u, v)
    i, j  = @index(Global, NTuple)	
    k_top = grid.Nz+1

    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, active_cells_map, u, v)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    k_top = grid.Nz+1

    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

@inline function compute_barotropic_mode!(U, V, grid, u, v) 
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U, V, grid, active_cells_map, u, v; active_cells_map)

    return nothing
end

function initialize_free_surface_state!(state, η, timestepper)

    parent(state.U) .= parent(state.U̅)
    parent(state.V) .= parent(state.V̅)

    initialize_auxiliary_state!(state, η, timestepper)

    fill!(state.η̅, 0)
    fill!(state.U̅, 0)
    fill!(state.V̅, 0)

    return nothing
end

initialize_auxiliary_state!(state, η, ::ForwardBackwardScheme) = nothing

function initialize_auxiliary_state!(state, η, timestepper)
    parent(state.Uᵐ⁻¹) .= parent(state.U̅)
    parent(state.Vᵐ⁻¹) .= parent(state.V̅)

    parent(state.Uᵐ⁻²) .= parent(state.U̅)
    parent(state.Vᵐ⁻²) .= parent(state.V̅)

    parent(state.ηᵐ)   .= parent(η)
    parent(state.ηᵐ⁻¹) .= parent(η)
    parent(state.ηᵐ⁻²) .= parent(η)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ, grid)
    i, j, k = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        u[i, j, k] = u[i, j, k] + (U̅[i, j, k_top-1] - U[i, j, k_top-1]) / Hᶠᶜ[i, j, 1]
        v[i, j, k] = v[i, j, k] + (V̅[i, j, k_top-1] - V[i, j, k_top-1]) / Hᶜᶠ[i, j, 1]
    end
end

function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    sefs       = free_surface.state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    Hᶠᶜ, Hᶜᶠ   = free_surface.auxiliary.Hᶠᶜ, free_surface.auxiliary.Hᶜᶠ
    arch       = architecture(grid)


    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    compute_barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U̅, V̅, U, V, Hᶠᶜ, Hᶜᶠ, grid)

    return nothing
end

"""
Explicitly step forward η in substeps.
"""
step_free_surface!(free_surface::SplitExplicitFreeSurface, model, timestepper, Δt) =
    split_explicit_free_surface_step!(free_surface, model, Δt, χ)

function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    @apply_regionally compute_barotropic_mode!(sefs.state.U̅, sefs.state.V̅, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.U̅, sefs.state.V̅, sefs.η))
end

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt, χ)

    # Note: free_surface.η.grid != model.grid for DistributedSplitExplicitFreeSurface
    # since halo_size(free_surface.η.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.η.grid

    # Wait for previous set up
    wait_free_surface_communication!(free_surface, architecture(free_surface_grid))

    # Calculate the substepping parameterers
    settings = free_surface.settings 
    Nsubsteps = calculate_substeps(settings.substepping, Δt)
    
    # barotropic time step as fraction of baroclinic step and averaging weights
    fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) 
    Nsubsteps = length(weights)

    # barotropic time step in seconds
    Δτᴮ = fractional_Δt * Δt  
    
    # reset free surface averages
    @apply_regionally begin 
        initialize_free_surface_state!(free_surface.state, free_surface.η, settings.timestepper)
        
        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, free_surface_grid, Δτᴮ, weights, Val(Nsubsteps))
        
        # Reset eta for the next timestep
        set!(free_surface.η, free_surface.state.η̅)
    end

    fields_to_fill = (free_surface.state.U̅, free_surface.state.V̅)
    fill_halo_regions!(fields_to_fill; async = true)

    # Preparing velocities for the barotropic correction
    @apply_regionally begin 
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    return nothing
end