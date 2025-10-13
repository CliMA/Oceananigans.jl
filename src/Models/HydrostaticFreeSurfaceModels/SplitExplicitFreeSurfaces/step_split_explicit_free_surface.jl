using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind
using Oceananigans.Fields: instantiated_location
using ..HydrostaticFreeSurfaceModels: barotropic_U, barotropic_V

# Evolution Kernels
#
# ∂t(η) = -∇⋅U
# ∂t(U) = - gH∇η + f
#
# the free surface field η and its average η̄ are located on `Face`s at the surface (grid.Nz +1). All other intermediate variables
# (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined
@kernel function _split_explicit_barotropic_velocity!(transport_weight, grid, Δτ, η, U, V, Gᵁ, Gⱽ, g, Ũ, Ṽ, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_velocities!(timestepper, i, j, 1, U, V)

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)
    
    # ∂τ(U) = - ∇η + G
    @inbounds begin
        U[i, j, 1] += Δτ * (- g * Hᶠᶜ * ∂xᶠᶜᶠ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
        V[i, j, 1] += Δτ * (- g * Hᶜᶠ * ∂yᶜᶠᶠ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1])
        
        # averaging the transport
        Ũ[i, j, 1] += transport_weight * U[i, j, 1]
        Ṽ[i, j, 1] += transport_weight * V[i, j, 1]
    end
end

@kernel function _split_explicit_free_surface!(averaging_weight, grid, Δτ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_free_surface!(timestepper, i, j, k_top, η)

    δh_U = (δxᶜᵃᵃ(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
            δyᵃᶜᵃ(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, U★, timestepper, V)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid) 

    @inbounds begin
        η[i, j, k_top] += Δτ * (F(i, j, k_top, grid, clock, (; η, U, V)) - δh_U)

        # time-averaging
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     += averaging_weight * U[i, j, 1]
        V̅[i, j, 1]     += averaging_weight * V[i, j, 1]
    end
end

# Change name
const FNS = FixedSubstepNumber
const FTS = FixedTimeStepSize

# since weights can be negative in the first few substeps (as in the default averaging kernel),
# we set a minimum number of substeps to execute to avoid numerical issues
const MINIMUM_SUBSTEPS = 5

@inline calculate_substeps(substepping::FNS, Δt=nothing) = length(substepping.averaging_weights)
@inline calculate_substeps(substepping::FTS, Δt) = max(MINIMUM_SUBSTEPS, ceil(Int, 2 * Δt / substepping.Δt_barotropic))

@inline calculate_adaptive_settings(substepping::FNS, substeps) = substepping.fractional_step_size, substepping.averaging_weights, substepping.transport_weights
@inline calculate_adaptive_settings(substepping::FTS, substeps) = weights_from_substeps(eltype(substepping.Δt_barotropic), substeps, substepping.averaging_kernel)

function iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock, weights, transport_weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.η
    grid        = free_surface.η.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # unpack state quantities, parameters and forcing terms
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
    Ũ, Ṽ    = state.Ũ, state.Ṽ

    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)
    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)

    U_args = (grid, Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, Ũ, Ṽ, timestepper)
    η_args = (grid, Δτᴮ, η, U, V, F, clock, η̅, U̅, V̅, timestepper)

    U_fill_halo_args = (U.data, U.boundary_conditions, U.indices, instantiated_location(U), grid, U.communication_buffers)
    V_fill_halo_args = (V.data, V.boundary_conditions, V.indices, instantiated_location(V), grid, V.communication_buffers)
    η_fill_halo_args = (η.data, η.boundary_conditions, η.indices, instantiated_location(η), grid, η.communication_buffers)

    GC.@preserve η_args U_args U_fill_halo_args V_fill_halo_args η_fill_halo_args begin

        # We need to perform ~50 time-steps which means
        # launching ~100 very small kernels: we are limited by
        # latency of argument conversion to GPU-compatible values.
        # To alleviate this penalty we convert first and then we substep!
        converted_η_args = convert_to_device(arch, η_args)
        converted_U_args = convert_to_device(arch, U_args)

        converted_U_fill_halo_args = convert_to_device(arch, U_fill_halo_args)
        converted_V_fill_halo_args = convert_to_device(arch, V_fill_halo_args)
        converted_η_fill_halo_args = convert_to_device(arch, η_fill_halo_args)

        for substep in 1:Nsubsteps
            @inbounds averaging_weight = weights[substep]
            @inbounds transport_weight = transport_weights[substep]
            
            # Advance barotropic velocities
            barotropic_velocity_kernel!(transport_weight, converted_U_args...)
            fill_halo_regions!(converted_U_fill_halo_args...; only_local_halos=true)
            fill_halo_regions!(converted_V_fill_halo_args...; only_local_halos=true)

            # Advance free surface
            free_surface_kernel!(averaging_weight, converted_η_args...)
            fill_halo_regions!(converted_η_fill_halo_args...; only_local_halos=true)
        end
    end

    return nothing
end

@kernel function _update_split_explicit_state!(η, U, V, grid, state)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        η[i, j, k_top] = state.η̅[i, j, k_top]
        U[i, j, 1]     = state.U̅[i, j, 1]
        V[i, j, 1]     = state.V̅[i, j, 1]
    end
end

#####
##### SplitExplicitFreeSurface barotropic subcycling
#####

function step_free_surface!(free_surface::SplitExplicitFreeSurface, model, baroclinic_timestepper, Δt)

    # Note: free_surface.η.grid != model.grid for DistributedSplitExplicitFreeSurface
    # since halo_size(free_surface.η.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.η.grid
    filtered_state    = free_surface.filtered_state
    substepping       = free_surface.substepping

    barotropic_velocities = free_surface.barotropic_velocities

    barotropic_timestepper = free_surface.timestepper
    baroclinic_timestepper = model.timestepper

    # Calculate the substepping parameters
    # barotropic time step as fraction of baroclinic step and averaging weights
    Nsubsteps = calculate_substeps(substepping, Δt)
    fractional_Δt, weights, transport_weights = calculate_adaptive_settings(substepping, Nsubsteps)
    Nsubsteps = length(weights)

    # barotropic time step in seconds
    Δτᴮ = fractional_Δt * Δt

    # Slow forcing terms
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    #free surface state
    η = free_surface.η
    U = barotropic_velocities.U
    V = barotropic_velocities.V
    F = model.forcing.η

    # Wait for setup step to finish
    wait_free_surface_communication!(free_surface, model, architecture(free_surface_grid))
    
    @apply_regionally begin
        # Reset the filtered fields and the barotropic timestepper to zero. 
        initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper)
        
        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, F, model.clock, weights, transport_weights, Val(Nsubsteps))

        # Update eta and velocities for the next timestep
        # The halos are updated in the `update_state!` function
        launch!(architecture(free_surface_grid), free_surface_grid, :xy,
                _update_split_explicit_state!, η, U, V, free_surface_grid, filtered_state)
    end

    # Fill all the barotropic state
    fill_halo_regions!((filtered_state.Ũ, filtered_state.Ṽ); async=true)
    fill_halo_regions!((U, V, η); async=true)

    return nothing
end
