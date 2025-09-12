using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

# Evolution Kernels
#
# ∂t(η) = -∇⋅U
# ∂t(U) = - gH∇η + f
#
# the free surface field η and its average η̄ are located on `Face`s at the surface (grid.Nz +1). All other intermediate variables
# (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined

@kernel function _split_explicit_free_surface!(grid, Δτ, η, U, V, timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_free_surface!(timestepper, i, j, k_top, η)
    @inbounds  η[i, j, k_top] -= Δτ * (δxTᶜᵃᵃ(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
                                       δyTᵃᶜᵃ(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, U★, timestepper, V)) * Az⁻¹ᶜᶜᶠ(i, j, k_top, grid)
end

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ,
                                                      η, U, V,
                                                      η̅, U̅, V̅,
                                                      Gᵁ, Gⱽ, g,
                                                      timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    cache_previous_velocities!(timestepper, i, j, 1, U)
    cache_previous_velocities!(timestepper, i, j, 1, V)

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)

    @inbounds begin
        # ∂τ(U) = - ∇η + G
        Uᵐ⁺¹ = U[i, j, 1] + Δτ * (- g * Hᶠᶜ * ∂xTᶠᶜᶠ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
        Vᵐ⁺¹ = V[i, j, 1] + Δτ * (- g * Hᶜᶠ * ∂yTᶜᶠᶠ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1])

        # time-averaging
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     += averaging_weight * Uᵐ⁺¹
        V̅[i, j, 1]     += averaging_weight * Vᵐ⁺¹

        # Updating the velocities
        U[i, j, 1] = Uᵐ⁺¹
        V[i, j, 1] = Vᵐ⁺¹
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

@inline calculate_adaptive_settings(substepping::FNS, substeps) = substepping.fractional_step_size, substepping.averaging_weights
@inline calculate_adaptive_settings(substepping::FTS, substeps) = weights_from_substeps(eltype(substepping.Δt_barotropic),
                                                                                        substeps, substepping.averaging_kernel)

function iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.η
    grid        = free_surface.η.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # unpack state quantities, parameters and forcing terms
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η, state.U, state.V

    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!, nothing, nothing)
    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!, nothing, nothing)

    η_args = (grid, Δτᴮ, η, U, V,
              timestepper)

    U_args = (grid, Δτᴮ, η, U, V,
              η̅, U̅, V̅, GUⁿ, GVⁿ, g,
              timestepper)

    GC.@preserve η_args U_args begin

        # We need to perform ~50 time-steps which means
        # launching ~100 very small kernels: we are limited by
        # latency of argument conversion to GPU-compatible values.
        # To alleviate this penalty we convert first and then we substep!
        converted_η_args = convert_to_device(arch, η_args)
        converted_U_args = convert_to_device(arch, U_args)

        @unroll for substep in 1:Nsubsteps
            Base.@_inline_meta
            averaging_weight = weights[substep]
            free_surface_kernel!(converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, converted_U_args...)
        end
    end

    return nothing
end

@kernel function _update_split_explicit_state!(η, U, V, grid, η̅, U̅, V̅)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        η[i, j, k_top] = η̅[i, j, k_top]
        U[i, j, 1]     = U̅[i, j, 1]
        V[i, j, 1]     = V̅[i, j, 1]
    end
end

#####
##### SplitExplicitFreeSurface barotropic subcylicing
#####

function step_free_surface!(free_surface::SplitExplicitFreeSurface, model, baroclinic_timestepper, Δt)

    # Note: free_surface.η.grid != model.grid for DistributedSplitExplicitFreeSurface
    # since halo_size(free_surface.η.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.η.grid
    filtered_state    = free_surface.filtered_state
    substepping       = free_surface.substepping

    barotropic_velocities = free_surface.barotropic_velocities

    # Wait for setup step to finish
    wait_free_surface_communication!(free_surface, model, architecture(free_surface_grid))

    barotropic_timestepper = free_surface.timestepper
    baroclinic_timestepper = model.timestepper

    stage = model.clock.stage

    # Reset the filtered fields and the barotropic timestepper to zero. 
    # In case of an RK3 timestepper, reset also the free surface state for the last stage.
    @apply_regionally initialize_free_surface_state!(free_surface, baroclinic_timestepper, barotropic_timestepper)

    # Calculate the substepping parameterers
    # barotropic time step as fraction of baroclinic step and averaging weights
    Nsubsteps = calculate_substeps(substepping, Δt)
    fractional_Δt, weights = calculate_adaptive_settings(substepping, Nsubsteps)
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
    η̅ = filtered_state.η
    U̅ = filtered_state.U
    V̅ = filtered_state.V

    # reset free surface averages
    @apply_regionally begin
        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, weights, Val(Nsubsteps))

        # Update eta and velocities for the next timestep
        # The halos are updated in the `update_state!` function
        launch!(architecture(free_surface_grid), free_surface_grid, :xy,
                _update_split_explicit_state!, η, U, V, free_surface_grid, η̅, U̅, V̅)

        # Preparing velocities for the barotropic correction
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    return nothing
end
