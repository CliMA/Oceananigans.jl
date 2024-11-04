using Oceananigans.Grids
using Oceananigans.Grids: topology
using Oceananigans.Utils
using Oceananigans.AbstractOperations: Δz
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using Oceananigans.Architectures: convert_args
using Oceananigans.ImmersedBoundaries: peripheral_node, immersed_inactive_node, GFBIBG
using Oceananigans.ImmersedBoundaries: inactive_node, IBG, c, f
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, retrieve_surface_active_cells_map, retrieve_interior_active_cells_map
using Oceananigans.ImmersedBoundaries: active_linear_index_to_tuple, ActiveCellsIBG, ActiveZColumnsIBG
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.DistributedComputations: Distributed
using Oceananigans: fields

using Printf
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

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
    
    advance_previous_free_surface!(timestepper, i, j, k_top, η)
    @inbounds  η[i, j, k_top] -= Δτ * (δxTᶜᵃᵃ(i, j, k_top, grid, Δy_qᶠᶜᶠ, U★, timestepper, U) +
                                       δyTᵃᶜᵃ(i, j, k_top, grid, Δx_qᶜᶠᶠ, U★, timestepper, V)) / Azᶜᶜᶠ(i, j, k_top, grid)
end

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ, 
                                                      η, U, V, 
                                                      η̅, U̅, V̅, 
                                                      Gᵁ, Gⱽ, g, 
                                                      timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    advance_previous_velocities!(timestepper, i, j, 1, U)
    advance_previous_velocities!(timestepper, i, j, 1, V)

    Hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    
    @inbounds begin
        # ∂τ(U) = - ∇η + G
        U[i, j, 1] +=  Δτ * (- g * Hᶠᶜ * ∂xTᶠᶜᶠ(i, j, k_top, grid, η★, timestepper, η) + Gᵁ[i, j, 1])
        V[i, j, 1] +=  Δτ * (- g * Hᶜᶠ * ∂yTᶜᶠᶠ(i, j, k_top, grid, η★, timestepper, η) + Gⱽ[i, j, 1])
                          
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

    free_surface_kernel!, _        = configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)
    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)

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
        converted_η_args = convert_args(arch, η_args)
        converted_U_args = convert_args(arch, U_args)

        @unroll for substep in 1:Nsubsteps
            Base.@_inline_meta
            averaging_weight = weights[substep]
            free_surface_kernel!(converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, converted_U_args...)
        end
    end

    return nothing
end

#####
##### SplitExplicitFreeSurface barotropic subcylicing
#####

ab2_step_free_surface!(free_surface::SplitExplicitFreeSurface, model, Δt, χ) =
    split_explicit_free_surface_step!(free_surface, model, Δt)

function split_explicit_free_surface_step!(free_surface::SplitExplicitFreeSurface, model, Δt)

    # Note: free_surface.η.grid != model.grid for DistributedSplitExplicitFreeSurface
    # since halo_size(free_surface.η.grid) != halo_size(model.grid)
    free_surface_grid = free_surface.η.grid
    filtered_state    = free_surface.filtered_state
    substepping       = free_surface.substepping
    timestepper       = free_surface.timestepper
    velocities        = free_surface.barotropic_velocities

    # Wait for previous set up
    wait_free_surface_communication!(free_surface, model, architecture(free_surface_grid))

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

    # reset free surface averages
    @apply_regionally begin
        initialize_free_surface_state!(filtered_state, free_surface.η, velocities, timestepper)

        # Solve for the free surface at tⁿ⁺¹
        iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, weights, Val(Nsubsteps))
    end

    # Reset eta and velocities for the next timestep
    # The halos are updated in the `update_state!` function
    parent(free_surface.η) .= parent(filtered_state.η)
    parent(velocities.U)   .= parent(filtered_state.U) 
    parent(velocities.V)   .= parent(filtered_state.V)
    
    # Preparing velocities for the barotropic correction
    @apply_regionally begin
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    return nothing
end
