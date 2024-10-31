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

using Printf
using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

# constants for AB3 time stepping scheme (from https://doi.org/10.1016/j.ocemod.2004.08.002)
const β = 0.281105
const α = 1.5 + β
const θ = - 0.5 - 2β
const γ = 0.088
const δ = 0.614
const ϵ = 0.013
const μ = 1 - δ - γ - ϵ

# Evolution Kernels
#
# ∂t(η) = -∇⋅U
# ∂t(U) = - gH∇η + f
#
# the free surface field η and its average η̄ are located on `Face`s at the surface (grid.Nz +1). All other intermediate variables
# (U, V, Ū, V̄) are barotropic fields (`ReducedField`) for which a k index is not defined

# Special ``partial'' divergence for free surface evolution
@inline div_Txᶜᶜᶠ(i, j, k, grid, U★::Function, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δxTᶜᵃᵃ(i, j, k, grid, Δy_qᶠᶜᶠ, U★, args...)
@inline div_Tyᶜᶜᶠ(i, j, k, grid, V★::Function, args...) =  1 / Azᶜᶜᶠ(i, j, k, grid) * δyTᵃᶜᵃ(i, j, k, grid, Δx_qᶜᶠᶠ, V★, args...)

# The functions `η★` `U★` and `V★` represent the value of free surface, barotropic zonal and meridional velocity at time step m+1/2

# Time stepping extrapolation U★, and η★

# AB3 step
@inline function U★(i, j, k, grid, ::AdamsBashforth3Scheme, Uᵐ, Uᵐ⁻¹, Uᵐ⁻²)
    FT = eltype(grid)
    return @inbounds FT(α) * Uᵐ[i, j, k] + FT(θ) * Uᵐ⁻¹[i, j, k] + FT(β) * Uᵐ⁻²[i, j, k]
end

@inline function η★(i, j, k, grid, ::AdamsBashforth3Scheme, ηᵐ⁺¹, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    FT = eltype(grid)
    return @inbounds FT(δ) * ηᵐ⁺¹[i, j, k] + FT(μ) * ηᵐ[i, j, k] + FT(γ) * ηᵐ⁻¹[i, j, k] + FT(ϵ) * ηᵐ⁻²[i, j, k]
end

# Forward Backward Step
@inline U★(i, j, k, grid, ::ForwardBackwardScheme, U, args...) = @inbounds U[i, j, k]
@inline η★(i, j, k, grid, ::ForwardBackwardScheme, η, args...) = @inbounds η[i, j, k]

@inline advance_previous_velocity!(i, j, k, ::ForwardBackwardScheme, U, Uᵐ⁻¹, Uᵐ⁻²) = nothing

@inline function advance_previous_velocity!(i, j, k, ::AdamsBashforth3Scheme, U, Uᵐ⁻¹, Uᵐ⁻²)
    @inbounds Uᵐ⁻²[i, j, k] = Uᵐ⁻¹[i, j, k]
    @inbounds Uᵐ⁻¹[i, j, k] =    U[i, j, k]

    return nothing
end

@inline advance_previous_free_surface!(i, j, k, ::ForwardBackwardScheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) = nothing

@inline function advance_previous_free_surface!(i, j, k, ::AdamsBashforth3Scheme, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)
    @inbounds ηᵐ⁻²[i, j, k] = ηᵐ⁻¹[i, j, k]
    @inbounds ηᵐ⁻¹[i, j, k] =   ηᵐ[i, j, k]
    @inbounds   ηᵐ[i, j, k] =    η[i, j, k]

    return nothing
end

@kernel function _split_explicit_free_surface!(grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
    i, j = @index(Global, NTuple)
    free_surface_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
end


@inline function free_surface_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², timestepper)
    k_top = grid.Nz+1

    @inbounds begin
        advance_previous_free_surface!(i, j, k_top, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²)

        η[i, j, k_top] -= Δτ * (div_Txᶜᶜᶠ(i, j, k_top-1, grid, U★, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²) +
                                div_Tyᶜᶜᶠ(i, j, k_top-1, grid, U★, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²))
    end

    return nothing
end

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²,
                                                      U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                                                      η̅, U̅, V̅, Gᵁ, Gⱽ, g, 
                                                      timestepper)
    i, j = @index(Global, NTuple)
    velocity_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²,
                        U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                        η̅, U̅, V̅, averaging_weight,
                        Gᵁ, Gⱽ, g, 
                        timestepper)
end

@inline function velocity_evolution!(i, j, grid, Δτ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²,
                                     U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
                                     η̅, U̅, V̅, averaging_weight,
                                     Gᵁ, Gⱽ, g, 
                                     timestepper)
    k_top = grid.Nz+1

    @inbounds begin
        advance_previous_velocity!(i, j, k_top-1, timestepper, U, Uᵐ⁻¹, Uᵐ⁻²)
        advance_previous_velocity!(i, j, k_top-1, timestepper, V, Vᵐ⁻¹, Vᵐ⁻²)

        Hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
        Hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
        
        # ∂τ(U) = - ∇η + G
        U[i, j, k_top-1] +=  Δτ * (- g * Hᶠᶜ * ∂xTᶠᶜᶠ(i, j, k_top, grid, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gᵁ[i, j, k_top-1])
        V[i, j, k_top-1] +=  Δτ * (- g * Hᶜᶠ * ∂yTᶜᶠᶠ(i, j, k_top, grid, η★, timestepper, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻²) + Gⱽ[i, j, k_top-1])
                          
        # time-averaging
        η̅[i, j, k_top]   += averaging_weight * η[i, j, k_top]
        U̅[i, j, k_top-1] += averaging_weight * U[i, j, k_top-1]
        V̅[i, j, k_top-1] += averaging_weight * V[i, j, k_top-1]
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

const FixedSubstepsSetting{N} = SplitExplicitSettings{<:FixedSubstepNumber{<:Any, <:NTuple{N, <:Any}}} where N
const FixedSubstepsSplitExplicit{F} = SplitExplicitFreeSurface{<:Any, <:Any, <:Any, <:Any, <:FixedSubstepsSetting{N}} where N

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
    η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅

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
    set!(free_surface.η, filtered_state.η)
    set!(velocities.U,   filtered_state.U) 
    set!(velocities.U,   filtered_state.V)
    
    # fields_to_fill = (velocities.U, velocities.V) TODO: do this?
    # fill_halo_regions!(fields_to_fill; async = true)

    # Preparing velocities for the barotropic correction
    @apply_regionally begin
        mask_immersed_field!(model.velocities.u)
        mask_immersed_field!(model.velocities.v)
    end

    return nothing
end
