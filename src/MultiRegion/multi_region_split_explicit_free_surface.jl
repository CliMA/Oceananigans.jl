using Oceananigans.Utils
using Oceananigans.Utils: configured_kernel
using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface,
                                                        SplitExplicitSettings,
                                                        SplitExplicitState,
                                                        FixedSubstepNumber, FixedTimeStepSize,
                                                        calculate_substeps,
                                                        averaging_shape_function,
                                                        ForwardBackwardScheme,
                                                        _split_explicit_free_surface!,
                                                        _split_explicit_barotropic_velocity!

using KernelAbstractions.Extras.LoopInfo: @unroll

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface, SplitExplicitAuxiliaryFields, iterate_split_explicit!

# SplitExplicitFreeSurface on MultiRegionGrids is implemented in two different ways:
# 1. extended_halos == true:  the halos are extended to include the full substepping
# 2. extended_halos == false: the halos are the same as the original grid and halos are filled after each substep
function SplitExplicitFreeSurface(grid::MultiRegionGrids;
                                  gravitational_acceleration = g_Earth,
                                  substeps = nothing,
                                  cfl = nothing,
                                  fixed_Δt = nothing,
                                  extended_halos = true, 
                                  averaging_kernel = averaging_shape_function,
                                  timestepper = ForwardBackwardScheme())

    settings = SplitExplicitSettings(grid;
                                     gravitational_acceleration,
                                     substeps,
                                     cfl,
                                     fixed_Δt,
                                     averaging_kernel,
                                     timestepper)

    
    return SplitExplicitFreeSurface(nothing,
                                    nothing,
                                    extended_halos,
                                    gravitational_acceleration,
                                    settings)
end

function SplitExplicitAuxiliaryFields(grid::MultiRegionGrids; extended_halos = true)

    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    Hᶠᶜ = Field((Face,   Center, Nothing), grid)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid)

    @apply_regionally calculate_column_height!(Hᶠᶜ, (Face, Center, Center))
    @apply_regionally calculate_column_height!(Hᶜᶠ, (Center, Face, Center))

    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ))

    if extended_halos
        # In a non-parallel grid we calculate only the interior
        @apply_regionally kernel_size    = augmented_kernel_size(grid, grid.partition)
        @apply_regionally kernel_offsets = augmented_kernel_offsets(grid, grid.partition)

        @apply_regionally kernel_parameters = KernelParameters(kernel_size, kernel_offsets)
    else
        kernel_parameters = nothing
    end

    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, kernel_parameters)
end

@inline function calculate_column_height!(height, location)
    dz = GridMetricOperation(location, Δz, height.grid)
    sum!(height, dz)
    return nothing
end

@inline augmented_kernel_size(grid, ::XPartition)           = (size(grid, 1) + 2halo_size(grid)[1]-2, size(grid, 2))
@inline augmented_kernel_size(grid, ::YPartition)           = (size(grid, 1), size(grid, 2) + 2halo_size(grid)[2]-2)
@inline augmented_kernel_size(grid, ::CubedSpherePartition) = (size(grid, 1) + 2halo_size(grid)[1]-2, size(grid, 2) + 2halo_size(grid)[2]-2)

@inline augmented_kernel_offsets(grid, ::XPartition) = (- halo_size(grid)[1] + 1, 0)
@inline augmented_kernel_offsets(grid, ::YPartition) = (0, - halo_size(grid)[2] + 1)
@inline augmented_kernel_offsets(grid, ::CubedSpherePartition) = (- halo_size(grid)[2] + 1, - halo_size(grid)[2] + 1)

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::MultiRegionGrids)
    settings = SplitExplicitSettings(grid; free_surface.settings.settings_kwargs...)

    settings.substepping isa FixedTimeStepSize &&
        throw(ArgumentError("SplitExplicitFreeSurface on MultiRegionGrids only suports FixedSubstepNumber; re-initialize SplitExplicitFreeSurface using substeps kwarg"))

    switch_device!(grid.devices[1])

    extended_halos = free_surface.auxiliary 

    if extended_halos
        old_halos = halo_size(getregion(grid, 1))
        Nsubsteps = calculate_substeps(settings.substepping)

        new_halos = multiregion_split_explicit_halos(old_halos, Nsubsteps+1, grid.partition)
        new_grid  = with_halo(new_halos, grid)
    else
        new_grid = grid
    end

    η = ZFaceField(new_grid, indices = (:, :, size(new_grid, 3)+1))

    return SplitExplicitFreeSurface(η,
                                    SplitExplicitState(new_grid, free_surface.settings.timestepper),
                                    SplitExplicitAuxiliaryFields(new_grid; extended_halos),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

@inline multiregion_split_explicit_halos(old_halos, step_halo, ::XPartition) = (max(step_halo, old_halos[1]), old_halos[2], old_halos[3])
@inline multiregion_split_explicit_halos(old_halos, step_halo, ::YPartition) = (old_halos[1], max(step_halo, old_halo[2]), old_halos[3])

const FillHaloSplitExplicit = SplitExplicitFreeSurface{<:Any, <:Any, <:SplitExplicitAuxiliaryFields{<:Any, <:Any, <:Nothing}}

# Fallback!
iterate_split_explicit!(free_surface, grid::MultiRegionGrids, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps = 
        @apply_regionally iterate_split_explicit!(free_surface, grid, Δτᴮ, weights, Val(Nsubsteps))

# Fill the halos after each substep
iterate_split_explicit!(free_surface::FillHaloSplitExplicit, grid::MultiRegionGrids, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps = 
        fill_halo_iterate_split_explicit!(free_surface, grid, Δτᴮ, weights, Val(Nsubsteps))

function fill_halo_iterate_split_explicit!(free_surface, grid::MultiRegionGrids, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η         = free_surface.η
    state     = free_surface.state
    auxiliary = free_surface.auxiliary
    settings  = free_surface.settings
    g         = free_surface.gravitational_acceleration

    # unpack state quantities, parameters and forcing terms 
    U, V             = state.U,    state.V
    Uᵐ⁻¹, Uᵐ⁻²       = state.Uᵐ⁻¹, state.Uᵐ⁻²
    Vᵐ⁻¹, Vᵐ⁻²       = state.Vᵐ⁻¹, state.Vᵐ⁻²
    ηᵐ, ηᵐ⁻¹, ηᵐ⁻²   = state.ηᵐ,   state.ηᵐ⁻¹, state.ηᵐ⁻²
    η̅, U̅, V̅          = state.η̅, state.U̅, state.V̅
    Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ = auxiliary.Gᵁ, auxiliary.Gⱽ, auxiliary.Hᶠᶜ, auxiliary.Hᶜᶠ

    timestepper = settings.timestepper

    η_args = (grid, Δτᴮ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
              U, V, Uᵐ⁻¹, Uᵐ⁻², Vᵐ⁻¹, Vᵐ⁻², 
              timestepper)

    U_args = (grid, Δτᴮ, η, ηᵐ, ηᵐ⁻¹, ηᵐ⁻², 
              U, Uᵐ⁻¹, Uᵐ⁻², V,  Vᵐ⁻¹, Vᵐ⁻²,
              η̅, U̅, V̅, Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, g, 
              timestepper)

    @unroll for substep in 1:Nsubsteps
        averaging_weight = weights[substep]

        fill_halo_regions!((U, V))
        @apply_regionally advance_free_surface_step!(arch, grid, η_args)
            
        fill_halo_regions!(η)
        @apply_regionally advance_barotropic_velocity_step!(arch, grid, averaging_weight, U_args)
    end

    return nothing
end

advance_free_surface_step!(arch, grid, η_args)        = launch!(arch, grid, :xy, _split_explicit_free_surface!, η_args...)
advance_barotropic_velocity_step!(arch, grid, averaging_weight, U_args) = launch!(arch, grid, :xy, _split_explicit_barotropic_velocity!, averaging_weight, U_args...)