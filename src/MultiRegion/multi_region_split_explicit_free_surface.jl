using Oceananigans.Utils
using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface

import Oceananigans.Models.HydrostaticFreeSurfaceModels: FreeSurface, SplitExplicitAuxiliaryFields

function SplitExplicitAuxiliaryFields(grid::MultiRegionGrids)
    
    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    # In a non-parallel grid we calculate only the interior
    @apply_regionally kernel_size    = augmented_kernel_size(grid, grid.partition)
    @apply_regionally kernel_offsets = augmented_kernel_offsets(grid, grid.partition)
    
    @apply_regionally kernel_parameters = KernelParameters(kernel_size, kernel_offsets)

    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, kernel_parameters)
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

function FreeSurface(free_surface::SplitExplicitFreeSurface, velocities, grid::MultiRegionGrids)

    switch_device!(grid.devices[1])
    old_halos = halo_size(getregion(grid, 1))
    Nsubsteps = length(settings.substepping.averaging_weights)

    new_halos = multiregion_split_explicit_halos(old_halos, Nsubsteps+1, grid.partition)         
    new_grid  = with_halo(new_halos, grid)

    η = ZFaceField(new_grid, indices = (:, :, size(new_grid, 3)+1))

    return SplitExplicitFreeSurface(η,
                                    SplitExplicitState(new_grid, free_surface.settings.timestepper),
                                    SplitExplicitAuxiliaryFields(new_grid),
                                    free_surface.gravitational_acceleration,
                                    free_surface.settings)
end

@inline multiregion_split_explicit_halos(old_halos, step_halo, ::XPartition) = (step_halo, old_halos[2], old_halos[3])
@inline multiregion_split_explicit_halos(old_halos, step_halo, ::YPartition) = (old_halos[1], step_halo, old_halos[3])
