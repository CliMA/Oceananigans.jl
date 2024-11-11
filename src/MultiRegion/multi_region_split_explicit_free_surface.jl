using Oceananigans.Utils
using Oceananigans.AbstractOperations: GridMetricOperation, Δz
using Oceananigans.Models.HydrostaticFreeSurfaceModels: free_surface_displacement_field
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: calculate_substeps, 
                                                                                  barotropic_bc, 
                                                                                  materialize_timestepper

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::MultiRegionGrids)

    free_surface.substepping isa FixedTimeStepSize &&
        throw(ArgumentError("SplitExplicitFreeSurface on MultiRegionGrids only suports FixedSubstepNumber; re-initialize SplitExplicitFreeSurface using substeps kwarg"))

    switch_device!(grid.devices[1])

    old_halos = halo_size(getregion(grid, 1))
    Nsubsteps = calculate_substeps(free_surface.substepping)

    extended_halos = multiregion_split_explicit_halos(old_halos, Nsubsteps+1, grid.partition)
    extended_grid  = with_halo(extended_halos, grid)

    η = free_surface_displacement_field(velocities, free_surface, extended_grid)
    η̅ = free_surface_displacement_field(velocities, free_surface, extended_grid)

    u_baroclinic = velocities.u
    v_baroclinic = velocities.v

    @apply_regionally u_bc = barotropic_bc(u_baroclinic)
    @apply_regionally v_bc = barotropic_bc(v_baroclinic)

    U = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = u_bc)
    V = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = v_bc)

    U̅ = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = u_bc)
    V̅ = Field{Center, Center, Nothing}(extended_grid, boundary_conditions = v_bc)

    filtered_state = (η = η̅, U = U̅, V = V̅)
    barotropic_velocities = (U = U, V = V)

    gravitational_acceleration = convert(eltype(extended_grid), free_surface.gravitational_acceleration)
    timestepper = materialize_timestepper(free_surface.timestepper, extended_grid, free_surface, velocities, u_bc, v_bc)

    # In a non-parallel grid we calculate only the interior
    @apply_regionally kernel_size    = augmented_kernel_size(grid, grid.partition)
    @apply_regionally kernel_offsets = augmented_kernel_offsets(grid, grid.partition)

    @apply_regionally kernel_parameters = KernelParameters(kernel_size, kernel_offsets)

    return SplitExplicitFreeSurface(η,
                                    barotropic_velocities,
                                    filtered_state,
                                    gravitational_acceleration,
                                    kernel_parameters,
                                    free_surface.substepping,
                                    timestepper)
end

@inline multiregion_split_explicit_halos(old_halos, step_halo, ::XPartition) = (max(step_halo, old_halos[1]), old_halos[2], old_halos[3])
@inline multiregion_split_explicit_halos(old_halos, step_halo, ::YPartition) = (old_halos[1], max(step_halo, old_halo[2]), old_halos[3])

@inline augmented_kernel_size(grid, ::XPartition)           = (size(grid, 1) + 2halo_size(grid)[1]-2, size(grid, 2))
@inline augmented_kernel_size(grid, ::YPartition)           = (size(grid, 1), size(grid, 2) + 2halo_size(grid)[2]-2)
@inline augmented_kernel_size(grid, ::CubedSpherePartition) = (size(grid, 1) + 2halo_size(grid)[1]-2, size(grid, 2) + 2halo_size(grid)[2]-2)

@inline augmented_kernel_offsets(grid, ::XPartition) = (- halo_size(grid)[1] + 1, 0)
@inline augmented_kernel_offsets(grid, ::YPartition) = (0, - halo_size(grid)[2] + 1)
@inline augmented_kernel_offsets(grid, ::CubedSpherePartition) = (- halo_size(grid)[2] + 1, - halo_size(grid)[2] + 1)

