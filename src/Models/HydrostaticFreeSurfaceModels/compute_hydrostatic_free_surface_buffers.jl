using Oceananigans.Utils: get_active_cells_map
using Oceananigans.Grids: halo_size, XFlatGrid, YFlatGrid
using Oceananigans.ImmersedBoundaries: SplitActiveCellsMapIBG
using Oceananigans.DistributedComputations: Distributed, DistributedGrid, AsynchronousDistributed, synchronize_communication!
using Oceananigans.Models: buffer_tendency_kernel_parameters, buffer_parameters
using Oceananigans.Models.NonhydrostaticModels: buffer_κ_kernel_parameters

# Fallback for non-distributed grids
complete_communication_and_compute_tracer_buffer!(model, grid, arch) = nothing
complete_communication_and_compute_momentum_buffer!(model, grid, arch) = nothing

"""
    complete_communication_and_compute_momentum_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)

Complete halo communication and compute momentum tendencies in the buffer regions for distributed grids.

This method is called after interior momentum tendencies are computed to:
1. synchronize halo communication for tracers and velocities,
2. compute diagnostic fields (buoyancy gradients, vertical velocity, pressure, closure_fields) in the buffer regions, and
3. compute momentum tendencies in cells that depend on halo data.
"""
function complete_communication_and_compute_momentum_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    # Synchronize tracers
    for tracer in model.tracers
        synchronize_communication!(tracer)
    end

    surface_params = buffer_surface_kernel_parameters(grid, arch)
    volume_params  = buffer_volume_kernel_parameters(grid, arch)

    κ_params = buffer_κ_kernel_parameters(grid, model.closure, arch)

    compute_buoyancy_gradients!(model.buoyancy, grid, model.tracers, parameters = volume_params)
    update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = surface_params)
    compute_closure_fields!(model.closure_fields, model.closure, model; parameters = κ_params)

    fill_halo_regions!(model.closure_fields; only_local_halos=true)

    # parameters for communicating North / South / East / West side
    @apply_regionally compute_momentum_buffer_contributions!(grid, arch, model)

    return nothing
end

"""
    compute_momentum_buffer_contributions!(grid::DistributedGrid, arch, model)

Compute momentum tendencies in the four halo-dependent buffer regions.
"""
function compute_momentum_buffer_contributions!(grid::DistributedGrid, arch, model)
    params = buffer_tendency_kernel_parameters(grid, arch)
    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)
        compute_buffer_region_momentum_tendencies!(grid, arch, model, name, params)
    end
    return nothing
end

@inline buffer_region_active_cells_map(grid::SplitActiveCellsMapIBG, name) = @inbounds grid.interior_active_cells[name]
@inline buffer_region_active_cells_map(grid, name) = nothing

@inline function compute_buffer_region_momentum_tendencies!(grid, arch, model, name, params)
    kernel_parameters = params[name]
    isnothing(kernel_parameters) && return nothing
    active_cells_map = buffer_region_active_cells_map(grid, name)
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map, region=name)
    return nothing
end

"""
    complete_communication_and_compute_tracer_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)

Complete halo communication and compute tracer tendencies in buffer regions for distributed grids.

This function is called after interior tracer tendencies are computed to:
1. synchronize halo communication for transport velocities and free surface,
2. update the vertical transport velocities in buffer regions, and
3. compute the tracer tendencies in cells that depend on halo data.
"""
function complete_communication_and_compute_tracer_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    # Synchronize velocities and free surface
    synchronize_communication!(model.free_surface)
    synchronize_communication!(model.velocities.u)
    synchronize_communication!(model.velocities.v)

    compute_tracer_buffer_contributions!(grid, arch, model)

    return nothing
end

"""
    compute_tracer_buffer_contributions!(grid::DistributedGrid, arch, model)

Compute tracer tendencies in the halo-dependent buffer regions (from 1 to 4).
"""
function compute_tracer_buffer_contributions!(grid::DistributedGrid, arch, model)
    params = buffer_tendency_kernel_parameters(grid, arch)
    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)
        compute_buffer_region_tracer_tendencies!(grid, arch, model, name, params)
    end
    return nothing
end

@inline function compute_buffer_region_tracer_tendencies!(grid, arch, model, name, params)
    kernel_parameters = params[name]
    isnothing(kernel_parameters) && return nothing # This side is not at a processor boundary
    active_cells_map = buffer_region_active_cells_map(grid, name)
    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map, region=name)
    return nothing
end

"""
    buffer_surface_kernel_parameters(grid, arch)

Return kernel parameters for computing 2D (surface) variables in buffer regions.

The buffer regions are strips along processor boundaries where computations depend on halo data.
Returns parameters for west, east, south, and north buffer regions.
"""
function buffer_surface_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    xside = isa(grid, XFlatGrid) ? UnitRange(1, Nx) : UnitRange(-Hx+2, Nx+Hx-1)
    yside = isa(grid, YFlatGrid) ? UnitRange(1, Ny) : UnitRange(-Hy+2, Ny+Hy-1)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    yside)
    param_east  = (Nx:Nx+Hx-1, yside)
    param_south = (xside,     -Hy+2:1)
    param_north = (xside,     Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end

"""
    buffer_volume_kernel_parameters(grid, arch)

Return kernel parameters for computing 3D (volume) variables in buffer regions.

Similar to `buffer_surface_kernel_parameters` but for three-dimensional fields.
The buffer regions span the full vertical extent of the grid.
"""
function buffer_volume_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    xside = isa(grid, XFlatGrid) ? UnitRange(1, Nx) : UnitRange(-Hx+2, Nx+Hx-1)
    yside = isa(grid, YFlatGrid) ? UnitRange(1, Ny) : UnitRange(-Hy+2, Ny+Hy-1)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west   = (-Hx+2:1,    yside,     1:Nz)
    param_east   = (Nx:Nx+Hx-1, yside,     1:Nz)
    param_south  = (xside,     -Hy+2:1,    1:Nz)
    param_north  = (xside,     Ny:Ny+Hy-1, 1:Nz)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end
