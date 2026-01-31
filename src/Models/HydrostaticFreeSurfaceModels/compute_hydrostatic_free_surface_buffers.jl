using Oceananigans.Grids: halo_size, XFlatGrid, YFlatGrid, get_active_cells_map
using Oceananigans.DistributedComputations: Distributed, DistributedGrid, AsynchronousDistributed, synchronize_communication!
using Oceananigans.ImmersedBoundaries: CellMaps
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters, buffer_κ_kernel_parameters, buffer_parameters

const DistributedActiveInteriorIBG = ImmersedBoundaryGrid{FT, TX, TY, TZ,
                                                          <:DistributedGrid, I, <:CellMaps, S,
                                                          <:Distributed} where {FT, TX, TY, TZ, I, S}

# Fallback for non-distributed grids
complete_communication_and_compute_tracer_buffer!(model, grid, arch) = nothing
complete_communication_and_compute_momentum_buffer!(model, grid, arch) = nothing

"""
    complete_communication_and_compute_momentum_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)

Complete halo communication and compute momentum tendencies in the buffer regions for distributed grids.

This method is called after interior momentum tendencies are computed to:
1. synchronize halo communication for tracers and velocities,
2. compute diagnostic fields (buoyancy gradients, vertical velocity, pressure, diffusivities) in the buffer regions, and
3. compute momentum tendencies in cells that depend on halo data.
"""
function complete_communication_and_compute_momentum_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    # Synchronize tracers
    for tracer in model.tracers
        synchronize_communication!(tracer)
    end

    # Synchronize velocities and free surface
    synchronize_communication!(model.velocities.u)
    synchronize_communication!(model.velocities.v)

    surface_params = buffer_surface_kernel_parameters(grid, arch)
    volume_params  = buffer_volume_kernel_parameters(grid, arch)

    κ_params = buffer_κ_kernel_parameters(grid, model.closure, arch)

    compute_buoyancy_gradients!(model.buoyancy, grid, model.tracers, parameters = volume_params)
    update_vertical_velocities!(model.velocities, grid, model; parameters = surface_params)
    update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = surface_params)
    compute_diffusivities!(model.closure_fields, model.closure, model; parameters = κ_params)

    fill_halo_regions!(model.closure_fields; only_local_halos=true)

    # parameters for communicating North / South / East / West side
    @apply_regionally compute_momentum_buffer_contributions!(grid, arch, model)

    return nothing
end

"""
    compute_momentum_buffer_contributions!(grid, arch, model)

Compute momentum tendencies in buffer regions adjacent to processor boundaries.

For regular distributed grids, uses `buffer_tendency_kernel_parameters` to determine
the buffer region indices. For immersed boundary grids with active cell maps,
iterates over halo-dependent cell maps for each direction.
"""
function compute_momentum_buffer_contributions!(grid, arch, model)
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters)
    return nothing
end

function compute_momentum_buffer_contributions!(grid::DistributedActiveInteriorIBG, arch, model)
    maps = grid.interior_active_cells

    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)

        active_cells_map = @inbounds maps[name]

        # If the map == nothing, we don't need to compute the buffer because
        # the buffer is not adjacent to a processor boundary.
        if !isnothing(active_cells_map)
            # We pass `nothing` as parameters since we will use the value in the `active_cells_map` as parameters
            compute_hydrostatic_momentum_tendencies!(model, model.velocities, nothing; active_cells_map)
        end
    end

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

    # synchronize the free surface
    synchronize_communication!(model.free_surface)

    # We need to synchronize the transport velocities only on
    # a split-explicit free surface model
    # (with other free surfaces `transport_velocities === velocities`)
    if model.free_surface isa SplitExplicitFreeSurface
        ũ, ṽ, _ = model.transport_velocities
        synchronize_communication!(ũ)
        synchronize_communication!(ṽ)

        surface_params = buffer_surface_kernel_parameters(grid, arch)
        update_vertical_velocities!(model.transport_velocities, grid, model; parameters=surface_params)
    end

    compute_tracer_buffer_contributions!(grid, arch, model)

    return nothing
end

"""
    compute_tracer_buffer_contributions!(grid, arch, model)

Compute tracer tendencies in buffer regions adjacent to processor boundaries.
"""
function compute_tracer_buffer_contributions!(grid, arch, model)
    kernel_parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_tracer_tendencies!(model, kernel_parameters)
    return nothing
end

function compute_tracer_buffer_contributions!(grid::DistributedActiveInteriorIBG, arch, model)
    maps = grid.interior_active_cells

    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)

        active_cells_map = @inbounds maps[name]

        # If the map == nothing, we don't need to compute the buffer because
        # the buffer is not adjacent to a processor boundary
        if !isnothing(active_cells_map)
            # We pass `nothing` as parameters since we will use the value in the `active_cells_map` as parameters
            compute_hydrostatic_tracer_tendencies!(model, nothing; active_cells_map)
        end
    end

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
