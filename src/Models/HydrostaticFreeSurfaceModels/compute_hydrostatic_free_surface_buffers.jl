using Oceananigans.Grids: halo_size, XFlatGrid, YFlatGrid, ZFlatGrid, get_active_cells_map
using Oceananigans.DistributedComputations: Distributed, DistributedGrid, AsynchronousDistributed, synchronize_communication!
using Oceananigans.ImmersedBoundaries: CellMaps
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters, buffer_parameters

const DistributedActiveInteriorIBG = ImmersedBoundaryGrid{FT, TX, TY, TZ,
                                                          <:DistributedGrid, I, <:CellMaps, S,
                                                          <:Distributed} where {FT, TX, TY, TZ, I, S}

# Fallback
complete_communication_and_compute_tracer_buffer!(model, grid, arch) = nothing
complete_communication_and_compute_momentum_buffer!(model, grid, arch) = nothing

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function complete_communication_and_compute_momentum_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    # Synchronize tracers
    for tracer in model.tracers
        synchronize_communication!(tracer)
    end
    
    # Synchronize velocities
    synchronize_communication!(model.velocities.u)
    synchronize_communication!(model.velocities.v)

    surface_params = buffer_surface_kernel_parameters(grid, arch)
    volume_params  = buffer_volume_kernel_parameters(grid, arch)

    κ_params = buffer_κ_kernel_parameters(grid, model.closure, arch)

    compute_buoyancy_gradients!(model.buoyancy, grid, tracers, parameters = volume_params)
    update_vertical_velocities!(model.velocities, grid, model; parameters = surface_params)
    update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = surface_params)
    compute_diffusivities!(model.closure_fields, model.closure, model; parameters = κ_params)
    fill_halo_regions!(model.closure_fields; only_local_halos=true)

    # parameters for communicating North / South / East / West side
    @apply_regionally compute_momentum_buffer_contributions!(grid, arch, model)

    return nothing
end

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

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function complete_communication_and_compute_tracer_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    ũ, ṽ, _ = model.transport_velocities
    synchronize_communication!(ũ)
    synchronize_communication!(ṽ)
    synchronize_communication!(model.free_surface)

    surface_params = buffer_surface_kernel_parameters(grid, arch)
    update_vertical_velocities!(model.transport_velocities, grid, model; parameters=surface_params)
    compute_tracer_buffer_contributions!(grid, arch, model)

    return nothing
end

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

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_surface_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    xside = isa(grid, XFlatGrid) ? UnitRange(1, Nx) : UnitRange(0, Nx+1)
    yside = isa(grid, YFlatGrid) ? UnitRange(1, Ny) : UnitRange(0, Ny+1)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    yside)
    param_east  = (Nx:Nx+Hx-1, yside)
    param_south = (xside,     -Hy+2:1)
    param_north = (xside,     Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end

function buffer_volume_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    xside = isa(grid, XFlatGrid) ? UnitRange(1, Nx) : UnitRange(0, Nx+1)
    yside = isa(grid, YFlatGrid) ? UnitRange(1, Ny) : UnitRange(0, Ny+1)
    zside = isa(grid, ZFlatGrid) ? UnitRange(1, Nz) : UnitRange(0, Nz+1)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west   = (-Hx+2:1,    yside,     zside)
    param_east   = (Nx:Nx+Hx-1, yside,     zside)
    param_south  = (xside,     -Hy+2:1,    zside)
    param_north  = (xside,     Ny:Ny+Hy-1, zside)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end
