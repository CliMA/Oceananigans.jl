import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.DistributedComputations: Distributed, DistributedGrid, AsynchronousDistributed, synchronize_communication!
using Oceananigans.ImmersedBoundaries: get_active_cells_map, CellMaps
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters,
                                                buffer_κ_kernel_parameters,
                                                buffer_parameters

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

    # Iterate over the fields to clear _ALL_ possible architectures
    for field in prognostic_fields(model)
        synchronize_communication!(field)
    end

    w_parameters = buffer_w_kernel_parameters(grid, arch)
    κ_parameters = buffer_κ_kernel_parameters(grid, model.closure, arch)

    update_vertical_velocities!(model.velocities, grid, model; parameters = w_parameters)
    update_hydrostatic_pressure!(model.pressure.pHY′, arch, grid, model.buoyancy, model.tracers; parameters = w_parameters)
    compute_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = κ_parameters)
    fill_halo_regions!(model.diffusivity_fields; only_local_halos=true)

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
        # the buffer is not adjacent to a processor boundary
        !isnothing(active_cells_map) &&
            compute_hydrostatic_momentum_tendencies!(model, model.velocities, :xyz; active_cells_map)
    end

    return nothing
end

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function complete_communication_and_compute_tracer_buffer!(model::HydrostaticFreeSurfaceModel, ::DistributedGrid, ::AsynchronousDistributed)
    grid = model.grid
    arch = architecture(grid)

    # Iterate over the fields to clear _ALL_ possible architectures
    for field in prognostic_fields(model)
        synchronize_communication!(field)
    end

    w_parameters = buffer_w_kernel_parameters(grid, arch)
    update_vertical_velocities!(model.transport_velocities, grid, model; parameters=w_parameters)
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
            compute_hydrostatic_tracer_tendencies!(model, :xyz; active_cells_map)
        end
    end

    return nothing
end

# w needs computing in the range - H + 1 : 0 and N - 1 : N + H - 1
function buffer_w_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    # Offsets in tangential direction are == -1 to
    # cover the required corners
    param_west  = (-Hx+2:1,    0:Ny+1)
    param_east  = (Nx:Nx+Hx-1, 0:Ny+1)
    param_south = (0:Nx+1,     -Hy+2:1)
    param_north = (0:Nx+1,     Ny:Ny+Hy-1)

    params = (param_west, param_east, param_south, param_north)

    return buffer_parameters(params, grid, arch)
end