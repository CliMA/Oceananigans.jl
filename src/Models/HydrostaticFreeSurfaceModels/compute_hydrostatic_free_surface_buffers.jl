import Oceananigans.Models: compute_buffer_tendencies!

using Oceananigans.Grids: halo_size
using Oceananigans.DistributedComputations: Distributed, DistributedGrid
using Oceananigans.ImmersedBoundaries: get_active_cells_map, CellMaps
using Oceananigans.Models.NonhydrostaticModels: buffer_tendency_kernel_parameters,
                                                buffer_p_kernel_parameters,
                                                buffer_κ_kernel_parameters,
                                                buffer_parameters

const DistributedActiveInteriorIBG = ImmersedBoundaryGrid{FT, TX, TY, TZ,
                                                          <:DistributedGrid, I, <:CellMaps, S,
                                                          <:Distributed} where {FT, TX, TY, TZ, I, S}

# We assume here that top/bottom BC are always synchronized (no partitioning in z)
function compute_buffer_tendencies!(model::HydrostaticFreeSurfaceModel)
    grid = model.grid
    arch = architecture(grid)
    parameters = buffer_tendency_kernel_parameters(grid, arch)

    compute_diffusivities!(diffusivity, closure, model; parameters)
    fill_halo_regions!(model.diffusivity_fields; only_local_halos=true)

    # parameters for communicating North / South / East / West side
    compute_buffer_tendency_contributions!(grid, arch, model)

    return nothing
end

function compute_buffer_tendency_contributions!(grid, arch, model)
    parameters = buffer_tendency_kernel_parameters(grid, arch)
    compute_hydrostatic_free_surface_tendency_contributions!(model, parameters; active_cells_map)
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, (parameters, parameters); active_cells_map)
    return nothing
end

function compute_buffer_tendency_contributions!(grid::DistributedActiveInteriorIBG, arch, model)
    maps = grid.interior_active_cells

    for name in (:west_halo_dependent_cells,
                 :east_halo_dependent_cells,
                 :south_halo_dependent_cells,
                 :north_halo_dependent_cells)

        active_cells_map = @inbounds maps[name]

        # If the map == nothing, we don't need to compute the buffer because
        # the buffer is not adjacent to a processor boundary
        if !isnothing(active_cells_map) 
            compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map)
            compute_hydrostatic_momentum_tendencies!(model, model.velocities, (:xyz, :xyz); active_cells_map)
        end
    end

    return nothing
end
