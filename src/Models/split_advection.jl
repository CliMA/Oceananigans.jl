using Oceananigans.Advection: AbstractUpwindBiasedAdvectionScheme,
                              VectorInvariant,
                              required_halo_size_x,
                              required_halo_size_y,
                              required_halo_size_z

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid,
                                        ActiveInteriorIBG,
                                        compute_stencil_interior_field,
                                        partition_active_map_by_stencil

using Oceananigans.Grids: topology
using Oceananigans.Utils: launch!

#####
##### Attach precomputed stencil split maps to the grid
#####

# No-op fallback for non-IBG grids or grids without active cells map
attach_stencil_active_cells(grid, advection) = grid

function attach_stencil_active_cells(grid::ActiveInteriorIBG, advection)
    buffer = maximum_stencil_buffer(advection)
    buffer == 0 && return grid

    stencil_field = compute_stencil_interior_field(grid, grid.immersed_boundary, buffer)
    stencil = partition_stencil_maps(grid.interior_active_cells, stencil_field, grid)

    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid.underlying_grid,
                                             grid.immersed_boundary,
                                             grid.interior_active_cells,
                                             grid.active_z_columns,
                                             stencil)
end

# Single map (non-distributed): partition directly into (; interior, boundary)
partition_stencil_maps(acm::AbstractArray, stencil_field, grid) =
    partition_active_map_by_stencil(acm, stencil_field, grid)

# Multi-map (distributed): partition each sub-map independently
function partition_stencil_maps(maps::NamedTuple, stencil_field, grid)
    names = keys(maps)
    values = ntuple(length(names)) do i
        acm = maps[i]
        isnothing(acm) ? nothing : partition_active_map_by_stencil(acm, stencil_field, grid)
    end
    return NamedTuple{names}(values)
end

maximum_stencil_buffer(advection::NamedTuple) = maximum(scheme_buffer, values(advection))
maximum_stencil_buffer(advection) = scheme_buffer(advection)

scheme_buffer(::Any) = 0
scheme_buffer(s::AbstractUpwindBiasedAdvectionScheme) = max(required_halo_size_x(s), required_halo_size_y(s), required_halo_size_z(s))
scheme_buffer(s::VectorInvariant) = 
        max(scheme_buffer(s.vorticity_scheme),
            scheme_buffer(s.vertical_advection_scheme),
            scheme_buffer(s.kinetic_energy_gradient_scheme),
            scheme_buffer(s.divergence_scheme))

#####
##### Interior/boundary split launch for advection kernels on ImmersedBoundaryGrid
#####
##### On IBG, advection kernels use many registers due to immersed boundary
##### conditional_δ checks and runtime red_order preventing coefficient constant-folding.
##### By launching the same kernel with grid.underlying_grid for interior cells (whose full
##### stencil is in the fluid domain), Julia specializes a separate GPU binary without
##### IBG overhead, achieving fewer registers and higher occupancy.
#####

# Fallback: single launch on non-IBG grids
split_advection_launch!(arch, grid, kp, acm, kernel!, Gvel, args...; kwargs...) = 
    launch!(arch, grid, kp, kernel!, Gvel, grid, args...; active_cells_map=acm, kwargs...)

# Resolve the correct (; interior, boundary) stencil pair for a given active_cells_map.
# Non-distributed: stencil is already (; interior, boundary)
get_stencil_pair(stencil::NamedTuple{(:interior, :boundary)}, acm, grid) = stencil

# Distributed: stencil mirrors the 5-part structure of interior_active_cells.
# Look up the matching sub-stencil by identity comparison on GPU arrays.
function get_stencil_pair(stencil::NamedTuple, acm, grid)
    maps = grid.interior_active_cells
    for name in keys(maps)
        if maps[name] === acm
            return @inbounds stencil[name]
        end
    end
    return nothing
end

# Split launch on IBG: interior cells use underlying_grid, boundary cells use IBG.
# Stencil split maps are precomputed and stored in grid.stencil_active_cells.
function split_advection_launch!(arch, grid::ImmersedBoundaryGrid, kp, active_cells_map,
                                   kernel!, Gvel, args...; kwargs...)

    stencil = grid.stencil_active_cells
    if isnothing(active_cells_map) || isnothing(stencil)
        launch!(arch, grid, kp, kernel!, Gvel, grid, args...; active_cells_map, kwargs...)
        return nothing
    end

    # Resolve the stencil pair for this active cells map
    stencil_pair = get_stencil_pair(stencil, active_cells_map, grid)

    # No matching stencil found — fall back to single launch
    if isnothing(stencil_pair)
        launch!(arch, grid, kp, kernel!, Gvel, grid, args...; active_cells_map, kwargs...)
        return nothing
    end

    # Interior cells: underlying_grid → no IBG conditional_δ, coefficients constant-fold
    if length(stencil_pair.interior) > 0
        launch!(arch, grid, kp, kernel!, Gvel, grid.underlying_grid, args...; active_cells_map=stencil_pair.interior, kwargs...)
    end

    # Boundary cells: full IBG immersed boundary handling
    if length(stencil_pair.boundary) > 0
        launch!(arch, grid, kp, kernel!, Gvel, grid, args...; active_cells_map=stencil_pair.boundary, kwargs...)
    end

    return nothing
end
