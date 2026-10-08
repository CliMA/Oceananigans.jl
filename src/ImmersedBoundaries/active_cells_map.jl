using Oceananigans.Architectures: CPU
using Oceananigans.Fields: Field
using Oceananigans.Grids: Grids, AbstractGrid, surface_kernel_parameters, extended_interior_kernel_parameters, volume_kernel_parameters
using Oceananigans.Utils: Utils, contiguousrange, worksize
using KernelAbstractions: @kernel, @index

# REMEMBER: since the active map is stripped out of the grid when `Adapt`ing to the GPU,
# The following types cannot be used to dispatch in kernels!!!

# An IBG with a single interior active cells map that includes the whole :xyz domain
const WholeActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray}

# An IBG with an interior active cells map subdivided in 5 different sub-maps.
# Only used (for the moment) in the case of distributed architectures where the boundary adjacent region
# has to be computed separately, these maps hold the whole interior (; interior), the active region in the
# "halo-independent" part of the domain (; halo_independent_cells), and the "halo-dependent" regions in the west,
# east, north, and south, respectively
const SplitActiveCellsMapIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:NamedTuple}

# `get_active_cells_map` returns the precomputed list of active (non-immersed) cell indices
# associated with a given iteration region. The `Val{...}` symbol selects which region.
#
# When a kernel is launched with one of these workspecs (via `launch!(arch, grid, workspec, ...)`)
# and the grid carries the corresponding map, the kernel is rewritten as a one-dimensional
# loop over the entries of the map, so only active (i, j[, k]) tuples are visited and the
# immersed cells are skipped entirely.
#
#   :xyz   -- the full three-dimensional interior. The map is the flat list of active cells
#             in the range 1:Nx, 1:Ny, 1:Nz. Used by any kernel that would otherwise loop
#             over every interior (i, j, k) point (e.g. tendency computations).
#
#   :xy    -- the "surface" interior: one entry per active horizontal column, i.e. only the
#             (i, j) columns that contain at least one active cell. Used by 2D / column-wise
#             kernels (free-surface / barotropic step, vertical integrals, surface bottom-
#             height computations, masking of horizontal slabs, ...) so that fully immersed
#             columns are skipped.
#
# The remaining symbols are only meaningful for `SplitActiveCellsMapIBG`, where the 3D
# interior map is partitioned into a "halo-independent" core and four disjoint "halo-dependent"
# boundary strips. This split lets distributed runs overlap MPI halo communication with
# computation: the core can be advanced while halo exchanges are still in flight, and the
# boundary strips are launched once the halo exchanges complete.
#
#   :core  -- the halo-independent part of the 3D interior, i.e. cells far enough from the
#             local subdomain edges that their stencils never reach into halo points. For a
#             `WholeActiveCellsMapIBG` (no split, e.g. serial runs) this is just the full
#             interior map and is equivalent to `:xyz`.
#
#   :west  -- halo-dependent strip on the western (-x) edge of the local subdomain.
#   :east  -- halo-dependent strip on the eastern (+x) edge of the local subdomain.
#   :south -- halo-dependent strip on the southern (-y) edge, between the western and eastern strips.
#   :north -- halo-dependent strip on the northern (+y) edge, between the western and eastern strips.
#
# Each of these four strips contains the active cells whose stencils touch a halo region,
# and must therefore wait for the halo exchanges before being computed.
#
# All the maps are views into a single array that also holds the active cells in the halo regions spanned by
# `surface_kernel_parameters` (columns), and `extended_interior_kernel_parameters` and `volume_kernel_parameters`
# (three-dimensional maps). On a grid with maps, these kernel parameters are views into that array too.
@inline Utils.get_active_cells_map(grid::WholeActiveCellsMapIBG, ::Val{:xyz})   = grid.interior_active_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:xyz})   = grid.interior_active_cells.interior
@inline Utils.get_active_cells_map(grid::ActiveZColumnsIBG,      ::Val{:xy})    = grid.active_z_columns
@inline Utils.get_active_cells_map(grid::WholeActiveCellsMapIBG, ::Val{:core})  = grid.interior_active_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:core})  = grid.interior_active_cells.halo_independent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:west})  = grid.interior_active_cells.west_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:east})  = grid.interior_active_cells.east_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:south}) = grid.interior_active_cells.south_halo_dependent_cells
@inline Utils.get_active_cells_map(grid::SplitActiveCellsMapIBG, ::Val{:north}) = grid.interior_active_cells.north_halo_dependent_cells

Grids.surface_kernel_parameters(grid::ActiveZColumnsIBG) = parent(grid.active_z_columns)
Grids.volume_kernel_parameters(grid::ActiveInteriorIBG) = parent(Utils.get_active_cells_map(grid, Val(:xyz)))

# The interior map is preceded by the active cells that `extended_interior_kernel_parameters` adds around it
function Grids.extended_interior_kernel_parameters(grid::ActiveInteriorIBG)
    interior = Utils.get_active_cells_map(grid, Val(:xyz))
    return view(parent(interior), 1:last(only(parentindices(interior))))
end

@inline active_cell(i, j, k, grid, ib) = !immersed_cell(i, j, k, grid, ib)

@inline inside(i, j, k, (ri, rj, rk)::NTuple{3}) = (i ∈ ri) & (j ∈ rj) & (k ∈ rk)
@inline inside(i, j, k, (ri, rj)::NTuple{2})     = (i ∈ ri) & (j ∈ rj)

# The position of the first of `regions` that contains `(i, j, k)`, or `length(regions) + 1` if none does
@inline function region_label(i, j, k, regions)
    label = length(regions) + 1
    for n in length(regions):-1:1
        @inbounds label = ifelse(inside(i, j, k, regions[n]), n, label)
    end
    return label
end

@kernel function _label_active_cells!(labels, grid, ib, regions)
    i, j, k = @index(Global, NTuple)
    @inbounds labels[i, j, k] = active_cell(i, j, k, grid, ib) * region_label(i, j, k, regions)
end

@kernel function _label_active_z_columns!(labels, grid, ib, regions)
    i, j = @index(Global, NTuple)
    active_column = false
    for k in 1:size(grid, 3)
        active_column = active_column | active_cell(i, j, k, grid, ib)
    end
    @inbounds labels[i, j, 1] = active_column * region_label(i, j, 1, regions)
end

index_type(grid) = maximum(size(grid) .+ halo_size(grid)) ≤ typemax(Int16) ? Int16 : Int32

# The indices of `labels` equal to each of `segments` in turn, gathered in one array on the architecture of `grid`
function labelled_indices(grid, labels, segments)
    labels = on_architecture(CPU(), labels)
    indices = NTuple{ndims(labels), index_type(grid)}[]
    lengths = map(segments) do segment
        segment_indices = Tuple.(findall(==(segment), labels))
        append!(indices, segment_indices)
        return length(segment_indices)
    end
    return on_architecture(architecture(grid), indices), lengths
end

"""
$(TYPEDSIGNATURES)

Return the active cells of each of the disjoint `regions` that cover the interior, preceded by their union.
All of them are views into one array that starts with the active cells that `extended_interior_kernel_parameters`
adds around the interior, and ends with the ones that `volume_kernel_parameters` adds around those.
"""
function active_cells_maps(grid, ib, regions)
    N = length(regions)
    extended_interior = contiguousrange(extended_interior_kernel_parameters(grid))

    # Labels 1:N are the regions, N+1 the rest of the extended interior, and N+2 the rest of the volume
    labels = Field{Center, Center, Center}(grid, Int8)
    launch!(architecture(grid), grid, volume_kernel_parameters(grid), _label_active_cells!, labels, grid, ib, (regions..., extended_interior))
    cells, lengths = labelled_indices(grid, labels.data, (N+1, (1:N)..., N+2))

    last_indices = cumsum(lengths)
    region_maps = ntuple(n -> SubArray(cells, (last_indices[n]+1:last_indices[n+1],)), N)
    interior    = SubArray(cells, (last_indices[1]+1:last_indices[N+1],))

    return (interior, region_maps...)
end

function build_active_cells_map(grid, ib)
    Wx, Wy, Wz = worksize(grid)
    return first(active_cells_maps(grid, ib, ((1:Wx, 1:Wy, 1:Wz),)))
end

function build_active_z_columns(grid, ib)
    Wx, Wy, _ = worksize(grid)
    labels = Field{Center, Center, Nothing}(grid, Int8)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _label_active_z_columns!, labels, grid, ib, ((1:Wx, 1:Wy),))
    columns, lengths = labelled_indices(grid, view(labels.data, :, :, 1), (1, 2))
    return SubArray(columns, (1:first(lengths),))
end
