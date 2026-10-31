#####
##### Materialised exchange plans for six-panel cubed spheres: the validation, the regional arrays, the panel side tables and the
##### kernel configuration of a halo exchange are built ONCE; a fill is then a single kernel launch with isbits arguments (no
##### per-call validation, table construction, KernelParameters or occupancy query). The fills themselves are the batched
##### kernels of cubed_sphere_cgrid_halos.jl and cubed_sphere_dgrid_halos.jl, unchanged; the centre-scalar plan adds the same
##### one-launch form for `(Center, Center)` scalars (side strips of depth `halo_width`, no corner blocks, no vertical halo).
#####

using Oceananigans.Utils: configure_kernel, KernelParameters

abstract type AbstractCubedSphereExchange end

struct CGridPairExchange{P, M, B, K} <: AbstractCubedSphereExchange
    pairs :: P              # ((uc regions, vc regions, Nz), ...)
    maps :: M               # NTuple{24, PanelSideMap}
    neighbors :: B          # NTuple{24, Int}
    N :: Int
    Nz :: Int
    halo_width :: Int
    signed :: Bool
    kernel! :: K            # configured once over (1:N+1, 1:Nz, 1:24)
end

struct DGridPairExchange{P, M, B, K} <: AbstractCubedSphereExchange
    pairs :: P
    maps :: M
    neighbors :: B
    N :: Int
    Nz :: Int
    halo_width :: Int
    signed :: Bool
    kernel! :: K
end

struct SharedEdgeExchange{P, M, B, K} <: AbstractCubedSphereExchange
    pairs :: P
    north_maps :: M
    east_maps :: M
    north :: B
    east :: B
    N :: Int
    Nz :: Int
    signed :: Bool
    kernel! :: K
end

struct CenterScalarExchange{S, M, B, K} <: AbstractCubedSphereExchange
    scalars :: S            # ((q regions, Nz), ...)
    maps :: M
    neighbors :: B
    N :: Int
    Nz :: Int
    halo_width :: Int
    kernel! :: K
end

configured(grid, params, kernel!) = first(configure_kernel(grid.architecture, getregion(grid, 1), params, kernel!))

"""
    CGridPairExchange(pairs::Tuple; halo_width = 3, signed = true)
    CGridPairExchange(uc, vc; kwargs...)

The materialised form of [`fill_cgrid_pair_halo_regions!`](@ref): the same checks, tables and kernel, prepared once;
`fill_halo_regions!(plan)` is one launch.
"""
function CGridPairExchange(pairs::Tuple; halo_width = 3, signed = true)
    foreach(pair -> pair isa Tuple && length(pair) == 2 && validate_cgrid_pair(pair..., halo_width), pairs)
    grid = validate_pair_batch(pairs, "C-grid exchange")
    N = grid.Nx
    regional_pairs = map(((uc, vc),) -> regional_pair(uc, vc), pairs)
    Nz = maximum(last, regional_pairs)
    maps, neighbors = panel_side_tables(grid)
    kernel! = configured(grid, KernelParameters(1:N+1, 1:Nz, 1:24), _fill_cubed_sphere_cgrid_halos!)
    return CGridPairExchange(regional_pairs, maps, neighbors, N, Nz, Int(halo_width), signed, kernel!)
end
CGridPairExchange(uc, vc; kwargs...) = CGridPairExchange(((uc, vc),); kwargs...)

"""
    DGridPairExchange(pairs::Tuple; halo_width = Hx - 1, signed = true)
    DGridPairExchange(u, v; kwargs...)

The materialised form of [`fill_dgrid_pair_halo_regions!`](@ref).
"""
function DGridPairExchange(pairs::Tuple; signed = true, halo_width = nothing)
    validate_pair_tuple(pairs, "D-grid exchange")
    w = isnothing(halo_width) ? first(first(pairs)).grid.Hx - 1 : halo_width
    w >= 1 || throw(ArgumentError("halo_width = $w must be at least 1"))
    foreach(((u, v),) -> validate_dgrid_pair(u, v, w), pairs)
    grid = validate_pair_batch(pairs, "D-grid exchange")
    N = grid.Nx
    regional_pairs = map(((u, v),) -> regional_pair(u, v), pairs)
    Nz = maximum(last, regional_pairs)
    maps, neighbors = panel_side_tables(grid)
    kernel! = configured(grid, KernelParameters(1:N+1, 1:Nz, 1:24), _fill_cubed_sphere_dgrid_halos!)
    return DGridPairExchange(regional_pairs, maps, neighbors, N, Nz, Int(w), signed, kernel!)
end
DGridPairExchange(u::DGridU, v::DGridV; kwargs...) = DGridPairExchange(((u, v),); kwargs...)

"""
    SharedEdgeExchange(pairs::Tuple; signed = true)
    SharedEdgeExchange(u, v; signed = true)

The materialised form of [`synchronize_shared_edges!`](@ref).
"""
function SharedEdgeExchange(pairs::Tuple; signed = true)
    validate_pair_tuple(pairs, "shared-edge synchronization")
    foreach(((u, v),) -> validate_dgrid_pair(u, v, 0), pairs)
    grid = validate_pair_batch(pairs, "shared-edge synchronization")
    N = grid.Nx
    regional_pairs = map(((u, v),) -> regional_pair(u, v), pairs)
    Nz = maximum(last, regional_pairs)
    north_maps, east_maps, north, east = shared_edge_tables(grid)
    kernel! = configured(grid, KernelParameters(1:N, 1:Nz, 1:6), _synchronize_cubed_sphere_dgrid_edges!)
    return SharedEdgeExchange(regional_pairs, north_maps, east_maps, north, east, N, Nz, signed, kernel!)
end
SharedEdgeExchange(u::DGridU, v::DGridV; signed = true) = SharedEdgeExchange(((u, v),); signed)

# the centre point (i, j) of a halo strip reads the neighbour's centre at the image of its odd doubled coordinates
@inline function fill_center_side_point!(q, q_neighbor, map, side, t, k, N, halo_width)
    for d in 1:halo_width
        if side == 1
            i, j = 1 - d, t
        elseif side == 2
            i, j = N + d, t
        elseif side == 3
            i, j = t, 1 - d
        else
            i, j = t, N + d
        end
        X = 2i - 1; Y = 2j - 1
        Xn = map.R₁₁ * X + map.R₁₂ * Y + map.Tx
        Yn = map.R₂₁ * X + map.R₂₂ * Y + map.Ty
        @inbounds q[i, j, k] = q_neighbor[(Xn + 1) ÷ 2, (Yn + 1) ÷ 2, k]
    end
    return nothing
end

@inline fill_center_scalars!(::Tuple{}, map, region, neighbor, side, t, k, N, halo_width) = nothing

@inline function fill_center_scalars!(scalars::Tuple, map, region, neighbor, side, t, k, N, halo_width)
    qs, Nz = first(scalars)
    if k <= Nz
        @inbounds fill_center_side_point!(qs[region], qs[neighbor], map, side, t, k, N, halo_width)
    end
    return fill_center_scalars!(Base.tail(scalars), map, region, neighbor, side, t, k, N, halo_width)
end

@kernel function _fill_cubed_sphere_center_halos!(scalars, maps, neighbors, N, halo_width)
    t, k, s = @index(Global, NTuple)
    @inbounds fill_center_scalars!(scalars, maps[s], side_region(s), neighbors[s], side_code(s), t, k, N, halo_width)
end

# only `nothing` writes no vertical halo: the default FluxBoundaryCondition(nothing) fills one halo level above and below
function validate_no_vertical_boundary_conditions(field, description)
    for region in 1:6
        bcs = getregion(field, region).boundary_conditions
        (bcs.top === nothing && bcs.bottom === nothing) ||
            throw(ArgumentError("$description fills horizontal halos only; region $region has vertical boundary conditions (top $(bcs.top), bottom $(bcs.bottom))"))
    end
    return nothing
end

"""
    CenterScalarExchange(fields::Tuple; halo_width = Hx)
    CenterScalarExchange(q; halo_width = Hx)

One launch over all regions, sides and levels that writes the side strips of depth `halo_width` (no corner blocks, no vertical
halos) of `(Center, Center)` scalars on a six-panel cubed sphere from the neighbouring panels, value for value as
`fill_halo_regions!(q; halo_width)` does with its per-region, per-side launches. The fields must have no vertical boundary
conditions (the vertical fill of `fill_halo_regions!` is not part of the plan). Checks run at construction only.
"""
function CenterScalarExchange(fields::Tuple; halo_width = nothing)
    (!isempty(fields) && all(f -> f isa CubedSphereField{<:Center, <:Center}, fields)) ||
        throw(ArgumentError("CenterScalarExchange expects a nonempty tuple of (Center, Center) cubed-sphere fields"))
    grid = first(fields).grid
    all(f -> f.grid === grid, fields) || throw(ArgumentError("CenterScalarExchange: every field must live on the same grid"))
    w = isnothing(halo_width) ? grid.Hx : halo_width
    foreach(f -> validate_partial_width_scalar_fill(f, w), fields)
    foreach(f -> validate_no_vertical_boundary_conditions(f, "CenterScalarExchange"), fields)
    reject_overlapping_storage(reduce(vcat, [regional_parents(f) for f in fields]), "CenterScalarExchange")
    N = grid.Nx
    scalars = map(f -> (f.data.regional_objects, size(getregion(f, 1), 3)), fields)
    Nz = maximum(last, scalars)
    maps, neighbors = panel_side_tables(grid)
    kernel! = configured(grid, KernelParameters(1:N, 1:Nz, 1:24), _fill_cubed_sphere_center_halos!)
    return CenterScalarExchange(scalars, maps, neighbors, N, Nz, Int(w), kernel!)
end
CenterScalarExchange(q::CubedSphereField; kwargs...) = CenterScalarExchange((q,); kwargs...)

BoundaryConditions.fill_halo_regions!(e::CGridPairExchange) = (e.kernel!(e.pairs, e.maps, e.neighbors, e.N, e.halo_width, e.signed); nothing)
BoundaryConditions.fill_halo_regions!(e::DGridPairExchange) = (e.kernel!(e.pairs, e.maps, e.neighbors, e.N, e.halo_width, e.signed); nothing)
BoundaryConditions.fill_halo_regions!(e::SharedEdgeExchange) = (e.kernel!(e.pairs, e.north_maps, e.east_maps, e.north, e.east, e.N, e.signed); nothing)
BoundaryConditions.fill_halo_regions!(e::CenterScalarExchange) = (e.kernel!(e.scalars, e.maps, e.neighbors, e.N, e.halo_width); nothing)

Base.summary(e::CGridPairExchange) = "CGridPairExchange($(length(e.pairs)) pair(s), N = $(e.N), Nz = $(e.Nz), halo_width = $(e.halo_width), signed = $(e.signed))"
Base.summary(e::DGridPairExchange) = "DGridPairExchange($(length(e.pairs)) pair(s), N = $(e.N), Nz = $(e.Nz), halo_width = $(e.halo_width), signed = $(e.signed))"
Base.summary(e::SharedEdgeExchange) = "SharedEdgeExchange($(length(e.pairs)) pair(s), N = $(e.N), Nz = $(e.Nz), signed = $(e.signed))"
Base.summary(e::CenterScalarExchange) = "CenterScalarExchange($(length(e.scalars)) field(s), N = $(e.N), Nz = $(e.Nz), halo_width = $(e.halo_width))"
Base.show(io::IO, e::AbstractCubedSphereExchange) = print(io, summary(e))
