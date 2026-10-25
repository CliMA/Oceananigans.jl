#####
##### Halo filling for D-grid vector pairs (u at (Center, Face), v at (Face, Center)) on cubed spheres with one region
##### per panel, and synchronization of the panel-edge values that both neighbours own.
#####
##### Each panel side is described by an integer affine map from this panel's doubled index coordinates
##### (X, Y) = (2x, 2y) — centres at odd, faces at even values — to the neighbour's. The maps are those encoded in the
##### Center-Center halo kernels above. A halo point takes the neighbour's value at the mapped point; when the map
##### rotates the panel the u and v roles exchange and the sign follows the rotated basis vector. Halo values are taken
##### from the panel across the crossed edge, including that panel's own copy of a shared edge. Corner halo regions and
##### the shared north (u, j = N+1) and east (v, i = N+1) edges are not written by `fill_halo_regions!`.
#####
##### Every point a fill or a synchronization reads is an owned point of the neighbour (interior or shared edge), and
##### no fill or synchronization writes an owned point it reads, so all regions, sides and field pairs of one call run
##### in a single launch over (t, k, region side) without changing any value.
#####

struct PanelSideMap
    R₁₁ :: Int
    R₁₂ :: Int
    R₂₁ :: Int
    R₂₂ :: Int
    Tx :: Int
    Ty :: Int
end

function panel_side_map(odd_region, side, N)
    if odd_region
        side === :west  && return PanelSideMap(0, -1, 1, 0, 2N, 2N)
        side === :east  && return PanelSideMap(1, 0, 0, 1, -2N, 0)
        side === :south && return PanelSideMap(1, 0, 0, 1, 0, 2N)
        side === :north && return PanelSideMap(0, 1, -1, 0, -2N, 2N)
    else
        side === :west  && return PanelSideMap(1, 0, 0, 1, 2N, 0)
        side === :east  && return PanelSideMap(0, -1, 1, 0, 2N, -2N)
        side === :south && return PanelSideMap(0, 1, -1, 0, 2N, 2N)
        side === :north && return PanelSideMap(1, 0, 0, 1, 0, -2N)
    end
    throw(ArgumentError("unknown side $side"))
end

const panel_sides = (:west, :east, :south, :north)

# Side s ∈ 1:24 of a six-panel launch is side (s - 1) % 4 + 1 (west, east, south, north) of region (s - 1) ÷ 4 + 1.
@inline side_region(s) = (s - 1) ÷ 4 + 1
@inline side_code(s) = (s - 1) % 4 + 1

function panel_side_tables(grid)
    N = grid.Nx
    connections = grid.connectivity.connections
    maps = ntuple(s -> panel_side_map(isodd(side_region(s)), panel_sides[side_code(s)], N), Val(24))
    neighbors = ntuple(s -> getproperty(getregion(connections, side_region(s)), panel_sides[side_code(s)]).from_rank, Val(24))
    return maps, neighbors
end

function shared_edge_tables(grid)
    N = grid.Nx
    connections = grid.connectivity.connections
    north_maps = ntuple(r -> panel_side_map(isodd(r), :north, N), Val(6))
    east_maps = ntuple(r -> panel_side_map(isodd(r), :east, N), Val(6))
    north = ntuple(r -> getregion(connections, r).north.from_rank, Val(6))
    east = ntuple(r -> getregion(connections, r).east.from_rank, Val(6))
    return north_maps, east_maps, north, east
end

# Value at this panel's u (component = 1) or v (component = 2) point (X, Y), read from the neighbour across `map`.
@inline function dgrid_neighbor_value(u_neighbor, v_neighbor, map, component, X, Y, k, signed)
    Xn = map.R₁₁ * X + map.R₁₂ * Y + map.Tx
    Yn = map.R₂₁ * X + map.R₂₂ * Y + map.Ty
    neighbor_is_u = isodd(Xn)
    s = ifelse(component == 1, ifelse(neighbor_is_u, map.R₁₁, map.R₂₁), ifelse(neighbor_is_u, map.R₁₂, map.R₂₂))
    s = ifelse(signed, s, abs(s))
    @inbounds value = ifelse(neighbor_is_u, u_neighbor[(Xn + 1) ÷ 2, Yn ÷ 2 + 1, k], v_neighbor[Xn ÷ 2 + 1, (Yn + 1) ÷ 2, k])
    return s * value
end

@inline function fill_dgrid_side_point!(u, v, u_neighbor, v_neighbor, map, side, t, k, N, halo_width, signed)
    for d in 1:halo_width
        if side == 1
            iu, ju, ku = 1 - d, t, t <= N + 1
            iv, jv, kv = 1 - d, t, t <= N
        elseif side == 2
            iu, ju, ku = N + d, t, t <= N + 1
            iv, jv, kv = N + 1 + d, t, t <= N
        elseif side == 3
            iu, ju, ku = t, 1 - d, t <= N
            iv, jv, kv = t, 1 - d, t <= N + 1
        else
            iu, ju, ku = t, N + 1 + d, t <= N
            iv, jv, kv = t, N + d, t <= N + 1
        end
        if ku
            @inbounds u[iu, ju, k] = dgrid_neighbor_value(u_neighbor, v_neighbor, map, 1, 2iu - 1, 2(ju - 1), k, signed)
        end
        if kv
            @inbounds v[iv, jv, k] = dgrid_neighbor_value(u_neighbor, v_neighbor, map, 2, 2(iv - 1), 2jv - 1, k, signed)
        end
    end
    return nothing
end

# `pairs` is a tuple of (u regions, v regions, Nz); pairs with fewer levels than the launch skip the extra k.
@inline fill_dgrid_pairs!(::Tuple{}, map, region, neighbor, side, t, k, N, halo_width, signed) = nothing

@inline function fill_dgrid_pairs!(pairs::Tuple, map, region, neighbor, side, t, k, N, halo_width, signed)
    us, vs, Nz = first(pairs)
    if k <= Nz
        @inbounds fill_dgrid_side_point!(us[region], vs[region], us[neighbor], vs[neighbor], map, side, t, k, N,
                                         halo_width, signed)
    end
    return fill_dgrid_pairs!(Base.tail(pairs), map, region, neighbor, side, t, k, N, halo_width, signed)
end

@kernel function _fill_cubed_sphere_dgrid_halos!(pairs, maps, neighbors, N, halo_width, signed)
    t, k, s = @index(Global, NTuple)
    @inbounds fill_dgrid_pairs!(pairs, maps[s], side_region(s), neighbors[s], side_code(s), t, k, N, halo_width, signed)
end

@inline synchronize_dgrid_pairs!(::Tuple{}, args...) = nothing

@inline function synchronize_dgrid_pairs!(pairs::Tuple, north_map, east_map, region, north, east, t, k, N, signed)
    us, vs, Nz = first(pairs)
    if k <= Nz
        @inbounds begin
            us[region][t, N + 1, k] = dgrid_neighbor_value(us[north], vs[north], north_map, 1, 2t - 1, 2N, k, signed)
            vs[region][N + 1, t, k] = dgrid_neighbor_value(us[east], vs[east], east_map, 2, 2N, 2t - 1, k, signed)
        end
    end
    return synchronize_dgrid_pairs!(Base.tail(pairs), north_map, east_map, region, north, east, t, k, N, signed)
end

@kernel function _synchronize_cubed_sphere_dgrid_edges!(pairs, north_maps, east_maps, north, east, N, signed)
    t, k, r = @index(Global, NTuple)
    @inbounds synchronize_dgrid_pairs!(pairs, north_maps[r], east_maps[r], r, north[r], east[r], t, k, N, signed)
end

const DGridU = CubedSphereField{<:Center, <:Face}
const DGridV = CubedSphereField{<:Face, <:Center}

covers(axis, range) = first(axis) <= first(range) && last(range) <= last(axis)

# Rejects every layout the parity side maps do not describe, before any write. `halo_width = 0` checks the storage
# that `synchronize_shared_edges!` touches.
function validate_dgrid_pair(u, v, halo_width)
    (u isa DGridU && v isa DGridV) ||
        throw(ArgumentError("D-grid exchange requires u at (Center, Face) and v at (Face, Center) on a cubed sphere; " *
                            "got $(summary(u)) and $(summary(v))"))
    grid = u.grid
    grid isa MultiRegionGrid || throw(ArgumentError("D-grid exchange requires a MultiRegionGrid, got $(summary(grid))"))
    v.grid === grid || throw(ArgumentError("u and v must live on the same grid"))
    validate_canonical_six_panels(grid, "D-grid exchange")
    N = grid.Nx
    LZ = location(u, 3)
    LZ == location(v, 3) || throw(ArgumentError("u and v must share a vertical location"))
    validate_full_indices(u, "D-grid exchange")
    validate_full_indices(v, "D-grid exchange")
    Nz = LZ === Nothing ? 1 : size(getregion(grid, 1), 3) + (LZ === Face)
    halo_width >= 0 || throw(ArgumentError("halo_width = $halo_width must be nonnegative"))
    w = halo_width
    for region in 1:6
        size(getregion(u, region)) == size(getregion(v, region)) == (N, N, Nz) ||
            throw(ArgumentError("D-grid exchange requires full (unwindowed) Fields of size ($N, $N, $Nz) in every region"))
        u_axes = axes(getregion(u, region).data)
        v_axes = axes(getregion(v, region).data)
        (covers(u_axes[1], 1-w:N+w) && covers(u_axes[2], 1-w:N+1+w) &&
         covers(v_axes[1], 1-w:N+1+w) && covers(v_axes[2], 1-w:N+w) &&
         covers(u_axes[3], 1:Nz) && covers(v_axes[3], 1:Nz)) ||
            throw(ArgumentError("region $region storage cannot hold a D-grid halo of width $w beside the shared " *
                                "north/east edges (u axes $(u_axes[1:2]), v axes $(v_axes[1:2]))"))
    end
    return nothing
end

# Checks shared by every batched pair exchange: a nonempty tuple of two-field tuples on one grid, regional storage of
# one concrete type per field (so a launch can index the regions), and no storage shared between any two regions.
validate_pair_tuple(pairs, description) =
    (pairs isa Tuple && !isempty(pairs) && all(pair -> pair isa Tuple && length(pair) == 2, pairs)) ||
        throw(ArgumentError("$description expects a nonempty tuple of (x, y) field pairs, got $(typeof(pairs))"))

function validate_pair_batch(pairs, description)
    validate_pair_tuple(pairs, description)
    grid = first(first(pairs)).grid
    for (x, y) in pairs
        (x.grid === grid && y.grid === grid) || throw(ArgumentError("$description: every field pair must live on the same grid"))
        for field in (x, y)
            regional = field.data.regional_objects
            regional isa NTuple{6, Any} && isconcretetype(eltype(regional)) ||
                throw(ArgumentError("$description requires six regional arrays of one concrete type per field"))
        end
    end
    reject_overlapping_storage(reduce(vcat, [vcat(regional_parents(x), regional_parents(y)) for (x, y) in pairs]), description)
    return grid
end

regional_pair(x, y) = (x.data.regional_objects, y.data.regional_objects, size(getregion(x, 1), 3))

"""
    fill_dgrid_pair_halo_regions!(pairs::Tuple; halo_width = Hx - 1, signed = true)
    fill_dgrid_pair_halo_regions!(u, v; halo_width = Hx - 1, signed = true)

Fill the horizontal halos of one or more D-grid vector pairs, `pairs = ((u₁, v₁), (u₂, v₂), ...)` with every `u` at
`(Center, Face)` and every `v` at `(Face, Center)` on the same six-panel cubed sphere, to depth `halo_width`. Each
pair is filled exactly as `fill_halo_regions!(u, v)`; all pairs, regions, sides and levels go in one kernel launch.
Pairs may have different vertical locations. All checks run before any write.
"""
function fill_dgrid_pair_halo_regions!(pairs::Tuple; signed = true, halo_width = nothing)
    validate_pair_tuple(pairs, "D-grid exchange")
    w = isnothing(halo_width) ? first(first(pairs)).grid.Hx - 1 : halo_width
    w >= 1 || throw(ArgumentError("halo_width = $w must be at least 1"))
    foreach(((u, v),) -> validate_dgrid_pair(u, v, w), pairs)
    grid = validate_pair_batch(pairs, "D-grid exchange")
    N = grid.Nx
    regional_pairs = map(((u, v),) -> regional_pair(u, v), pairs)
    Nz = maximum(last, regional_pairs)
    maps, neighbors = panel_side_tables(grid)
    launch!(grid.architecture, getregion(grid, 1), KernelParameters(1:N+1, 1:Nz, 1:24),
            _fill_cubed_sphere_dgrid_halos!, regional_pairs, maps, neighbors, N, w, signed)
    return nothing
end

fill_dgrid_pair_halo_regions!(u::DGridU, v::DGridV; kwargs...) = fill_dgrid_pair_halo_regions!(((u, v),); kwargs...)

"""
    fill_halo_regions!(u::CubedSphereField{<:Center, <:Face}, v::CubedSphereField{<:Face, <:Center};
                       signed = true, halo_width = u.grid.Hx - 1)

Fill the horizontal halos of a D-grid vector pair (`u` tangential along x-edges, `v` along y-edges) to depth
`halo_width`, excluding corner halo regions and the shared north/east edges. Vertical halos are not filled.
Only `CubedSpherePartition(R = 1)` is supported. The shared north (`u`, `j = N+1`) and east (`v`, `i = N+1`) edges
occupy the first halo slot, so `halo_width` is at most `Hx - 1`. See [`fill_dgrid_pair_halo_regions!`](@ref) for
several pairs in one launch.
"""
BoundaryConditions.fill_halo_regions!(u::DGridU, v::DGridV, args...; signed = true, halo_width = u.grid.Hx - 1, kwargs...) =
    fill_dgrid_pair_halo_regions!(((u, v),); signed, halo_width)

"""
    synchronize_shared_edges!(u::CubedSphereField{<:Center, <:Face}, v::CubedSphereField{<:Face, <:Center}; signed = true)
    synchronize_shared_edges!(pairs::Tuple; signed = true)

Overwrite each panel's copy of its north (`u` at `j = N+1`) and east (`v` at `i = N+1`) edges with the neighbouring
panel's own value of the same edge, for one D-grid pair or a tuple of pairs `((u₁, v₁), (u₂, v₂), ...)`, in one
kernel launch and without temporary storage: the neighbour's own value sits on its first row or column, which no
synchronization writes. Only `CubedSpherePartition(R = 1)` is supported.
"""
function synchronize_shared_edges!(pairs::Tuple; signed = true)
    validate_pair_tuple(pairs, "shared-edge synchronization")
    foreach(((u, v),) -> validate_dgrid_pair(u, v, 0), pairs)
    grid = validate_pair_batch(pairs, "shared-edge synchronization")
    N = grid.Nx
    regional_pairs = map(((u, v),) -> regional_pair(u, v), pairs)
    Nz = maximum(last, regional_pairs)
    north_maps, east_maps, north, east = shared_edge_tables(grid)
    launch!(grid.architecture, getregion(grid, 1), KernelParameters(1:N, 1:Nz, 1:6),
            _synchronize_cubed_sphere_dgrid_edges!, regional_pairs, north_maps, east_maps, north, east, N, signed)
    return nothing
end

synchronize_shared_edges!(u::DGridU, v::DGridV; signed = true) = synchronize_shared_edges!(((u, v),); signed)
