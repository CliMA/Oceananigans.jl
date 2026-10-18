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

@kernel function _fill_cubed_sphere_dgrid_side_halo!(u, v, u_neighbor, v_neighbor, map, side, N, halo_width, signed)
    t, k = @index(Global, NTuple)
    for d in 1:halo_width
        if side === :west
            iu, ju, ku = 1 - d, t, t <= N + 1
            iv, jv, kv = 1 - d, t, t <= N
        elseif side === :east
            iu, ju, ku = N + d, t, t <= N + 1
            iv, jv, kv = N + 1 + d, t, t <= N
        elseif side === :south
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
end

@kernel function _synchronize_cubed_sphere_dgrid_edges!(u, v, u_north, v_north, u_east, v_east, north_map, east_map, N, signed)
    t, k = @index(Global, NTuple)
    @inbounds begin
        u[t, N + 1, k] = dgrid_neighbor_value(u_north, v_north, north_map, 1, 2t - 1, 2N, k, signed)
        v[N + 1, t, k] = dgrid_neighbor_value(u_east, v_east, east_map, 2, 2N, 2t - 1, k, signed)
    end
end

const DGridU = CubedSphereField{<:Center, <:Face}
const DGridV = CubedSphereField{<:Face, <:Center}

covers(axis, range) = first(axis) <= first(range) && last(range) <= last(axis)

# Rejects every layout the parity side maps do not describe, before any write. `halo_width = 0` checks the storage
# that `synchronize_shared_edges!` touches.
function validate_dgrid_pair(u, v, halo_width)
    grid = u.grid
    grid isa MultiRegionGrid || throw(ArgumentError("D-grid exchange requires a MultiRegionGrid, got $(summary(grid))"))
    v.grid === grid || throw(ArgumentError("u and v must live on the same grid"))
    partition = grid.partition
    (partition isa CubedSpherePartition && partition.Rx == 1 && partition.Ry == 1 && length(partition) == 6) ||
        throw(ArgumentError("D-grid exchange supports only CubedSpherePartition(R = 1), one region per panel"))
    canonical = CubedSphereConnectivity(CubedSpherePartition(R = 1)).connections
    for region in 1:6, side in (:west, :east, :south, :north)
        connection = getproperty(getregion(grid.connectivity.connections, region), side)
        expected = getproperty(getregion(canonical, region), side)
        (connection.from_rank == expected.from_rank && typeof(connection.from_side) == typeof(expected.from_side)) ||
            throw(ArgumentError("region $region $side connectivity differs from the canonical six-panel cubed sphere"))
    end
    N = grid.Nx
    LZ = location(u, 3)
    LZ == location(v, 3) || throw(ArgumentError("u and v must share a vertical location"))
    validate_full_indices(u, "D-grid exchange")
    validate_full_indices(v, "D-grid exchange")
    Nz = LZ === Nothing ? 1 : size(getregion(grid, 1), 3) + (LZ === Face)
    halo_width >= 0 || throw(ArgumentError("halo_width = $halo_width must be nonnegative"))
    w = halo_width
    for region in 1:6
        size(getregion(grid, region))[1:2] == (N, N) ||
            throw(ArgumentError("region $region is not $N × $N; D-grid exchange requires square panels of equal size"))
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

"""
    fill_halo_regions!(u::CubedSphereField{<:Center, <:Face}, v::CubedSphereField{<:Face, <:Center};
                       signed = true, halo_width = u.grid.Hx - 1)

Fill the horizontal halos of a D-grid vector pair (`u` tangential along x-edges, `v` along y-edges) to depth
`halo_width`, excluding corner halo regions and the shared north/east edges. Vertical halos are not filled.
Only `CubedSpherePartition(R = 1)` is supported. The shared north (`u`, `j = N+1`) and east (`v`, `i = N+1`) edges
occupy the first halo slot, so `halo_width` is at most `Hx - 1`.
"""
function BoundaryConditions.fill_halo_regions!(u::DGridU, v::DGridV, args...; signed = true,
                                               halo_width = u.grid.Hx - 1, kwargs...)
    halo_width >= 1 || throw(ArgumentError("halo_width = $halo_width must be at least 1"))
    validate_dgrid_pair(u, v, halo_width)
    grid = u.grid
    N = grid.Nx
    u_regions = u.data.regional_objects
    v_regions = v.data.regional_objects
    Nz = size(getregion(u, 1), 3)
    for region in 1:length(grid.partition)
        connections = getregion(grid.connectivity.connections, region)
        for (side, connection) in ((:west, connections.west), (:east, connections.east),
                                   (:south, connections.south), (:north, connections.north))
            map = panel_side_map(isodd(region), side, N)
            neighbor = connection.from_rank
            launch!(grid.architecture, getregion(grid, region), KernelParameters(1:N+1, 1:Nz),
                    _fill_cubed_sphere_dgrid_side_halo!, u_regions[region], v_regions[region],
                    u_regions[neighbor], v_regions[neighbor], map, side, N, halo_width, signed)
        end
    end
    return nothing
end

"""
    synchronize_shared_edges!(u::CubedSphereField{<:Center, <:Face}, v::CubedSphereField{<:Face, <:Center}; signed = true)

Overwrite each panel's copy of its north (`u` at `j = N+1`) and east (`v` at `i = N+1`) edges with the neighbouring
panel's own value of the same edge. Only `CubedSpherePartition(R = 1)` is supported.
"""
function synchronize_shared_edges!(u::DGridU, v::DGridV; signed = true)
    validate_dgrid_pair(u, v, 0)
    grid = u.grid
    N = grid.Nx
    u_regions = u.data.regional_objects
    v_regions = v.data.regional_objects
    Nz = size(getregion(u, 1), 3)
    u_copies = map(copy, u_regions)
    v_copies = map(copy, v_regions)
    for region in 1:length(grid.partition)
        connections = getregion(grid.connectivity.connections, region)
        north, east = connections.north.from_rank, connections.east.from_rank
        launch!(grid.architecture, getregion(grid, region), KernelParameters(1:N, 1:Nz),
                _synchronize_cubed_sphere_dgrid_edges!, u_regions[region], v_regions[region],
                u_copies[north], v_copies[north], u_copies[east], v_copies[east],
                panel_side_map(isodd(region), :north, N), panel_side_map(isodd(region), :east, N), N, signed)
    end
    return nothing
end
