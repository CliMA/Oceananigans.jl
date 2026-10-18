#####
##### FMS-compatible halo filling for C-grid normal-velocity pairs (uc at (Face, Center), vc at (Center, Face)) on
##### cubed spheres with one region per panel, using the panel side maps of the D-grid pair. In doubled index
##### coordinates uc sits at (2(i-1), 2j-1) and vc at (2i-1, 2(j-1)), so a mapped point with even X is a uc donor.
##### Owned shared-edge copies (uc at i = 1, N+1; vc at j = 1, N+1), corner blocks, slots deeper than `halo_width`,
##### vertical halos and interiors are never written; shared edges are not synchronized.
#####

@inline function cgrid_neighbor_value(uc_neighbor, vc_neighbor, map, component, X, Y, k, signed)
    Xn = map.R₁₁ * X + map.R₁₂ * Y + map.Tx
    Yn = map.R₂₁ * X + map.R₂₂ * Y + map.Ty
    neighbor_is_uc = iseven(Xn)
    s = ifelse(component == 1, ifelse(neighbor_is_uc, map.R₁₁, map.R₂₁), ifelse(neighbor_is_uc, map.R₁₂, map.R₂₂))
    s = ifelse(signed, s, abs(s))
    @inbounds value = ifelse(neighbor_is_uc, uc_neighbor[Xn ÷ 2 + 1, (Yn + 1) ÷ 2, k], vc_neighbor[(Xn + 1) ÷ 2, Yn ÷ 2 + 1, k])
    return s * value
end

@kernel function _fill_cubed_sphere_cgrid_side_halo!(uc, vc, uc_neighbor, vc_neighbor, map, side, N, halo_width, signed)
    t, k = @index(Global, NTuple)
    for d in 1:halo_width
        if side === :west
            iu, ju, ku = 1 - d, t, t <= N
            iv, jv, kv = 1 - d, t, t <= N + 1
        elseif side === :east
            iu, ju, ku = N + 1 + d, t, t <= N
            iv, jv, kv = N + d, t, t <= N + 1
        elseif side === :south
            iu, ju, ku = t, 1 - d, t <= N + 1
            iv, jv, kv = t, 1 - d, t <= N
        else
            iu, ju, ku = t, N + d, t <= N + 1
            iv, jv, kv = t, N + 1 + d, t <= N
        end
        if ku
            @inbounds uc[iu, ju, k] = cgrid_neighbor_value(uc_neighbor, vc_neighbor, map, 1, 2(iu - 1), 2ju - 1, k, signed)
        end
        if kv
            @inbounds vc[iv, jv, k] = cgrid_neighbor_value(uc_neighbor, vc_neighbor, map, 2, 2iv - 1, 2(jv - 1), k, signed)
        end
    end
end

function regional_parents(field)
    return [parent(getregion(field, region).data) for region in 1:length(field.grid.partition)]
end

function reject_overlapping_storage(arrays, description)
    for a in 1:length(arrays), b in a+1:length(arrays)
        Base.mightalias(arrays[a], arrays[b]) && throw(ArgumentError("$description: regional storage overlaps (arrays $a and $b)"))
    end
    return nothing
end

function validate_canonical_six_panels(grid, description)
    grid isa MultiRegionGrid || throw(ArgumentError("$description requires a MultiRegionGrid, got $(summary(grid))"))
    partition = grid.partition
    (partition isa CubedSpherePartition && partition.Rx == 1 && partition.Ry == 1 && length(partition) == 6) ||
        throw(ArgumentError("$description supports only CubedSpherePartition(R = 1), one region per panel"))
    canonical = CubedSphereConnectivity(CubedSpherePartition(R = 1)).connections
    for region in 1:6, side in (:west, :east, :south, :north)
        connection = getproperty(getregion(grid.connectivity.connections, region), side)
        expected = getproperty(getregion(canonical, region), side)
        (connection.from_rank == expected.from_rank && typeof(connection.from_side) == typeof(expected.from_side)) ||
            throw(ArgumentError("region $region $side connectivity differs from the canonical six-panel cubed sphere"))
    end
    N = grid.Nx
    for region in 1:6
        size(getregion(grid, region))[1:2] == (N, N) ||
            throw(ArgumentError("region $region is not $N × $N; $description requires square panels of equal size"))
    end
    return nothing
end

full_vertical_size(grid, LZ) = LZ === Nothing ? 1 : size(getregion(grid, 1), 3) + (LZ === Face)

# Size alone cannot detect windows: a halo-inclusive index range keeps the logical size of a full field.
function validate_full_indices(field, description)
    for region in 1:length(field.grid.partition)
        all(index -> index isa Colon, getregion(field.indices, region)) ||
            throw(ArgumentError("$description requires full fields: region $region has indices $(getregion(field.indices, region))"))
    end
    return nothing
end

# Partial-width scalar fill (F2-STEP1-CONTRACTS §6.1): edge strips of depth `halo_width`, no corners, no padding writes.
function validate_partial_width_scalar_fill(field, halo_width)
    grid = field.grid
    validate_canonical_six_panels(grid, "partial-width scalar fill")
    halo_width isa Integer || throw(ArgumentError("halo_width must be an integer, got $(typeof(halo_width))"))
    w = halo_width
    N = grid.Nx
    1 <= w <= min(grid.Hx, grid.Hy) || throw(ArgumentError("halo_width = $w must be in 1:$(min(grid.Hx, grid.Hy))"))
    N >= w || throw(ArgumentError("panel size $N must be at least halo_width = $w"))
    LZ = location(field, 3)
    LZ === Nothing && throw(ArgumentError("partial-width scalar fill does not support vertically reduced fields"))
    validate_full_indices(field, "partial-width scalar fill")
    Nz = full_vertical_size(grid, LZ)
    Hz = grid.Hz
    for region in 1:6
        size(getregion(field, region)) == (N, N, Nz) ||
            throw(ArgumentError("partial-width scalar fill requires full (unwindowed, unreduced) Fields of size ($N, $N, $Nz)"))
        data_axes = axes(getregion(field, region).data)
        # the vertical boundary-condition fill that follows touches the whole vertical halo
        (covers(data_axes[1], 1-w:N+w) && covers(data_axes[2], 1-w:N+w) && covers(data_axes[3], 1-Hz:Nz+Hz)) ||
            throw(ArgumentError("region $region storage cannot hold a scalar halo of width $w and vertical range $(1-Hz):$(Nz+Hz)"))
    end
    reject_overlapping_storage(regional_parents(field), "partial-width scalar fill")
    return w
end

# Rejects, before any write, every layout the parity side maps do not describe and every storage overlap.
function validate_cgrid_pair(uc, vc, halo_width)
    (uc isa Field && vc isa Field) || throw(ArgumentError("C-grid exchange requires Fields, got $(typeof(uc)) and $(typeof(vc))"))
    (location(uc)[1:2] == (Face, Center) && location(vc)[1:2] == (Center, Face)) ||
        throw(ArgumentError("C-grid exchange requires uc at (Face, Center) and vc at (Center, Face); got $(location(uc)[1:2]) and $(location(vc)[1:2])"))
    grid = uc.grid
    validate_canonical_six_panels(grid, "C-grid exchange")
    vc.grid === grid || throw(ArgumentError("uc and vc must live on the same grid"))
    validate_full_indices(uc, "C-grid exchange")
    validate_full_indices(vc, "C-grid exchange")
    halo_width isa Integer || throw(ArgumentError("halo_width must be an integer, got $(typeof(halo_width))"))
    w = halo_width
    N = grid.Nx
    1 <= w <= 3 || throw(ArgumentError("halo_width = $w must be in 1:3"))
    w <= min(grid.Hx, grid.Hy) - 1 || throw(ArgumentError("halo_width = $w must be at most min(Hx, Hy) - 1 = $(min(grid.Hx, grid.Hy) - 1)"))
    N > w || throw(ArgumentError("panel size $N must exceed halo_width = $w"))
    LZ = location(uc, 3)
    LZ == location(vc, 3) || throw(ArgumentError("uc and vc must share a vertical location"))
    LZ === Nothing && throw(ArgumentError("C-grid exchange does not support vertically reduced fields"))
    Nz = full_vertical_size(grid, LZ)
    for region in 1:6
        size(getregion(uc, region)) == size(getregion(vc, region)) == (N, N, Nz) ||
            throw(ArgumentError("C-grid exchange requires full (unwindowed, unreduced) Fields of size ($N, $N, $Nz) in every region"))
        uc_axes = axes(getregion(uc, region).data)
        vc_axes = axes(getregion(vc, region).data)
        (covers(uc_axes[1], 1-w:N+1+w) && covers(uc_axes[2], 1-w:N+w) &&
         covers(vc_axes[1], 1-w:N+w) && covers(vc_axes[2], 1-w:N+1+w) &&
         covers(uc_axes[3], 1:Nz) && covers(vc_axes[3], 1:Nz)) ||
            throw(ArgumentError("region $region storage cannot hold a C-grid halo of width $w beside the shared edges"))
    end
    reject_overlapping_storage(vcat(regional_parents(uc), regional_parents(vc)), "C-grid exchange")
    return nothing
end

"""
    fill_cgrid_pair_halo_regions!(uc, vc; halo_width = 3, signed = true)

Fill the horizontal halos of a C-grid normal-velocity pair (`uc` at `(Face, Center)`, `vc` at `(Center, Face)`) on a
six-panel cubed sphere to depth `halo_width ∈ 1:3`, following the FMS `CGRID_NE` exchange: owned shared-edge copies,
corner blocks, deeper slots, vertical halos and interiors are left untouched. Only `CubedSpherePartition(R = 1)`
with canonical connectivity is supported; all checks run before any write.
"""
function fill_cgrid_pair_halo_regions!(uc, vc; halo_width = 3, signed = true)
    validate_cgrid_pair(uc, vc, halo_width)
    grid = uc.grid
    N = grid.Nx
    uc_regions = uc.data.regional_objects
    vc_regions = vc.data.regional_objects
    Nz = size(getregion(uc, 1), 3)
    for region in 1:6
        connections = getregion(grid.connectivity.connections, region)
        for (side, connection) in ((:west, connections.west), (:east, connections.east),
                                   (:south, connections.south), (:north, connections.north))
            neighbor = connection.from_rank
            launch!(grid.architecture, getregion(grid, region), KernelParameters(1:N+1, 1:Nz),
                    _fill_cubed_sphere_cgrid_side_halo!, uc_regions[region], vc_regions[region],
                    uc_regions[neighbor], vc_regions[neighbor], panel_side_map(isodd(region), side, N), side, N,
                    halo_width, signed)
        end
    end
    return nothing
end
