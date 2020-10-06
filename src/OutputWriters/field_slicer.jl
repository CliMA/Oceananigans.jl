"""
    struct FieldSlicer{I, J, K, W}

Slices fields along indices with or without halo regions as specified.
"""
struct FieldSlicer{I, J, K, W}
    i :: I
    j :: J
    k :: K
    with_halos :: W
end

"""
    FieldSlicer(i=Colon(), j=Colon(), k=Colon(), with_halos=false)

Returns `FieldSlicer` that slices a field prior to output or time-averaging.

The keyword arguments `i, j, k` prescribe an `Integer` index or `UnitRange` of indices 
in `x, y, z`, respectively.

The default for `i`, `j`, and `k` is `Colon()` which indicates "all indices".

The keyword `with_halos` denotes whether halo data is saved or not. Halo regions are
sliced off output for `UnitRange` and `Colon` index specifications.
"""
FieldSlicer(; i=Colon(), j=Colon(), k=Colon(), with_halos=false) =
    FieldSlicer(i, j, k, with_halos)

#####
##### Slice of life, err... data
#####

# Integer slice
parent_slice_indices(loc, topo, N, H, i::Int, with_halos) = UnitRange(i + H, i + H)

# Colon slicing
parent_slice_indices(loc, topo, N, H, 
                     ::Colon, with_halos) = with_halos ? UnitRange(1, N+2H) : UnitRange(H+1, H+N)

parent_slice_indices(::Type{Face}, ::Type{Bounded}, N, H,
                     ::Colon, with_halos) = with_halos ? UnitRange(1, N+1+2H) : UnitRange(H+1, H+N+1)

# Slicing along reduced dimensions
parent_slice_indices(::Type{Nothing}, loc, N, H, ::Colon, with_halos) = 1:1
parent_slice_indices(::Type{Nothing}, loc, N, H, ::UnitRange, with_halos) = 1:1
parent_slice_indices(::Type{Nothing}, loc, N, H, ::Int, with_halos) = 1:1

# Safe slice ranges without halos
right_parent_index_without_halos(loc, topo, N, H, right) = min(H + N, right + H)
right_parent_index_without_halos(::Type{Face}, ::Type{Bounded}, N, H, right) = min(H + N + 1, right + H)

function parent_slice_indices(loc, topo, N, H, rng::UnitRange, with_halos)

    if with_halos
        left = rng[1] + H
        right = rng[end] + H
    else
        left = max(H, rng[1] + H)
        right = right_parent_index_without_halos(loc, topo, N, H, rng[end])
    end

    return UnitRange(left, right)
end

"""
    slice(slicer, field)

Returns a view over parent(field) associated with slice.i, slice.j, slice.k.
"""
function slice_parent(slicer, field::AbstractField)

    # Unpack
    Nx, Ny, Nz = field.grid.Nx, field.grid.Ny, field.grid.Nz
    Hx, Hy, Hz = field.grid.Hx, field.grid.Hy, field.grid.Hz
    Lx, Ly, Lz = location(field)
    Tx, Ty, Tz = topology(field)

    x_data_range = slicer.i
    y_data_range = slicer.j
    z_data_range = slicer.k

    # Convert slicer indices to parent indices, and managing halos
    x_parent_range = parent_slice_indices(Lx, Tx, Nx, Hx, x_data_range, slicer.with_halos)
    y_parent_range = parent_slice_indices(Ly, Ty, Ny, Hy, y_data_range, slicer.with_halos)
    z_parent_range = parent_slice_indices(Lz, Tz, Nz, Hz, z_data_range, slicer.with_halos)

    return field.data.parent[x_parent_range, y_parent_range, z_parent_range]
end

slice_parent(::Nothing, field) = parent(field)
