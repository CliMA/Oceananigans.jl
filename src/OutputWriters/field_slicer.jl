struct FieldSlicer{I, W}
    indices :: I
    with_halos :: W
end

function FieldSlicer(grid=nothing; i=:, j=:, k=:, 
                                   #x=nothing, y=nothing, z=nothing,
                                   with_halos=false)

    #if !isnothing(x)
    #    isnothing(grid) || throw(ArgumentError("Grid must be provided to specify physical slice ranges!"))
    #end


    return FieldSlicer((i, j, k), with_halos)
end

HaloSlicer() = FieldSlicer()

#####
##### Slice of life, err... data
#####

parent_slice_indices(loc, topo, N, H, ::Colon, with_halos) = with_halos ? (1:N+2H) : (H:N+H)

parent_slice_indices(::Type{Face}, ::Type{Bounded}, N, H, ::Colon, with_halos) =
    with_halos ? (1:N+1+2H) : (H:N+1+H)

parent_slice_indices(::Type{Nothing}, args...) = 1:1

right_parent_index_without_halos(loc, topo, N, H, right) = min(N + H, right + H)
right_parent_index_without_halos(::Type{Face}, ::Type{Bounded}, N, H, right) = min(N + H + 1, right + H)

function parent_slice_indices(loc, topo, N, H, rng, with_halos)

    if with_halos
        left = rng[1] + H
        right = rng[end] + H
    else
        left = max(H, rng[1] + H)
        right = right_parent_index_without_halos(loc, topo, N, H, rng[end])
    end

    return left:right
end

"""
    slice(slicer, field)

Returns a view over parent(field) associated with slice.indices.
"""
function slice_parent(slicer, field)

    # Unpack
    Nx, Ny, Nz = field.grid.Nx, field.grid.Ny, field.grid.Nz
    Hx, Hy, Hz = field.grid.Hx, field.grid.Hy, field.grid.Hz
    Lx, Ly, Lz = location(field)
    Tx, Ty, Tz = topology(field)

    x_data_range, y_data_range, z_data_range = slicer.indices

    # Convert slicer indices to parent indices, and managing halos
    x_parent_range = parent_slice_indices(Lx, Tx, Nx, Hx, x_data_range, slicer.with_halos)
    y_parent_range = parent_slice_indices(Ly, Ty, Ny, Hy, y_data_range, slicer.with_halos)
    z_parent_range = parent_slice_indices(Lz, Tz, Nz, Hz, z_data_range, slicer.with_halos)

    return view(parent(field), x_parent_range, y_parent_range, z_parent_range)
end

slice_parent(::Nothing, field) = parent(field)
