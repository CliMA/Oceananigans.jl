indices(obj, i=default_indices(3)) = i
indices(f::Field, i=default_indices(3)) = f.indices
indices(a::SubArray, i=default_indices(ndims(a))) = a.indices
indices(a::OffsetArray, i=default_indices(ndims(a))) = indices(parent(a), i)

function interior_x_indices(f::Field)
    loc = map(instantiate, location(f))
    interior_indices = interior_x_indices(f.grid, loc)
    return compute_index_intersection(interior_indices, f.indices[1])
end

function interior_y_indices(f::Field)
    loc = map(instantiate, location(f))
    interior_indices = interior_y_indices(f.grid, loc)
    return compute_index_intersection(interior_indices, f.indices[2])
end

function interior_z_indices(f::Field)
    loc = map(instantiate, location(f))
    interior_indices = interior_z_indices(f.grid, loc)
    return compute_index_intersection(interior_indices, f.indices[3])
end

# Interior indices for a field with a given location and topology
function interior_indices(f::Field)
    ind_x = interior_x_indices(f)
    ind_y = interior_y_indices(f)
    ind_z = interior_z_indices(f)
    return (ind_x, ind_y, ind_z)
end

# Life is pretty simple in this case.
compute_index_intersection(to_idx::Colon, from_idx::Colon, args...) = Colon()

# Because `from_idx` imposes no restrictions, we just return `to_idx`.
compute_index_intersection(to_idx::AbstractUnitRange, from_idx::Colon, args...) = to_idx

# In case of no locations specified, Because `to_idx` imposes no restrictions, we just return `from_idx`.
compute_index_intersection(to_idx::Colon, from_idx::AbstractUnitRange) = from_idx

# This time we account for the possible range-reducing effect of interpolation on `from_idx`.
function compute_index_intersection(to_idx::Colon, from_idx::AbstractUnitRange, to_loc, from_loc)
    shifted_idx = restrict_index_on_location(from_idx, from_loc, to_loc)
    validate_shifted_index(shifted_idx)
    return shifted_idx
end

# Compute the intersection of two index ranges
function compute_index_intersection(to_idx::AbstractUnitRange, from_idx::AbstractUnitRange, to_loc, from_loc)
    shifted_idx = restrict_index_on_location(from_idx, from_loc, to_loc)
    validate_shifted_index(shifted_idx)

    range_intersection = UnitRange(max(first(shifted_idx), first(to_idx)), min(last(shifted_idx), last(to_idx)))

    # Check validity of the intersection index range
    first(range_intersection) > last(range_intersection) &&
        throw(ArgumentError("Indices $(from_idx) and $(to_idx) interpolated from $(from_loc) to $(to_loc) do not intersect!"))

    return range_intersection
end

# Compute the intersection of two index ranges where the location is the same
function compute_index_intersection(to_idx::AbstractUnitRange, from_idx::AbstractUnitRange)
    range_intersection = UnitRange(max(first(from_idx), first(to_idx)),
                                   min(last(from_idx), last(to_idx)))

    # Check validity of the intersection index range
    first(range_intersection) > last(range_intersection) &&
        throw(ArgumentError("Indices $(from_idx) and $(to_idx) do not intersect!"))

    return range_intersection
end

validate_shifted_index(shifted_idx) = first(shifted_idx) > last(shifted_idx) &&
    throw(ArgumentError("Cannot compute index intersection for indices $(from_idx) interpolating from $(from_loc) to $(to_loc)!"))

"""
    restrict_index_on_location(from_idx, from_loc, to_loc)

Return a "restricted" index range for the result of interpolating from
`from_loc` to `to_loc`, over the index range `from_idx`:

* Windowed fields interpolated from `Center`s to `Face`s lose the first index.
* Conversely, windowed fields interpolated from `Face`s to `Center`s lose the last index
"""
restrict_index_on_location(from_idx, ::Type{Face},   ::Type{Face})   = UnitRange(first(from_idx),   last(from_idx))
restrict_index_on_location(from_idx, ::Type{Center}, ::Type{Center}) = UnitRange(first(from_idx),   last(from_idx))
restrict_index_on_location(from_idx, ::Type{Face},   ::Type{Center}) = UnitRange(first(from_idx),   last(from_idx)-1)
restrict_index_on_location(from_idx, ::Type{Center}, ::Type{Face})   = UnitRange(first(from_idx)+1, last(from_idx))
