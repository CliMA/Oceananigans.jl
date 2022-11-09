"""
    insert_location(ex::Expr, location)

Insert a symbolic representation of `location` into the arguments of an `expression`.

Used in the `@at` macro for specifying the location of an `AbstractOperation`.
"""
function insert_location!(ex::Expr, location)
    if ex.head === :call && ex.args[1] ∈ operators
        push!(ex.args, ex.args[end])
        ex.args[3:end-1] .= ex.args[2:end-2]
        ex.args[2] = location
    end

    for arg in ex.args
        insert_location!(arg, location)
    end

    return nothing
end

"Fallback for when `insert_location` is called on objects other than expressions."
insert_location!(anything, location) = nothing

# A very special UnaryOperation
@inline interpolate_identity(x) = x
@unary interpolate_identity

interpolate_operation(L, x) = x

function interpolate_operation(L, x::AbstractField)
    L == location(x) && return x # Don't interpolate unnecessarily
    return interpolate_identity(L, x)
end

"""
    @at location abstract_operation

Modify the `abstract_operation` so that it returns values at
`location`, where `location` is a 3-tuple of `Face`s and `Center`s.
"""
macro at(location, abstract_operation)
    insert_location!(abstract_operation, location)

    # We wrap it all in an interpolator to help "stubborn" binary operations
    # arrive in the right place.
    wrapped_operation = quote
        interpolate_operation($(esc(location)), $(esc(abstract_operation)))
    end

    return wrapped_operation
end

using Oceananigans.Fields: default_indices

# Numbers and functions do not have index restrictions
indices(f::Function) = default_indices(3)
indices(f::Number)   = default_indices(3)

"""
    intersect_indices(loc, operands...)

Utility to compute the intersection of `operands' indices.
"""
function intersect_indices(loc, operands...)

    idx1 = compute_index_intersection(Colon(), loc[1], operands...; dim=1)
    idx2 = compute_index_intersection(Colon(), loc[2], operands...; dim=2)
    idx3 = compute_index_intersection(Colon(), loc[3], operands...; dim=3)
            
    return (idx1, idx2, idx3)
end

compute_index_intersection(to_idx, to_loc, op; dim) =
    _compute_index_intersection(to_idx,
                                indices(op, dim),
                                to_loc,
                                location(op, dim))

"""Compute index intersection recursively for `dim`ension ∈ (1, 2, 3)."""
function compute_index_intersection(to_idx, to_loc, op1, op2, more_ops...; dim)
    new_to_idx = _compute_index_intersection(to_idx, indices(op1, dim), to_loc, location(op1, dim))
    return compute_index_intersection(new_to_idx, to_loc, op2, more_ops...; dim)
end

# Life is pretty simple in this case.
_compute_index_intersection(to_idx::Colon, from_idx::Colon, args...) = Colon()

# Because `from_idx` imposes no restrictions, we just return `to_idx`.
_compute_index_intersection(to_idx::UnitRange, from_idx::Colon, args...) = to_idx

# This time we account for the possible range-reducing effect of interpolation on `from_idx`.
function _compute_index_intersection(to_idx::Colon, from_idx::UnitRange, to_loc, from_loc)
    shifted_idx = restrict_index_for_interpolation(from_idx, from_loc, to_loc)
    validate_shifted_index(shifted_idx)
    return shifted_idx
end

# Compute the intersection of two index ranges
function _compute_index_intersection(to_idx::UnitRange, from_idx::UnitRange, to_loc, from_loc)
    shifted_idx = restrict_index_for_interpolation(from_idx, from_loc, to_loc)
    validate_shifted_index(shifted_idx)
    
    range_intersection = UnitRange(max(first(shifted_idx), first(to_idx)), min(last(shifted_idx), last(to_idx)))
    
    # Check validity of the intersection index range
    first(range_intersection) > last(range_intersection) &&
        throw(ArgumentError("Indices $(from_idx) and $(to_idx) interpolated from $(from_loc) to $(to_loc) do not intersect!"))

    return range_intersection
end

validate_shifted_index(shifted_idx) = first(shifted_idx) > last(shifted_idx) &&
    throw(ArgumentError("Cannot compute index intersection for indices $(from_idx) interpolating from $(from_loc) to $(to_loc)!"))

"""
    restrict_index_for_interpolation(from_idx, from_loc, to_loc)

Return a "restricted" index range for the result of interpolating from
`from_loc` to `to_loc`, over the index range `from_idx`:

    * Windowed fields interpolated from `Center`s to `Face`s lose the first index.
    * Conversely, windowed fields interpolated from `Face`s to `Center`s lose the last index
"""
restrict_index_for_interpolation(from_idx, ::Type{Face},   ::Type{Face})   = UnitRange(first(from_idx),   last(from_idx))
restrict_index_for_interpolation(from_idx, ::Type{Center}, ::Type{Center}) = UnitRange(first(from_idx),   last(from_idx))
restrict_index_for_interpolation(from_idx, ::Type{Face},   ::Type{Center}) = UnitRange(first(from_idx),   last(from_idx)-1)
restrict_index_for_interpolation(from_idx, ::Type{Center}, ::Type{Face})   = UnitRange(first(from_idx)+1, last(from_idx))
