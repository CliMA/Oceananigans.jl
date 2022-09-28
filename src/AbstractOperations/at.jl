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

"""
    utility to propagate field indices in abstract operations
"""
# Numbers and functions do not have indices
indices(f::Function) = (:, :, :)
indices(f::Number)   = (:, :, :)

# easy index propagation 
function interpolate_indices(args, loc_op)
    idxs = Any[:, :, :]
    for i in 1:3
        for arg in args
            idxs[i] = interpolate_index(indices(arg)[i], idxs[i], location(arg)[i], loc_op[i])
        end
    end

    return Tuple(idxs)
end

interpolate_index(::Colon, ::Colon, args...)       = Colon()
interpolate_index(::Colon, b::UnitRange, args...)  = b

# If we interpolate from a `Center` to a `Face` we lose the first index,
# otherwise we lose the last index
# REMEMBER! Not supported abstract operations which require an interpolation of sliced fields!
function interpolate_index(a::UnitRange, ::Colon, loc, new_loc)  
    if loc == new_loc
        return a
    else
        if a[1] == a[2]
            throw(ArgumentError("Cannot interpolate from $loc to $new_loc a Sliced field!"))
        end
        if new_loc == Face
            return UnitRange(a[1]+1, a[2])
        else
            return UnitRange(a[1], a[2]-1)    
        end
    end
end

# REMEMBER: issue an error when the indices are not compatible (e.g. parallel fields on different planes)
function interpolate_index(a::UnitRange, b::UnitRange, loc, new_loc)   
    if loc == new_loc
        return UnitRange(max(a[1], b[1]), min(a[2], b[2]))
    else
        if a[1] == a[2]
            throw(ArgumentError("Cannot interpolate from $loc to $new_loc a Sliced field!"))
        end
        if new_loc == Face
            return UnitRange(max(a[1]+1, b[1]),min(a[2], b[2]))
        else
            return UnitRange(max(a[1], b[1]),min(a[2]-1, b[2]))
        end
    end
end