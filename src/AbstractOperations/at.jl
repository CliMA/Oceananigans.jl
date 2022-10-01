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

# Fallback (used by KernelFunctionOperation)
indices(f) = default_indices(3)


"""
    interpolate_indices(operands..., loc_operation = abstract_operation_location)

Utility to propagate operands' indices in `AbstractOperations`s with multiple operands 
(`BinaryOperation`s and `MultiaryOperation`s).
"""
function interpolate_indices(args...; loc_operation = (Center, Center, Center))
    idxs = Any[:, :, :]
    for i in 1:3
        for arg in args
            idxs[i] = interpolate_index(indices(arg)[i], idxs[i], location(arg)[i], loc_operation[i])
        end
    end

    return Tuple(idxs)
end

interpolate_index(::Colon, ::Colon, args...)       = Colon()
interpolate_index(::Colon, b::UnitRange, args...)  = b

function interpolate_index(a::UnitRange, ::Colon, loc, new_loc)  
    a = corrected_index(a, loc, new_loc)

    # Abstract operations that require an interpolation of a sliced fields are not supported!
    first(a) > last(a) && throw(ArgumentError("Cannot interpolate a sliced field from $loc to $(new_loc)!"))
    return a
end

function interpolate_index(a::UnitRange, b::UnitRange, loc, new_loc)   
    a = corrected_index(a, loc, new_loc)

    # Abstract operations that require an interpolation of a sliced fields are not supported!
    first(a) > last(a) && throw(ArgumentError("Cannot interpolate a sliced field from $loc to $(new_loc)!"))
    
    indices = UnitRange(max(first(a), first(b)), min(last(a), last(b)))
    
    # Abstract operations between parallel non-intersecating windowed fields are not supported
    first(indices) > last(indices) && throw(ArgumentError("BinaryOperation operand indices $(a) and $(b) do not intersect!"))
    return indices
end

# Windowed fields interpolated from `Center`s to `Face`s lose the first index.
# Viceverse, windowed fields interpolated from `Face`s to `Center`s lose the last index
corrected_index(a, ::Type{Face},   ::Type{Face})   = UnitRange(first(a),   last(a))
corrected_index(a, ::Type{Center}, ::Type{Center}) = UnitRange(first(a),   last(a))
corrected_index(a, ::Type{Face},   ::Type{Center}) = UnitRange(first(a),   last(a)-1)
corrected_index(a, ::Type{Center}, ::Type{Face})   = UnitRange(first(a)+1, last(a))
