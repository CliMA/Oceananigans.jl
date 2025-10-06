using Oceananigans
using Oceananigans.Fields: default_indices, compute_index_intersection

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

# Numbers and functions do not have index restrictions
indices(::Function) = default_indices(3)
indices(::Number)   = default_indices(3)

"""
    intersect_indices(loc, operands...)

Utility to compute the intersection of `operands' indices.
"""
function intersect_indices(loc, operands...)

    idx1 = compute_operand_intersection(Colon(), loc[1], operands...; dim=1)
    idx2 = compute_operand_intersection(Colon(), loc[2], operands...; dim=2)
    idx3 = compute_operand_intersection(Colon(), loc[3], operands...; dim=3)

    return (idx1, idx2, idx3)
end

# Fallback for `KernelFunctionOperation`s with no argument
compute_operand_intersection(::Colon, to_loc; kw...) = Colon()

compute_operand_intersection(to_idx, to_loc, op; dim) =
    compute_index_intersection(to_idx, indices(op)[dim],
                               to_loc, location(op, dim))

"""Compute index intersection recursively for `dim`ension ∈ (1, 2, 3)."""
function compute_operand_intersection(to_idx, to_loc, op1, op2, more_ops...; dim)
    new_to_idx = compute_index_intersection(to_idx, indices(op1)[dim], to_loc, location(op1, dim))
    return compute_operand_intersection(new_to_idx, to_loc, op2, more_ops...; dim)
end