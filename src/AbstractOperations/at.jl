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
    L == location(x) && return x # Don't interpolate unecessarily
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
