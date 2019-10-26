@inline identity(i, j, k, grid, c) = @inbounds c[i, j, k]
@inline identity(i, j, k, grid, a::Number) = a

"""Evaluate the function `F` with signature `F(i, j, k, grid, args...)` at index `i, j, k` without 
interpolation."""
@inline identity(i, j, k, grid, F::TF, args...) where TF<:Function = F(i, j, k, grid, args...)

# Utilities for inferring the interpolation function needed to 
# interpolate a field from one location to the next.
interpolation_code(::Type{Face}) = :f
interpolation_code(::Type{Cell}) = :c
interpolation_code(::Face) = :f
interpolation_code(::Cell) = :c
interpolation_code(from::L, to::L) where L = :a
interpolation_code(from, to) = interpolation_code(to)

for ξ in ("x", "y", "z")
    ▶sym = Symbol(:▶, ξ, :sym)
    @eval begin
        $▶sym(s::Symbol) = $▶sym(Val(s))
        $▶sym(::Union{Val{:f}, Val{:c}}) = $ξ
        $▶sym(::Val{:a}) = ""
    end
end

instantiate(L::NTuple{N, <:DataType}) where N = Tuple(X() for X in L)
instantiate(x, y...) = instantiate(tuple(x, y...))

"""
    interpolation_operator(from, to)

Returns the function to interpolate a field `from = (XA, YZ, ZA)`, `to = (XB, YB, ZB)`, 
where the `XA`s and `XB`s are types `Face` or `Cell` (rather than type instances).
"""
interpolation_operator(from::NTuple{N, <:DataType}, to::NTuple{N, <:DataType}) where N =
    interpolation_operator(instantiate(from), instantiate(to))

"""
    interpolation_operator(from, to)

Returns the function to interpolate a field `from = (XA, YZ, ZA)`, `to = (XB, YB, ZB)`, 
where the `XA`s and `XB`s are `Face()` or `Cell()` instances.
"""
function interpolation_operator(from, to)
    x, y, z = (interpolation_code(X, Y) for (X, Y) in zip(from, to))

    if all(ξ === :a for ξ in (x, y, z))
        return identity
    else 
        return eval(Symbol(:▶, ▶xsym(x), ▶ysym(y), ▶zsym(z), :_, x, y, z))
    end
end

"""
    interpolation_operator(::Nothing, to)

Return the `identity` interpolator function. This is needed to obtain the interpolation 
opertator for fields that have no instrinsic location, like numbers or functions.
"""
interpolation_operator(::Nothing, to) = identity

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

"""
    @at location abstract_operation

Modify the `abstract_operation` so that it returns values at
`location`, where `location` is a 3-tuple of `Face`s and `Cell`s.
"""
macro at(location, abstract_operation)
    insert_location!(abstract_operation, location)
    return esc(abstract_operation)
end
