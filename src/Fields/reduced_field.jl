abstract type AbstractReducedField{X, Y, Z, A, G, N} <: AbstractField{X, Y, Z, A, G}

"""
    struct ReducedField{X, Y, Z, A, G, N} <: AbstractField{X, Y, Z, A, G}

Representation of a field at the location `(X, Y, Z)` with data of type `A`
on a grid of type `G` that is 'reduced' on `N` dimensions.
"""
struct ReducedField{X, Y, Z, A, G, N} <: AbstractReducedField{X, Y, Z, A, G, N}
    data :: A
    grid :: G
    dims :: NTuple{N, Int}

    """
        ReducedField{X, Y, Z}(data, grid, dims)

    Returns a `ReducedField` at location `(X, Y, Z)` with `data` on `grid`
    that is reduced over the dimensions in `dims`.
    """
    function ReducedField{X, Y, Z}(data, grid, dims) where {X, Y, Z}
        L = (X, Y, Z)

        for i = 1:3
            L[i] != Nothing && ArgumentError("The location of reduced dimensions must be
                                             Nothing")
        end

        Tx, Ty, Tz = total_size((X, Y, Z), grid)

        if size(data) != (Tx, Ty, Tz)
            e = "Cannot construct field at ($X, $Y, $Z) with size(data)=$(size(data)). " *
                "`data` must have size ($Tx, $Ty, $Tz)."
            throw(ArgumentError(e))
        end

        return new{X, Y, Z, typeof(data), typeof(grid), typeof(dims)}(data, grid, dims)
    end
end

function ReducedField(reduced_loc, data, grid; dims)
    dims = tuple(dims)
    X, Y, Z = Tuple(i ∈ dims ? reduced_loc[j] : Nothing for (i, j) in enumerate(dims))
    return ReducedField{X, Y, Z}(data, grid, dims)
end

@propagate_inbounds getindex(r::AbstractReducedField{Nothing, Y, Z}, i, j, k) where {Y, Z} = r.data[1, j, k]
@propagate_inbounds getindex(r::AbstractReducedField{X, Nothing, Z}, i, j, k) where {X, Z} = r.data[i, 1, k]
@propagate_inbounds getindex(r::AbstractReducedField{X, Y, Nothing}, i, j, k) where {X, Y} = r.data[i, j, 1]

@propagate_inbounds getindex(r::AbstractReducedField{X, Nothing, Nothing}, i, j, k) where X = r.data[i, 1, 1]
@propagate_inbounds getindex(r::AbstractReducedField{Nothing, Y, Nothing}, i, j, k) where Y = r.data[1, j, 1]
@propagate_inbounds getindex(r::AbstractReducedField{Nothing, Nothing, Z}, i, j, k) where Z = r.data[1, 1, k]
