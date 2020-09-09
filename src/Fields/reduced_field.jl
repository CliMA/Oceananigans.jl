#####
##### AbstractReducedField stuff
#####

abstract type AbstractReducedField{X, Y, Z, A, G, N} <: AbstractField{X, Y, Z, A, G} end

const ARF = AbstractReducedField

@propagate_inbounds Base.getindex(r::ARF{Nothing, Y, Z},     i, j, k) where {Y, Z} = r.data[1, j, k]
@propagate_inbounds Base.setindex!(r::ARF{Nothing, Y, Z}, d, i, j, k) where {Y, Z} = r.data[1, j, k] = d

@propagate_inbounds Base.getindex(r::ARF{X, Nothing, Z},     i, j, k) where {X, Z} = r.data[i, 1, k]
@propagate_inbounds Base.setindex!(r::ARF{X, Nothing, Z}, d, i, j, k) where {X, Z} = r.data[i, 1, k] = d

@propagate_inbounds Base.getindex(r::ARF{X, Y, Nothing},     i, j, k) where {X, Y} = r.data[i, j, 1]
@propagate_inbounds Base.setindex!(r::ARF{X, Y, Nothing}, d, i, j, k) where {X, Y} = r.data[i, j, 1] = d

@propagate_inbounds Base.getindex(r::ARF{X, Nothing, Nothing},     i, j, k) where X = r.data[i, 1, 1]
@propagate_inbounds Base.setindex!(r::ARF{X, Nothing, Nothing}, d, i, j, k) where X = r.data[i, 1, 1] = d

@propagate_inbounds Base.getindex(r::ARF{Nothing, Y, Nothing},     i, j, k) where Y = r.data[1, j, 1]
@propagate_inbounds Base.setindex!(r::ARF{Nothing, Y, Nothing}, d, i, j, k) where Y = r.data[1, j, 1] = d

@propagate_inbounds Base.getindex(r::ARF{Nothing, Nothing, Z},     i, j, k) where Z = r.data[1, 1, k]
@propagate_inbounds Base.setindex!(r::ARF{Nothing, Nothing, Z}, d, i, j, k) where Z = r.data[1, 1, k] = d

@propagate_inbounds Base.getindex(r::ARF{Nothing, Nothing, Nothing},     i, j, k) = r.data[1, 1, 1]
@propagate_inbounds Base.setindex!(r::ARF{Nothing, Nothing, Nothing}, d, i, j, k) = r.data[1, 1, 1] = d

const DimsType = NTuple{N, Int} where N

function validate_reduced_dims(dims)
    dims = Tuple(dims)

    # Check type
    dims isa DimsType || error("Reduced dims must be an integer or tuple of integers.")
    
    # Check length
    length(dims) > 3  && error("Models are 3-dimensional. Cannot average over 4+ dimensions.")

    # Check values
    all(1 <= d <= 3 for d in dims) || error("Dimensions must be one of 1, 2, 3.")

    return dims
end

function validate_reduced_locations(X, Y, Z, dims)
    loc = (X, Y, Z)

    # Check reduced locations
    for i in dims
        loc[i] != Nothing && ArgumentError("The location of reduced dimensions must be Nothing")
    end

    return nothing
end

#####
##### Concrete ReducedField
#####

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

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        return new{X, Y, Z, typeof(data), typeof(grid), length(dims)}(data, grid, dims)
    end
end

"""
    ReducedField(X, Y, Z, arch, grid; dims, data=nothing)

Returns a `ReducedField` reduced over `dims` on `grid` and `arch`itecture.
The location `(X, Y, Z)` may be the parent, three-dimension location or the reduced location.
If `data` is specified, it should be an `OffsetArray` with singleton reduced dimensions;
otherwise `data` is allocated.
"""
function ReducedField(Xr, Yr, Zr, arch, grid; dims, data=nothing)

    dims = validate_reduced_dims(dims)

    # Reduce non-reduced dimensions
    X, Y, Z = reduced_location((Xr, Yr, Zr); dims=dims)

    if isnothing(data)
        data = new_data(arch, grid, (X, Y, Z))
    end

    return ReducedField{X, Y, Z}(data, grid, dims)
end

ReducedField(loc::Tuple, args...; kwargs...) = ReducedField(loc..., args...; kwargs...)

#####
##### ReducedField utils
#####

reduced_location(loc; dims) = Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)
