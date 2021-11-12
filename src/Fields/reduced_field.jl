using Adapt
using Statistics

import Oceananigans.BoundaryConditions: fill_halo_regions!

#####
##### AbstractReducedField stuff
#####

abstract type AbstractReducedField{X, Y, Z, A, G, T, N} <: AbstractDataField{X, Y, Z, A, G, T, 3} end

const ARF = AbstractReducedField

fill_halo_regions!(field::AbstractReducedField, arch, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...; reduced_dimensions=field.dims)

const DimsType = NTuple{N, Int} where N

function validate_reduced_dims(dims)
    dims = Tuple(dims)

    # Check type
    dims isa DimsType || error("Reduced dims must be an integer or tuple of integers.")

    # Check length
    length(dims) > 3  && error("Models are 3-dimensional. Cannot reduce over 4+ dimensions.")

    # Check values
    all(1 <= d <= 3 for d in dims) || error("Dimensions must be one of 1, 2, 3.")

    return dims
end

function validate_reduced_locations(X, Y, Z, dims)
    loc = (X, Y, Z)

    # Check reduced locations
    for i in dims
        isnothing(loc[i]) || ArgumentError("The location of reduced dimensions must be Nothing")
    end

    return nothing
end

#####
##### Concrete ReducedField
#####

struct ReducedField{X, Y, Z, A, D, G, T, N, B} <: AbstractReducedField{X, Y, Z, A, G, T, N}
                   data :: D
           architecture :: A
                   grid :: G
                   dims :: NTuple{N, Int}
    boundary_conditions :: B

    """
        ReducedField{X, Y, Z}(data, grid, dims)

    Returns a `ReducedField` at location `(X, Y, Z)` with `data` on `grid`
    that is reduced over the dimensions in `dims`.
    """
    function ReducedField{X, Y, Z}(data::D, arch::A, grid::G, dims, bcs::B) where {X, Y, Z, A, D, G, B}

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        N = length(dims)
        T = eltype(grid)

        return new{X, Y, Z, A, D, G, T, N, B}(data, arch, grid, dims, bcs)
    end
end

"""
    ReducedField(X, Y, Z, arch, grid; dims, data=nothing, boundary_conditions=nothing)

Returns a `ReducedField` reduced over `dims` on `grid` and `arch`itecture with `boundary_conditions`.

The location `(X, Y, Z)` may be the parent, three-dimension location or the reduced location.

If `data` is specified, it should be an `OffsetArray` with singleton reduced dimensions;
otherwise `data` is allocated.

If `boundary_conditions` are not provided, default boundary conditions are constructed
using the reduced location.
"""
function ReducedField(FT::DataType, Xr, Yr, Zr, arch, grid::AbstractGrid; dims, data=nothing,
                      boundary_conditions=nothing)

    dims = validate_reduced_dims(dims)

    # Reduce non-reduced dimensions
    X, Y, Z = reduced_location((Xr, Yr, Zr); dims=dims)

    if isnothing(data)
        data = new_data(FT, arch, grid, (X, Y, Z))
    end

    if isnothing(boundary_conditions)
        boundary_conditions = FieldBoundaryConditions(grid, (X, Y, Z))
    end

    return ReducedField{X, Y, Z}(data, arch, grid, dims, boundary_conditions)
end

ReducedField(Xr, Yr, Zr, arch, grid::AbstractGrid; dims, kw...) = ReducedField(eltype(grid), Xr, Yr, Zr, arch, grid; dims, kw...)
ReducedField(Lr::Tuple, arch, grid::AbstractGrid; dims, kw...) = ReducedField(Lr..., arch, grid; dims, kw...)
ReducedField(FT::DataType, Lr::Tuple, arch, grid::AbstractGrid; dims, kw...) = ReducedField(FT, Lr..., arch, grid; dims, kw...)

# Canonical `similar` for AbstractReducedField
Base.similar(r::AbstractReducedField{X, Y, Z, Arch}) where {X, Y, Z, Arch} =
    ReducedField(X, Y, Z, Arch(), r.grid; dims=r.dims, boundary_conditions=r.boundary_conditions)

#####
##### ReducedField utils
#####

reduced_location(loc; dims) = Tuple(i ∈ dims ? Nothing : loc[i] for i in 1:3)

Adapt.adapt_structure(to, reduced_field::ReducedField{X, Y, Z}) where {X, Y, Z} =
    ReducedField{X, Y, Z}(adapt(to, reduced_field.data), nothing, adapt(to, reduced_field.grid), reduced_field.dims, nothing)

#####
##### Field reductions
#####

# Risky to use these without tests. Docs would also be nice.
Statistics.norm(a::AbstractDataField) = sqrt(mapreduce(x -> x * x, +, interior(a)))
Statistics.dot(a::AbstractDataField, b::AbstractDataField) = mapreduce((x, y) -> x * y, +, interior(a), interior(b))

# The more general case, for AbstractOperations
function Statistics.norm(a::AbstractField)
    arch = architecture(a)
    grid = a.grid

    r = zeros(arch, grid, 1)
    
    Base.mapreducedim!(x -> x * x, +, r, a)

    return CUDA.@allowscalar sqrt(r[1])
end

# TODO: In-place allocations with function mappings need to be fixed in Julia Base...
const SumReduction     = typeof(Base.sum!)
const ProdReduction    = typeof(Base.prod!)
const MaximumReduction = typeof(Base.maximum!)
const MinimumReduction = typeof(Base.minimum!)
const AllReduction     = typeof(Base.all!)
const AnyReduction     = typeof(Base.any!)

initialize_reduced_field!(::SumReduction,  f, r::AbstractReducedField, c) = Base.initarray!(interior(r), Base.add_sum, true, interior(c))
initialize_reduced_field!(::ProdReduction, f, r::AbstractReducedField, c) = Base.initarray!(interior(r), Base.mul_prod, true, interior(c))
initialize_reduced_field!(::AllReduction,  f, r::AbstractReducedField, c) = Base.initarray!(interior(r), &, true, interior(c))
initialize_reduced_field!(::AnyReduction,  f, r::AbstractReducedField, c) = Base.initarray!(interior(r), |, true, interior(c))

initialize_reduced_field!(::MaximumReduction, f, r::AbstractReducedField, c) = Base.mapfirst!(f, interior(r), c)
initialize_reduced_field!(::MinimumReduction, f, r::AbstractReducedField, c) = Base.mapfirst!(f, interior(r), c)

filltype(f, grid) = eltype(grid)
filltype(::Union{AllReduction, AnyReduction}, grid) = Bool

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any)

    reduction! = Symbol(reduction, '!')

    @eval begin

        # In-place
        Base.$(reduction!)(f::Function, r::AbstractReducedField, a::AbstractArray; kwargs...) =
            Base.$(reduction!)(f, interior(r), a; kwargs...)

        Base.$(reduction!)(r::AbstractReducedField, a::AbstractArray; kwargs...) =
            Base.$(reduction!)(identity, interior(r), a; kwargs...)

        # Allocating
        function Base.$(reduction)(f::Function, c::AbstractField; dims=:)
            if dims isa Colon
                r = zeros(architecture(c), c.grid, 1, 1, 1)
                Base.$(reduction!)(f, r, c)
                return CUDA.@allowscalar r[1, 1, 1]
            else
                FT = filltype(Base.$(reduction!), c.grid)
                r = ReducedField(FT, location(c), architecture(c), c.grid; dims)
                initialize_reduced_field!(Base.$(reduction!), f, r, c)
                Base.$(reduction!)(f, r, c, init=false)
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; dims=:) = Base.$(reduction)(identity, c; dims)
    end
end

Statistics._mean(f, c::AbstractField, ::Colon) = sum(f, c) / length(c)
