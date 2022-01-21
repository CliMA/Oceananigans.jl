using Oceananigans.Fields: AbstractField, ReducedField
using CUDA: @allowscalar
import Statistics: norm

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

struct MaskedObject{LX, LY, LZ, O, G, M, T} <: AbstractArray{T, 3} 
   obj  :: O
   grid :: G
   mask :: M

    function MaskedObject{LX, LY, LZ}(obj::O, grid::G, mask::M) where {LX, LY, LZ, O, G, M}
        T = eltype(obj)
        return new{LX, LY, LZ, O, G, M, T}(obj, grid, mask)
    end
end

@inline masked_object(LX, LY, LZ, obj, grid, mask) = MaskedObject{LX, LY, LZ}(obj, grid, mask)
@inline masked_object(obj::ImmersedField, mask)    = masked_object(location(obj)..., obj, obj.grid, mask)

# One-directional masking


Adapt.adapt_structure(to, mo::MaskedObject{LX, LY, LZ}) where {LX, LY, LZ} =
            MaskedObject{LX, LY, LZ}(adapt(to, mo.obj), adapt(to, mo.grid), adapt(to, mo.mask))

@inline Base.size(mo::MaskedObject) = Base.size(mo.obj)

@inline function Base.getindex(mo::MaskedObject{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    return ifelse(solid_interface(LX(), LY(), LZ(), i, j, k, mo.grid), 
                  mo.mask, 
                  getindex(mo.obj, i, j, k))
end

# Allocating and in-place reductions with masking
for (reduction, mask) in zip((:sum, :maximum, :minimum, :all, :any, :prod), (0, -Inf, Inf, 1, 1, 1))

    reduction! = Symbol(reduction, '!')

    @eval begin
        # In-place
        Base.$(reduction!)(f::Function, r::ReducedField, a::ImmersedField; kwargs...) = 
            Base.$(reduction!)(f, interior(r), masked_object(a, $mask); kwargs...)
        Base.$(reduction!)(r::ReducedField, a::ImmersedField; kwargs...) =
            Base.$(reduction!)(identity, interior(r), masked_object(a, $mask); kwargs...)

        # Allocating
        Base.$(reduction)(f::Function, c::ImmersedField; kwargs...) = Base.$(reduction)(f, masked_object(c, $mask); kwargs...)
        Base.$(reduction)(c::ImmersedField; kwargs...) = Base.$(reduction)(masked_object(c, $mask); kwargs...)
    end
end

@inline immersed_length(c::ImmersedField) = sum(masked_object(c / c, 0))

Statistics.dot(a::ImmersedField, b::Field) = Statistics.dot(masked_object(a, 0), b)
Statistics.dot(a::Field, b::ImmersedField) = Statistics.dot(masked_object(a, 0), b)

function Statistics.norm(c::ImmersedField)
    r = zeros(c.grid, 1)
    Base.mapreducedim!(x -> x * x, +, r, masked_object(c, 0))
    return @allowscalar sqrt(r[1])
end

Statistics._mean(f, c::ImmersedField, ::Colon) = sum(f, c) / immersed_length(c)

function Statistics._mean(f, c::ImmersedField; dims)
    r = sum(f, c; dims = dims)
    n = sum(f, masked_object(c / c, 0); dims = dims)

    @show r, n
    return r ./ n
end

Statistics._mean(c::ImmersedField; dims) = Statistics._mean(identity, c, dims=dims)