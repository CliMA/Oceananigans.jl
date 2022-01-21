using Oceananigans.Fields: AbstractField, ReducedField
using Oceananigans.AbstractOperations: AbstractOperation
using CUDA: @allowscalar
import Statistics: norm
import Oceananigans.Fields: masked_object
import Oceananigans.AbstractOperations: operation_name

# ###
# ###  reduction operations involving immersed boundary grids exclude the immersed region 
# ### (we exclude also the values on the faces of the immersed boundary with `solid_interface`)
# ###

const ImmersedField = AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}

struct MaskedObject{LX, LY, LZ, O, G, M, C, T} <: AbstractOperation{LX, LY, LZ, G, T} 
   obj  :: O
   grid :: G
   mask :: M
   condition :: C

    function MaskedObject{LX, LY, LZ}(obj::O, grid::G, mask::M, condition::C) where {LX, LY, LZ, O, G, M, C}
        T = eltype(obj)
        return new{LX, LY, LZ, O, G, M, C, T}(obj, grid, mask, condition)
    end
end

@inline function masked_object(obj::ImmersedField, mask) 
    return masked_object(location(obj)..., obj, obj.grid, mask, immersed_condition)
end

@inline immersed_condition(i, j, k, mo::MaskedObject{LX, LY, LZ}) = solid_interface(LX(), LY(), LZ(), i, j, k, mo.grid) 

@inline masked_object(LX, LY, LZ, obj, grid, mask, condition) = MaskedObject{LX, LY, LZ}(obj, grid, mask, condition)

operation_name(mo::MaskedObject) = "Masked field"

Adapt.adapt_structure(to, mo::MaskedObject{LX, LY, LZ}) where {LX, LY, LZ} =
            MaskedObject{LX, LY, LZ}(adapt(to, mo.obj), adapt(to, mo.grid), adapt(to, mo.mask), adapt(to, mo.condition))

@inline Base.size(mo::MaskedObject) = Base.size(mo.obj)

@inline function Base.getindex(mo::MaskedObject{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    return ifelse(condition(i, j, k, mo), 
                  get_mask(mo.mask, i, j, k), 
                  getindex(mo.obj, i, j, k))
end

@inline get_mask(mask::Number       , i, j, k) = mask
@inline get_mask(mask::AbstractArray, i, j, k) = mask[i, j, k]
@inline get_mask(mask::Base.Callable, i, j, k) = mask(i, j, k)

@inline immersed_length(c::AbstractField) = length(c)
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