using Oceananigans.Utils: prettysummary

struct ZeroField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end
struct OneField{T, N}  <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end

ZeroField(T=Int) = ZeroField{T, 3}() # default 3D, integer 0
OneField(T=Int) = OneField{T, 3}() # default 3D, integer 1

@inline Base.getindex(::ZeroField{T, N},   ind...) where {N, T} = zero(T)
@inline Base.getindex(::OneField{T, N},    ind...) where {N, T} = one(T)

struct ConstantField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N}
    constant :: T
    ConstantField{N}(constant::T) where {T, N} = new{T, N}(constant)
end

# Default 3-dimensional field
ConstantField(constant) = ConstantField{3}(constant)

@inline Base.getindex(f::ConstantField, ind...) = f.constant

const CF = Union{ConstantField, ZeroField, OneField}

@inline Base.summary(f::CF) = string("ConstantField(", prettysummary(f.constant), ")")
Base.show(io::IO, f::CF) = print(io, summary(f))

###
### Convenience to investigate the 
###

struct OneFieldGridded{LX, LY, LZ, G, I, T} <: AbstractField{LX, LY, LZ, G, T, 3}
    grid :: G
    indices :: I

    function OneFieldGridded{T}(loc, grid::G, indices::I) where {G, I, T}
        return new{loc..., G, I, T}(grid, indices)
    end
end

OneFieldGridded(loc, grid, T=Int; indices = default_indices(loc, 3)) = OneFieldGridded{T}(loc, grid, indices)
indices(o::OneFieldGridded)                = o.indices

@inline Base.getindex(of::OneFieldGridded{LX, LY, LZ, G, I, T}, ind...) where {LX, LY, LZ, G, I, T} = one(T)

function Base.axes(f::OneFieldGridded)
    if f.indices === (:, : ,:)
        return Base.OneTo.(size(f))
    else
        return Tuple(f.indices[i] isa Colon ? Base.OneTo(size(f, i)) : 
                     f.indices[i] isa Integer ? range(f.indices[i],f.indices[i]) : 
                     f.indices[i] for i = 1:3)
    end
end

const XReducedOneField = OneFieldGridded{Nothing}
const YReducedOneField = OneFieldGridded{<:Any, Nothing}
const ZReducedOneField = OneFieldGridded{<:Any, <:Any, Nothing}

const YZReducedOneField = OneFieldGridded{<:Any, Nothing, Nothing}
const XZReducedOneField = OneFieldGridded{Nothing, <:Any, Nothing}
const XYReducedOneField = OneFieldGridded{Nothing, Nothing, <:Any}

const XYZReducedOneField = OneFieldGridded{Nothing, Nothing, Nothing}

@inline reduced_indices(i, j, k, r::XReducedOneField) = @inbounds (indices(r)[1], j, k)
@inline reduced_indices(i, j, k, r::YReducedOneField) = @inbounds (i, indices(r)[2], k)
@inline reduced_indices(i, j, k, r::ZReducedOneField) = @inbounds (i, j, indices(r)[3])

@inline reduced_indices(i, j, k, r::YZReducedOneField) = @inbounds (i, indices(r)[2], indices(r)[3])
@inline reduced_indices(i, j, k, r::XZReducedOneField) = @inbounds (indices(r)[1], j, indices(r)[3])
@inline reduced_indices(i, j, k, r::XYReducedOneField) = @inbounds (indices(r)[1], indices(r)[2], k)

@inline reduced_indices(i, j, k, r::XYZReducedOneField) = @inbounds indices(r)