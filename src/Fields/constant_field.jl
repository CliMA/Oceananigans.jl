using Oceananigans.Utils: prettysummary

struct ZeroField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end
struct OneField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end

ZeroField(T=Int) = ZeroField{T, 3}() # default 3D, integer 0
OneField(T=Int) = OneField{T, 3}() # default 3D, integer 1

@inline Base.getindex(::ZeroField{T, N}, ind...) where {N, T} = zero(T)
@inline Base.getindex(::OneField{T, N}, ind...) where {N, T} = one(T)

@inline Base.summary(::ZeroField{T}) where T = string("ZeroField with type $T")
@inline Base.summary(::OneField{T})  where T = string("OneField with type $T")

Base.show(io::IO, f::ZeroField{T}) where T = print(io, summary(f))
Base.show(io::IO,  f::OneField{T}) where T = print(io, summary(f))

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
 
