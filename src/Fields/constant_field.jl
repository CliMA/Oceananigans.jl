struct ZeroField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end
struct OneField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N} end

ZeroField(T=Int) = ZeroField{T, 3}() # default 3D, integer 0
OneField(T=Int) = OneField{T, 3}() # default 3D, integer 0

@inline Base.getindex(::ZeroField{T, N}, ind...) where {N, T} = zero(T)
@inline Base.getindex(::OneField{T, N}, ind...) where {N, T} = one(T)

struct ConstantField{T, N} <: AbstractField{Nothing, Nothing, Nothing, Nothing, T, N}
    constant :: T
    ConstantField{N}(constant::T) where T = new{T, N}(constant)
end

# Default 3-dimensional field
ConstantField(constant) = ConstantField{3}(constant)

@inline Base.getindex(f::ConstantField, ind...) = f.constant

