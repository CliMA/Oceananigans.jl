using Adapt: Adapt
using OffsetArrays: OffsetArrays, AbstractOffsetArray

"""
    StaticStridesOffsetArray(parent, offsets)

Return an array that wraps `parent` with `offsets`, whose sizes and offsets are type parameters, so that the strides and
offsets of the array are compile-time constants.
"""
struct StaticStridesOffsetArray{T, N, S, O, A <: AbstractArray{T, N}} <: AbstractOffsetArray{T, N}
    parent :: A
end

function StaticStridesOffsetArray(parent::AbstractArray{T, N}, offsets::Vararg{Integer, N}) where {T, N}
    return StaticStridesOffsetArray{T, N, size(parent), Int.(offsets), typeof(parent)}(parent)
end

const SSOA{T, N, S, O, A} = StaticStridesOffsetArray{T, N, S, O, A}

Base.size(::SSOA{T, N, S}) where {T, N, S} = S
Base.IndexStyle(::Type{<:SSOA{T, N, S, O, A}}) where {T, N, S, O, A} = IndexStyle(A)
OffsetArrays.offsets(::SSOA{T, N, S, O}) where {T, N, S, O} = O
OffsetArrays.unwrap(a::SSOA{T, N, S, O}) where {T, N, S, O} = parent(a), p -> StaticStridesOffsetArray{eltype(p), N, S, O, typeof(p)}(p)
@inline linear_index(::SSOA{T, N, S, O}, I) where {T, N, S, O} = @inbounds LinearIndices(S)[(I .- O)...]

@inline function Base.getindex(a::SSOA{T, N, S, O, <:DenseArray}, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    return @inbounds parent(a)[linear_index(a, I)]
end

@inline function Base.setindex!(a::SSOA{T, N, S, O, <:DenseArray}, v, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    @inbounds parent(a)[linear_index(a, I)] = v
    return a
end
