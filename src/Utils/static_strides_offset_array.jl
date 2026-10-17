using OffsetArrays: OffsetArrays, AbstractOffsetArray

"""
    StaticStridesOffsetArray(parent, offsets)

Return an array that wraps `parent` with `offsets`. The offsets and the sizes of all dimensions but the last are type
parameters, so that every index and stride of the array is a compile-time constant.
"""
struct StaticStridesOffsetArray{T, N, S, O, A <: AbstractArray{T, N}} <: AbstractOffsetArray{T, N}
    parent :: A
end

function StaticStridesOffsetArray(parent::AbstractArray{T, N}, offsets::Vararg{Integer, N}) where {T, N}
    S = Base.front(size(parent))
    O = Int.(offsets)
    return StaticStridesOffsetArray{T, N, S, O, typeof(parent)}(parent)
end

Base.size(a::StaticStridesOffsetArray{T, N, S}) where {T, N, S} = (S..., size(parent(a), N))

OffsetArrays.offsets(::StaticStridesOffsetArray{T, N, S, O}) where {T, N, S, O} = O
OffsetArrays.unwrap(a::StaticStridesOffsetArray{T, N, S, O}) where {T, N, S, O} = parent(a), p -> StaticStridesOffsetArray{eltype(p), N, S, O, typeof(p)}(p)

@inline function linear_index(::StaticStridesOffsetArray{T, N, S, O}, I) where {T, N, S, O}
    strides = (1, cumprod(S)...)
    return 1 + sum(map((i, o, s) -> (i - o - 1) * s, I, O, strides))
end

@inline function Base.getindex(a::StaticStridesOffsetArray{T, N, S, O, <:DenseArray}, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    return @inbounds parent(a)[linear_index(a, I)]
end

@inline function Base.setindex!(a::StaticStridesOffsetArray{T, N, S, O, <:DenseArray}, v, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    @inbounds parent(a)[linear_index(a, I)] = v
    return a
end
