using OffsetArrays: OffsetArrays, AbstractOffsetArray, OffsetArray

struct StaticStridesOffsetArray{T, N, S, O, A <: AbstractArray{T, N}} <: AbstractOffsetArray{T, N}
    parent :: A
end

"""
$(TYPEDSIGNATURES)

Return an array that wraps `parent` with offset `indices`, whose sizes and offsets are type parameters, so that the
size and offsets of the array are compile-time constants. The `indices` take any form accepted by `OffsetArray`, which
also throws an `OverflowError` if they overflow the axes of `parent`.
"""
StaticStridesOffsetArray(parent::AbstractArray, indices...) = StaticStridesOffsetArray(OffsetArray(parent, indices...))

function StaticStridesOffsetArray(a::OffsetArray{T, N}) where {T, N}
    p = parent(a)
    return StaticStridesOffsetArray{T, N, size(p), OffsetArrays.offsets(a), typeof(p)}(p)
end

const SSOA = StaticStridesOffsetArray

# the AbstractOffsetArray interface: parent, offsets and parenttype (IndexStyle and elsize derive from the latter)
Base.parent(a::SSOA) = a.parent
OffsetArrays.offsets(::SSOA{T, N, S, O}) where {T, N, S, O} = O
OffsetArrays.parenttype(::Type{<:SSOA{T, N, S, O, A}}) where {T, N, S, O, A} = A

Base.size(::SSOA{T, N, S}) where {T, N, S} = S
OffsetArrays.unwrap(a::SSOA{T, N, S, O}) where {T, N, S, O} = parent(a), p -> StaticStridesOffsetArray{eltype(p), N, S, O, typeof(p)}(p)
@inline linear_index(::SSOA{T, N, S, O}, I) where {T, N, S, O} = @inbounds LinearIndices(S)[(I .- O)...]
Base.dropdims(a::SSOA; dims) = StaticStridesOffsetArray(dropdims(parent(a); dims), OffsetArrays.offsets(a)[setdiff(1:ndims(a), dims)]...)

@inline function Base.getindex(a::SSOA{T, N, S, O, <:DenseArray}, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    return @inbounds parent(a)[linear_index(a, I)]
end

@inline function Base.setindex!(a::SSOA{T, N, S, O, <:DenseArray}, v, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck checkbounds(a, I...)
    @inbounds parent(a)[linear_index(a, I)] = v
    return a
end
