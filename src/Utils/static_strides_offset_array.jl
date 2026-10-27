using OffsetArrays: OffsetArrays, AbstractOffsetArray

struct StaticStridesOffsetArray{T, N, S, O, A <: AbstractArray{T, N}} <: AbstractOffsetArray{T, N}
    parent :: A
end

"""
$(TYPEDSIGNATURES)

Return an array that wraps `parent` with `offsets`, whose sizes and offsets are type parameters, so that the size and
offsets of the array are compile-time constants.
"""
function StaticStridesOffsetArray(parent::AbstractArray{T, N}, offsets::Vararg{Integer, N}) where {T, N}
    # OffsetArrays does not validate the offsets stored by AbstractOffsetArray subtypes
    map(OffsetArrays.overflow_check, axes(parent), offsets)
    return StaticStridesOffsetArray{T, N, size(parent), Int.(offsets), typeof(parent)}(parent)
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
