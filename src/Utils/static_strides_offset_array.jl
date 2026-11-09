using Adapt: Adapt
using OffsetArrays: OffsetArrays, AbstractOffsetArray
using Oceananigans.Architectures: Architectures, AbstractSerialArchitecture, on_architecture

struct StaticStridesOffsetArray{T, N, S, O, A <: AbstractArray{T, N}} <: AbstractOffsetArray{T, N}
    parent :: A
end

"""
$(TYPEDSIGNATURES)

Return an array that wraps `parent` with `offsets`, whose sizes and offsets are type parameters, so that the size and
offsets of the array are compile-time constants. Throw an `OverflowError` if the `offsets` overflow the axes of `parent`.
"""
function StaticStridesOffsetArray(parent::AbstractArray{T, N}, offsets::Vararg{Integer, N}) where {T, N}
    map(OffsetArrays.overflow_check, axes(parent), offsets)
    return StaticStridesOffsetArray{T, N, size(parent), Int.(offsets), typeof(parent)}(parent)
end

# collates nested offsets so that `offsets` holds the whole shift and `parent` the underlying array
function StaticStridesOffsetArray(a::AbstractOffsetArray{T, N}, offsets::Vararg{Integer, N}) where {T, N}
    map(OffsetArrays.overflow_check, axes(a), offsets)
    return StaticStridesOffsetArray(parent(a), (OffsetArrays.offsets(a) .+ offsets)...)
end

const SSOA = StaticStridesOffsetArray

# the AbstractOffsetArray interface: parent, offsets and parenttype (IndexStyle and elsize derive from the latter)
Base.parent(a::SSOA) = a.parent
OffsetArrays.offsets(::SSOA{T, N, S, O}) where {T, N, S, O} = O
OffsetArrays.parenttype(::Type{<:SSOA{T, N, S, O, A}}) where {T, N, S, O, A} = A

Base.size(::SSOA{T, N, S}) where {T, N, S} = S
@inline Base.axes(::SSOA{T, N, S, O}) where {T, N, S, O} = map((n, o) -> OffsetArrays.IdOffsetRange(Base.OneTo(n), o), S, O)

rewrap(a::SSOA{T, N, S, O}, p) where {T, N, S, O} = StaticStridesOffsetArray{eltype(p), N, S, O, typeof(p)}(p)

Base.copy(a::SSOA) = rewrap(a, copy(parent(a)))
Base.zero(a::SSOA) = rewrap(a, zero(parent(a)))
Base.map(::Type{T}, a::SSOA) where T = rewrap(a, map(T, parent(a)))
Adapt.adapt_structure(to, a::SSOA) = rewrap(a, Adapt.adapt(to, parent(a)))
Architectures.on_architecture(arch::AbstractSerialArchitecture, a::SSOA) = rewrap(a, on_architecture(arch, parent(a)))

@inline linear_index(::SSOA{T, N, S, O}, I) where {T, N, S, O} = @inbounds LinearIndices(S)[(I .- O)...]
Base.dropdims(a::SSOA; dims) = StaticStridesOffsetArray(dropdims(parent(a); dims), OffsetArrays.offsets(a)[setdiff(1:ndims(a), dims)]...)

# bounds-checked accesses index the parent along each dimension; only `@inbounds` accesses use the static strides
@inline function Base.getindex(a::SSOA{T, N, S, O, <:DenseArray}, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck return parent(a)[(I .- O)...]
    return @inbounds parent(a)[linear_index(a, I)]
end

@inline function Base.setindex!(a::SSOA{T, N, S, O, <:DenseArray}, v, I::Vararg{Int, N}) where {T, N, S, O}
    @boundscheck begin
        parent(a)[(I .- O)...] = v
        return a
    end
    @inbounds parent(a)[linear_index(a, I)] = v
    return a
end
