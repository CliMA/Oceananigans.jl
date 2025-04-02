struct HalolessArray{LX, LY, LZ, N, T, D, P} <: AbstractArray{T, N}
    data :: D
    topology :: P
end

Base.size(hla::HalolessArray) = size(hla.data)

function HalolessArray{LX, LY, LZ}(data::Array, topo) where {LX, LY, LZ}
    D = typeof(data)
    T = eltype(data)
    P = typeof(topo)
    N = ndims(data)
    return HalolessArray{LX, LY, LZ, N, T, D, P}(data, topo)
end

@inline function mangle_index(i, ::Periodic, ℓ, N)
    return if i < 1
        N - i
    elseif i > N
        i - N
    else
        i
    end
end

@inline function mangle_index(i, ::Bounded, ::Center, N)
    return if i < 1
        #  0 -> 1
        # -1 -> 2
        1 - i
    elseif i > N
        #  N+1 -> N
        #  N+2 -> N-1
        2N + 1 - i
    else
        i
    end
end

@inline function mangle_index(i, ::Bounded, ::Face, N)
    return if i < 2
        1
    elseif i > N + 1
        N
    else
        i
    end
end

using Base: @propagate_inbounds

@inline function mangle_indices(hla::HalolessArray{LX, LY, LZ}, (i, j, k)) where {LX, LY, LZ}
    @assert ndims(hla.data) == 3
    Nx, Ny, Nz = size(hla.data)
    TX, TY, TZ = hla.topology
    i′ = mangle_index(i, TX(), LX(), Nx)
    j′ = mangle_index(j, TY(), LY(), Ny)
    k′ = mangle_index(k, TZ(), LZ(), Nz)
    return i′, j′, k′
end

@inline mangle_indices(hla::HalolessArray{<:Any, <:Any, Nothing}, ind::Tuple{Int, Int, Int}) = mangle_indices(hla, (ind[1], ind[2]))
@inline mangle_indices(hla::HalolessArray{Nothing, <:Any, Nothing}, ind::Tuple{Int, Int, Int}) = mangle_indices(hla, (ind[2],))
    
@inline function mangle_indices(hla::HalolessArray{LX, LY, Nothing}, (i, j)) where {LX, LY}
    Nx = size(hla.data, 1)
    Ny = size(hla.data, 2)
    TX, TY, TZ = hla.topology
    i′ = mangle_index(i, TX(), LX(), Nx)
    j′ = mangle_index(j, TY(), LY(), Ny)
    return i′, j′
end

@inline function mangle_indices(hla::HalolessArray{Nothing, LY, Nothing}, (j,)) where LY
    # @assert ndims(hla.data) == 1
    Ny = length(hla.data)
    TX, TY, TZ = hla.topology
    j′ = mangle_index(j, TY(), LY(), Ny)
    return j′
end

@propagate_inbounds Base.getindex(hla::HalolessArray, ind...) = getindex(hla.data, mangle_indices(hla, ind)...)
# @propagate_inbounds Base.setindex!(hla::HalolessArray, v, ind...) = setindex!(hla.data, v, mangle_indices(hla, ind)...)
@propagate_inbounds Base.setindex!(hla::HalolessArray, v, ind...) = setindex!(hla.data, v, ind...)

