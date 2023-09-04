using Oceananigans.Grids: architecture
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @index, @kernel

# Fallbacks for slab decompositions
transpose_z_to_y!(::SlabYFields) = nothing
transpose_y_to_z!(::SlabYFields) = nothing
transpose_x_to_y!(::SlabXFields) = nothing
transpose_y_to_x!(::SlabXFields) = nothing

@kernel function _pack_buffer_z!(yzbuff, zfield, N)
    i, j, k = @index(Global, NTuple)
    @inbounds yzbuff.send[i + N[1] * ((j-1) + N[2] * (k-1))] = zfield[i, j, k]
end

@kernel function _pack_buffer_x!(xybuff, xfield, N)
    i, j, k = @index(Global, NTuple)
    @inbounds xybuff.send[k + N[3] * ((j-1) + N[2] * (i-1))] = xfield[i, j, k]
end

@kernel function _pack_buffer_y!(xybuff, yfield, N)
    i, j, k = @index(Global, NTuple)
    @inbounds xybuff.send[i + N[1] * ((k-1) + N[3] * (j-1))] = yfield[i, j, k]
end

@kernel function _unpack_buffer_x!(yxbuff, xfield, N, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        add = ifelse(i > n[1], n[1]*N[2]*N[3], 0)
        i′  = ifelse(i > n[1], i - n[1], i)
        @inbounds xfield[i, j, k] = yxbuff.recv[i′ + n[1] * ((k-1) + N[3] * (j-1)) + add]
    end
end

@kernel function _unpack_buffer_z!(yzbuff, zfield, N, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        add = ifelse(k > n[3], n[3]*N[1]*N[2], 0)
        k′  = ifelse(k > n[3], k - n[3], k)
        @inbounds zfield[i, j, k] = yzbuff.recv[i + N[1] * ((k′-1) + n[3] * (j-1)) + add]
    end
end

@kernel function _unpack_buffer_yz!(yzbuff, yfield, N, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        add = ifelse(j > n[2], n[2]*N[1]*N[3], 0)
        j′  = ifelse(j > n[2], j - n[2], j)
        @inbounds yfield[i, j, k] = yzbuff.recv[i + N[1] * ((j′-1) + n[2] * (k-1)) + add]
    end
end

@kernel function _unpack_buffer_yx!(yzbuff, yfield, N, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        add = ifelse(j > n[2], n[2]*N[1]*N[3], 0)
        j′  = ifelse(j > n[2], j - n[2], j)
        @inbounds yfield[i, j, k] = yzbuff.recv[k + N[3] * ((j′-1) + n[2] * (i-1)) + add]
    end
end

pack_buffer_x!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_x!, buff, f, size(f))
pack_buffer_y!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_y!, buff, f, size(f))
pack_buffer_z!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_z!, buff, f, size(f))

unpack_buffer_x!(f, fo, buff)  = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_x!,  buff, f, size(f), size(fo))
unpack_buffer_z!(f, fo, buff)  = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_z!,  buff, f, size(f), size(fo))
unpack_buffer_yx!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_yx!, buff, f, size(f), size(fo))
unpack_buffer_yz!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_yz!, buff, f, size(f), size(fo))

for (from, to, buff) in zip([:y, :z, :y, :x], [:z, :y, :x, :y], [:yz, :yz, :xy, :xy])
    transpose!      = Symbol(:transpose_, from, :_to_, to, :(!))
    pack_buffer!    = Symbol(:pack_buffer_, from, :(!))
    unpack_buffer!  = to == :y ? Symbol(:unpack_buffer_, to, from, :(!)) : Symbol(:unpack_buffer_, to, :(!))
    
    buffer = Symbol(buff, :buff)
    fromfield = Symbol(from, :field)
    tofield = Symbol(to, :field)

    @eval begin
        function $transpose!(pf::ParallelFields)
            $pack_buffer!(pf.$buffer, pf.$fromfield)
            # Actually transpose!
            MPI.Alltoallv!(pf.$buffer.send, pf.$buffer.recv, pf.counts.$buff, pf.counts.$buff, pf.comms.$buff)

            $unpack_buffer!(pf.$tofield, pf.$fromfield, pf.$buffer)
            
            fill_halo_regions!(pf.$tofield; only_local_halos = true)
            return nothing
        end
    end
end
