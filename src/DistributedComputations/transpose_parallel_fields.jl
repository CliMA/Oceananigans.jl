using Oceananigans.Grids: architecture
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @index, @kernel
using MPI: VBuffer, Alltoallv!

# Transpose directions are assumed to work only in the following configuration
# z -> y -> x -> y -> z
# where z stands for z-complete data, y for y-complete data, and x for x-complete data
# The initial field is always assumed to be in the z-complete configuration

# Fallbacks for slab decompositions
transpose_z_to_y!(::SlabYFields) = nothing
transpose_y_to_z!(::SlabYFields) = nothing
transpose_x_to_y!(::SlabXFields) = nothing
transpose_y_to_x!(::SlabXFields) = nothing

# Since z -> y -> x -> y -> z we only nedd to define the `pack` and `unpack` kernels
# for the x and z configurations once, y requires two definitions depending on which
# configuration it's interacting with

@kernel function _pack_buffer_z!(yzbuff, zfield, N)
    i, j, k = @index(Global, NTuple)
    @inbounds yzbuff.send[j + N[2] * (i-1 + N[1] * (k-1))] = zfield[i, j, k]
end

@kernel function _pack_buffer_x!(xybuff, xfield, N)
    i, j, k = @index(Global, NTuple)
    @inbounds xybuff.send[j + N[2] * (k-1 + N[3] * (i-1))] = xfield[i, j, k]
end

@kernel function _pack_buffer_yx!(xybuff, yfield, N) # y -> x
    i, j, k = @index(Global, NTuple)
    @inbounds xybuff.send[i + N[1] * (k-1 + N[3] * (j-1))] = yfield[i, j, k]
end

@kernel function _pack_buffer_yz!(xybuff, yfield, N) # y -> z
    i, j, k = @index(Global, NTuple)
    @inbounds xybuff.send[k + N[3] * (i-1 + N[1] * (j-1))] = yfield[i, j, k]
end

@kernel function _unpack_buffer_x!(xybuff, xfield, N, n)
    i, j, k = @index(Global, NTuple)
    nm = n[1], N[2], N[3]
    @inbounds begin
        i′  = mod(i - 1, nm[1]) + 1
        m   = (i - 1) ÷ nm[1]
        idx = i′ + nm[1] * (k-1 + nm[3] * (j-1)) + m * prod(nm)
        xfield[i, j, k] = xybuff.recv[idx]
    end
end

@kernel function _unpack_buffer_z!(yzbuff, zfield, N, n)
    i, j, k = @index(Global, NTuple)
    nm = N[1], N[2], n[3]
    @inbounds begin
        k′  = mod(k - 1, nm[3]) + 1
        m   = (k - 1) ÷ nm[3]
        idx = k′ + nm[3] * (i-1 + nm[1] * (j-1)) + m * prod(nm)
        zfield[i, j, k] = yzbuff.recv[idx]
    end
end

@kernel function _unpack_buffer_yz!(yzbuff, yfield, N, n) # z -> y
    i, j, k = @index(Global, NTuple)
    nm = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, nm[2]) + 1
        m   = (j - 1) ÷ nm[2]
        idx = j′ + nm[2] * (i-1 + nm[1] * (k-1)) + m * prod(nm)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

@kernel function _unpack_buffer_yx!(yzbuff, yfield, N, n) # x -> y
    i, j, k = @index(Global, NTuple)
    nm = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, nm[2]) + 1
        m   = (j - 1) ÷ nm[2] 
        idx = j′ + nm[2] * (k-1 + nm[3] * (i-1)) + m * prod(nm)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

pack_buffer_x!(buff, f)  = launch!(architecture(f), f.grid, :xyz, _pack_buffer_x!,  buff, f, size(f))
pack_buffer_z!(buff, f)  = launch!(architecture(f), f.grid, :xyz, _pack_buffer_z!,  buff, f, size(f))
pack_buffer_yx!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_yx!, buff, f, size(f))
pack_buffer_yz!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_yz!, buff, f, size(f))

unpack_buffer_x!(f, fo, buff)  = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_x!,  buff, f, size(f), size(fo))
unpack_buffer_z!(f, fo, buff)  = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_z!,  buff, f, size(f), size(fo))
unpack_buffer_yx!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_yx!, buff, f, size(f), size(fo))
unpack_buffer_yz!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_yz!, buff, f, size(f), size(fo))

for (from, to, buff) in zip([:y, :z, :y, :x], [:z, :y, :x, :y], [:yz, :yz, :xy, :xy])
    transpose!      = Symbol(:transpose_, from, :_to_, to, :(!))
    pack_buffer!    = from == :y ? Symbol(:pack_buffer_, from, to, :(!)) : Symbol(:pack_buffer_, from, :(!))
    unpack_buffer!  = to == :y ? Symbol(:unpack_buffer_, to, from, :(!)) : Symbol(:unpack_buffer_, to, :(!))
    
    buffer = Symbol(buff, :buff)
    fromfield = Symbol(from, :field)
    tofield = Symbol(to, :field)

    @eval begin
        function $transpose!(pf::ParallelFields)
            $pack_buffer!(pf.$buffer, pf.$fromfield) # pack the one-dimensional buffer for Alltoallv! call
            sync_device!(architecture(pf.$fromfield)) # Device needs to be synched with host before MPI call
            Alltoallv!(VBuffer(pf.$buffer.send, pf.counts.$buff), VBuffer(pf.$buffer.recv, pf.counts.$buff), pf.comms.$buff) # Actually transpose!
            $unpack_buffer!(pf.$tofield, pf.$fromfield, pf.$buffer) # unpack the one-dimensional buffer into the 3D field
            return nothing
        end
    end
end
