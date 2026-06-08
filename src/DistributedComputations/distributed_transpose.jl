using Oceananigans.Grids: architecture
using Oceananigans.Architectures: on_architecture
using KernelAbstractions: @index, @kernel
using MPI: VBuffer, Alltoallv!

# Transpose directions are assumed to work only in the following configuration
# z -> y -> x -> y -> z
# where z stands for z-local data, y for y-local data, and x for x-local data
# The initial field is always assumed to be in the z-complete configuration

# Fallbacks for slab decompositions
transpose_z_to_y!(::SlabYFields) = nothing
transpose_y_to_z!(::SlabYFields) = nothing
transpose_x_to_y!(::SlabXFields) = nothing
transpose_y_to_x!(::SlabXFields) = nothing

# Since z -> y -> x -> y -> z we only nedd to define the `pack` and `unpack` kernels
# for the x and z configurations once, y requires two definitions depending on which
# configuration it's interacting with. Therefore, `_pack_buffer_x!` and `_pack_buffer_z!`
# are packing the buffer from a x-local configuration and from a z-local configuration.
# There is no ambiguity here because the x- and z- configurations communicate only with the y-configuration.
# On the other hand, for the y-configuration there are two ways to pack a buffer and two ways to unpack it
# depending on whether the y-configuration is going (or coming from) a x- or a z-configuration

@kernel function _pack_buffer_z_to_y!(yzbuff, zfield, N)
    i, j, k = @index(Global, NTuple)
    Nx, Ny, _ = N
    @inbounds yzbuff.send[j + Ny * (i-1 + Nx * (k-1))] = zfield[i, j, k]
end

@kernel function _pack_buffer_x_to_y!(xybuff, xfield, N)
    i, j, k = @index(Global, NTuple)
    _, Ny, Nz = N
    @inbounds xybuff.send[j + Ny * (k-1 + Nz * (i-1))] = xfield[i, j, k]
end

# packing a y buffer for communication with a x-local direction (y -> x communication)
@kernel function _pack_buffer_y_to_x!(xybuff, yfield, N)
    i, j, k = @index(Global, NTuple)
    Nx, _, Nz = N
    @inbounds xybuff.send[i + Nx * (k-1 + Nz * (j-1))] = yfield[i, j, k]
end

# packing a y buffer for communication with a z-local direction (y -> z communication)
@kernel function _pack_buffer_y_to_z!(xybuff, yfield, N)
    i, j, k = @index(Global, NTuple)
    Nx, _, Nz = N
    @inbounds xybuff.send[k + Nz * (i-1 + Nx * (j-1))] = yfield[i, j, k]
end

@kernel function _unpack_buffer_x_from_y!(xybuff, xfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = n[1], N[2], N[3]
    @inbounds begin
        i′  = mod(i - 1, size[1]) + 1
        m   = (i - 1) ÷ size[1]
        idx = i′ + size[1] * (k-1 + size[3] * (j-1)) + m * prod(size)
        xfield[i, j, k] = xybuff.recv[idx]
    end
end

@kernel function _unpack_buffer_z_from_y!(yzbuff, zfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = N[1], N[2], n[3]
    @inbounds begin
        k′  = mod(k - 1, size[3]) + 1
        m   = (k - 1) ÷ size[3]
        idx = k′ + size[3] * (i-1 + size[1] * (j-1)) + m * prod(size)
        zfield[i, j, k] = yzbuff.recv[idx]
    end
end

# unpacking a y buffer from a communication with z-local direction (z -> y)
@kernel function _unpack_buffer_y_from_z!(yzbuff, yfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, size[2]) + 1
        m   = (j - 1) ÷ size[2]
        idx = j′ + size[2] * (i-1 + size[1] * (k-1)) + m * prod(size)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

# unpacking a y buffer from a communication with x-local direction (x -> y)
@kernel function _unpack_buffer_y_from_x!(yzbuff, yfield, N, n)
    i, j, k = @index(Global, NTuple)
    size = N[1], n[2], N[3]
    @inbounds begin
        j′  = mod(j - 1, size[2]) + 1
        m   = (j - 1) ÷ size[2]
        idx = j′ + size[2] * (k-1 + size[3] * (i-1)) + m * prod(size)
        yfield[i, j, k] = yzbuff.recv[idx]
    end
end

pack_buffer_x_to_y!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_x_to_y!, buff, f, size(f))
pack_buffer_z_to_y!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_z_to_y!, buff, f, size(f))
pack_buffer_y_to_x!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_y_to_x!, buff, f, size(f))
pack_buffer_y_to_z!(buff, f) = launch!(architecture(f), f.grid, :xyz, _pack_buffer_y_to_z!, buff, f, size(f))

unpack_buffer_x_from_y!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_x_from_y!, buff, f, size(f), size(fo))
unpack_buffer_z_from_y!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_z_from_y!, buff, f, size(f), size(fo))
unpack_buffer_y_from_x!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_y_from_x!, buff, f, size(f), size(fo))
unpack_buffer_y_from_z!(f, fo, buff) = launch!(architecture(f), f.grid, :xyz, _unpack_buffer_y_from_z!, buff, f, size(f), size(fo))

for (from, to, buff) in zip([:y, :z, :y, :x], [:z, :y, :x, :y], [:yz, :yz, :xy, :xy])
    transpose!      = Symbol(:transpose_, from, :_to_, to, :(!))
    pack_buffer!    = Symbol(:pack_buffer_, from, :_to_, to, :(!))
    unpack_buffer!  = Symbol(:unpack_buffer_, to, :_from_, from, :(!))

    buffer = Symbol(buff, :buff)
    fromfield = Symbol(from, :field)
    tofield = Symbol(to, :field)

    transpose_name = string(transpose!)
    to_name = string(to)
    from_name = string(from)

    pack_buffer_name = string(pack_buffer!)
    unpack_buffer_name = string(unpack_buffer!)

    @eval begin
        """
            $($transpose_name)(pf::TransposableField)

        Transpose the fields in `TransposableField` from a $($from_name)-local configuration
        (located in `pf.$($from_name)field`) to a $($to_name)-local configuration located
        in `pf.$($to_name)field`.

        Transpose Algorithm:
        ====================

        The transpose algorithm works in the following manner

        1. We `pack` the three-dimensional data into a one-dimensional buffer to be sent to the other cores
           We need to synchronize the GPU afterwards before any communication can take place. The packing is
           done in the `$($pack_buffer_name)` function.

        2. The one-dimensional buffer is communicated to all the cores using an in-place `Alltoallv!` MPI
           routine. From the [MPI.jl documentation](https://juliaparallel.org/MPI.jl/stable/reference/collective/):

           Every process divides the Buffer into `Comm_size(comm)` chunks of equal size,
           sending the j-th chunk to the process of rank j-1. Every process stores the data received from rank j-1 process
           in the j-th chunk of the buffer.

           ```
           rank    send buf                             recv buf
           ----    --------                             --------
           0      a, b, c, d, e, f       Alltoall      a, b, A, B, α, β
           1      A, B, C, D, E, F  ---------------->  c, d, C, D, γ, ψ
           2      α, β, γ, ψ, η, ν                     e, f, E, F, η, ν
           ```

           The `Alltoallv` function allows chunks of different sizes to be sent to different cores by passing a `count`,
           for the moment, chunks of the same size are passed, requiring that the ranks divide the number of grid
           cells evenly.

        3. Once the chunks have been communicated, we `unpack` the received one-dimensional buffer into the three-dimensional
           field making sure the configuration of the data fits the reshaping. The unpacking is
           done via the `$($unpack_buffer_name)` function.

        Limitations:
        ============

        - The tranpose is configured to work only in the following four directions:

          1. z-local to y-local
          2. y-local to x-local
          3. x-local to y-local
          4. y-local to z-local

          i.e., there is no direct transpose connecting a x-local to a z-local configuration.

        - Since (at the moment) the `Alltoallv` allows only chunks of the same size to be communicated, and
          x-local and z-local only communicate through the y-local configuration, the limitations are that:

          * The number of ranks that divide the x-direction should divide evenly the y-direction
          * The number of ranks that divide the y-direction should divide evenly the x-direction

          which implies that

          * For 2D fields in XY (flat z-direction) we can traspose only if the partitioning is in X
        """
        function $transpose!(pf::TransposableField)
            $pack_buffer!(pf.$buffer, pf.$fromfield) # pack the one-dimensional buffer for Alltoallv! call
            sync_device!(architecture(pf.$fromfield)) # Device needs to be synched with host before MPI call
            Alltoallv!(VBuffer(pf.$buffer.send, pf.counts.$buff), VBuffer(pf.$buffer.recv, pf.counts.$buff), pf.comms.$buff) # Actually transpose!
            $unpack_buffer!(pf.$tofield, pf.$fromfield, pf.$buffer) # unpack the one-dimensional buffer into the 3D field
            return nothing
        end
    end
end
