using Oceananigans.Grids: architecture
using Oceananigans.Architectures: arch_array
using KernelAbstractions: @index, @kernel

####
#### Twin transposed grid
####
struct ParallelFields{FX, FY, FZ, YZ, XY, C, Comms}
    xfield :: FX # X-direction is free
    yfield :: FY # Y-direction is free
    zfield :: FZ # Z-direction is free (original field)
    yzbuff :: YZ # if `nothing` slab decomposition with `Ry == 1`
    xybuff :: XY # if `nothing` slab decomposition with `Rx == 1`
    counts :: C
    comms :: Comms
end

const SlabYFields = ParallelFields{<:Any, <:Any, <:Any, <:Nothing} # Y-direction is free
const SlabXFields = ParallelFields{<:Any, <:Any, <:Any, <:Any, <:Nothing} # X-direction is free

function ParallelFields(field_in)
    zgrid  = field_in.grid
    zfield = field_in # Assuming we start from a z-free configuration

    ygrid = TwinGrid(zgrid; free_dim = :y)
    xgrid = TwinGrid(zgrid; free_dim = :x)

    zarch = architecture(zgrid)
    yarch = architecture(ygrid)

    loc = location(field_in)
    Rx, Ry, _ = zarch.ranks
    yfield = Ry == 1 ? zfield : Field(loc, ygrid)
    xfield = Rx == 1 ? yfield : Field(loc, xgrid)
    
    Nx = size(xgrid)
    Ny = size(ygrid)
    Nz = size(zgrid)

    yzbuffer = Ry == 1 ? nothing : (send = arch_array(zarch, zeros(prod(Ny))), 
                                    recv = arch_array(zarch, zeros(prod(Nz))))
    xybuffer = Rx == 1 ? nothing : (send = arch_array(zarch, zeros(prod(Nx))), 
                                    recv = arch_array(zarch, zeros(prod(Ny))))
    
    yzcomm = MPI.Comm_split(MPI.COMM_WORLD, zarch.local_index[1], zarch.local_index[1])
    xycomm = MPI.Comm_split(MPI.COMM_WORLD, yarch.local_index[3], yarch.local_index[3])
            
    yzcounts = zeros(Int, zarch.ranks[2] * zarch.ranks[3])
    xycounts = zeros(Int, yarch.ranks[1] * yarch.ranks[2])

    yzrank = MPI.Comm_rank(yzcomm)
    xyrank = MPI.Comm_rank(xycomm)

    yzcounts[yzrank + 1] = Ny[1] * Nz[2] * Ny[3]
    xycounts[xyrank + 1] = Ny[1] * Nx[2] * Nx[3]
    
    MPI.Allreduce!(yzcounts, +, yzcomm)
    MPI.Allreduce!(xycounts, +, xycomm)

    return ParallelFields(xfield, yfield, zfield, 
                          yzbuffer, xybuffer,
                          (; yz = yzcounts, xy = xycounts),
                          (; yz = yzcomm,   xy = xycomm))
end

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

function TwinGrid(grid::DistributedGrid; free_dim = :y)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    R = arch.ranks

    nx, ny, nz = n = size(grid)
    Nx, Ny, Nz = map(sum, concatenate_local_sizes(n, arch))

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, R[1], ri, rj, rk, arch.communicator)
    TY = reconstruct_global_topology(TY, R[2], rj, ri, rk, arch.communicator)
    TZ = reconstruct_global_topology(TZ, R[3], rk, ri, rj, arch.communicator)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    ## This will not work with 3D parallelizations!!
    xG = R[1] == 1 ? x : assemble(x, nx, R[1], ri, rj, rk, arch.communicator)
    yG = R[2] == 1 ? y : assemble(y, ny, R[2], rj, ri, rk, arch.communicator)
    zG = R[3] == 1 ? z : assemble(z, nz, R[3], rk, ri, rj, arch.communicator)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    if free_dim == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif free_dim == :x
        ranks = 1, R[1], R[2]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz ÷ ranks[3]

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif free_dims == :z
        @warn "That is the standard grid!!!"
        return grid
    end

    new_arch = DistributedArch(child_arch; 
                               ranks,
                               topology = (TX, TY, TZ))

    return construct_grid(grid, new_arch, FT; 
                          size = (nnx, nny, nnz), 
                        extent = (xG, yG, zG),
                      topology = (TX, TY, TZ))
end

construct_grid(::RectilinearGrid, arch, FT; size, extent, topology) = 
        RectilinearGrid(arch, FT; size, 
                        x = extent[1],
                        y = extent[2], 
                        z = extent[3],
                        topology)

construct_grid(::LatitudeLongitudeGrid, arch, FT; size, extent, topology) = 
        LatitudeLongitudeGrid(arch, FT; size, 
                        longitude = extent[1],
                         latitude = extent[2], 
                                z = extent[3],
                                topology)

