using Oceananigans.Grids: architecture
using Oceananigans.Architectures: arch_array

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

function ParallelFields(field_in, FT = eltype(field_in); with_halos = false)
    zgrid  = field_in.grid # We support only a 2D partition in X and Y

    ygrid = TwinGrid(zgrid; free_dim = :y)
    xgrid = TwinGrid(zgrid; free_dim = :x)

    Nx = size(xgrid)
    Ny = size(ygrid)
    Nz = size(zgrid)

    zarch = architecture(zgrid)
    yarch = architecture(ygrid)

    loc = location(field_in)

    Rx, Ry, _ = zarch.ranks
    if with_halos 
        zfield = Field(loc, zgrid, FT)
        yfield = Ry == 1 ? zfield : Field(loc, ygrid, FT)
        xfield = Rx == 1 ? yfield : Field(loc, xgrid, FT)
    else
        zfield = Field(loc, zgrid, FT; indices = (1:Nz[1], 1:Nz[2], 1:Nz[3]))
        yfield = Ry == 1 ? zfield : Field(loc, ygrid, FT; indices = (1:Ny[1], 1:Ny[2], 1:Ny[3]))
        xfield = Rx == 1 ? yfield : Field(loc, xgrid, FT; indices = (1:Nx[1], 1:Nx[2], 1:Nx[3]))
    end

    yzbuffer = Ry == 1 ? nothing : (send = arch_array(zarch, zeros(FT, prod(Ny))), 
                                    recv = arch_array(zarch, zeros(FT, prod(Nz))))
    xybuffer = Rx == 1 ? nothing : (send = arch_array(zarch, zeros(FT, prod(Nx))), 
                                    recv = arch_array(zarch, zeros(FT, prod(Ny))))
    
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

#####
##### Twin transposed grid
#####

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

    new_arch = Distributed(child_arch; 
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

