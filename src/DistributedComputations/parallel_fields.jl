using Oceananigans.Grids: architecture
using Oceananigans.Architectures: on_architecture

struct ParallelFields{FX, FY, FZ, YZ, XY, C, Comms}
    xfield :: FX # X-direction is free
    yfield :: FY # Y-direction is free
    zfield :: FZ # Z-direction is free (original field)
    yzbuff :: YZ # if `nothing` slab decomposition with `Ry == 1`
    xybuff :: XY # if `nothing` slab decomposition with `Rx == 1`
    counts :: C
    comms  :: Comms
end

const SlabYFields = ParallelFields{<:Any, <:Any, <:Any, <:Nothing} # Y-direction is free
const SlabXFields = ParallelFields{<:Any, <:Any, <:Any, <:Any, <:Nothing} # X-direction is free

"""
    ParallelFields(field_in, FT = eltype(field_in); with_halos = false)

Constructs a ParallelFields object that containes the allocated memory and the ruleset required
for distributed transpositions.

# Arguments
- `field_in`: The input field. It needs to be in a _z-free_ configuration (i.e. ranks[3] == 1).
- `FT`: The element type of the field. Defaults to the element type of `field_in`.
- `with_halos`: A boolean indicating whether to include halos in the field. Defaults to `false`.
"""
function ParallelFields(field_in, FT = eltype(field_in); with_halos = false)
    
    zgrid = field_in.grid # We support only a 2D partition in X and Y
    ygrid = twin_grid(zgrid; free_dimension = :y)
    xgrid = twin_grid(zgrid; free_dimension = :x)

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

    yzbuffer = Ry == 1 ? nothing : (send = on_architecture(zarch, zeros(FT, prod(Ny))), 
                                    recv = on_architecture(zarch, zeros(FT, prod(Nz))))
    xybuffer = Rx == 1 ? nothing : (send = on_architecture(zarch, zeros(FT, prod(Nx))), 
                                    recv = on_architecture(zarch, zeros(FT, prod(Ny))))
    
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

"""
    twin_grid(grid::DistributedGrid; free_dimension = :y)

Constructs a "twin" grid based on the provided distributed `grid` object.
The twin grid is a grid that discretizes the same domain of the original grid, just with a different partitioning strategy 
whereas the "free dimension" (i.e. the  non-partitioned dimension) is specified by the keyword argument `free_dimension`.
This could be either `:x` or `:y`.

Note that `free_dimension = :z` will return the original grid as we do not allow partitioning in the `z` direction 
"""
function twin_grid(grid::DistributedGrid; free_dimension = :y)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    R = arch.ranks

    nx, ny, nz = n = size(grid)
    Nx, Ny, Nz = global_size(arch, n)

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, R[1], ri, rj, rk, arch.communicator)
    TY = reconstruct_global_topology(TY, R[2], rj, ri, rk, arch.communicator)
    TZ = reconstruct_global_topology(TZ, R[3], rk, ri, rj, arch.communicator)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    ## This will not work with 3D parallelizations!!
    xG = R[1] == 1 ? x : assemble_coordinate(x, nx, R[1], ri, rj, rk, arch.communicator)
    yG = R[2] == 1 ? y : assemble_coordinate(y, ny, R[2], rj, ri, rk, arch.communicator)
    zG = R[3] == 1 ? z : assemble_coordinate(z, nz, R[3], rk, ri, rj, arch.communicator)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    if free_dimension == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif free_dimension == :x
        ranks = 1, R[1], R[2]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz ÷ ranks[3]

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif free_dimension == :z
        #TODO: a warning here?
        return grid
    end

    new_arch  = Distributed(child_arch; partition = Partition(ranks...))
    global_sz = global_size(new_arch, (nnx, nny, nnz))

    return construct_grid(grid, new_arch, FT; 
                          size = global_sz, 
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

