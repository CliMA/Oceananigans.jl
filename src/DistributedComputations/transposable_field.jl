using Oceananigans.Grids: architecture, deflate_tuple
using Oceananigans.Architectures: on_architecture

struct TransposableField{FX, FY, FZ, YZ, XY, C, Comms}
    xfield :: FX # X-direction is free (x-local)
    yfield :: FY # Y-direction is free (y-local)
    zfield :: FZ # Z-direction is free (original field, z-local)
    yzbuff :: YZ # if `nothing` slab decomposition with `Ry == 1`
    xybuff :: XY # if `nothing` slab decomposition with `Rx == 1`
    counts :: C
    comms  :: Comms
end

const SlabYFields = TransposableField{<:Any, <:Any, <:Any, <:Nothing} # Y-direction is free
const SlabXFields = TransposableField{<:Any, <:Any, <:Any, <:Any, <:Nothing} # X-direction is free

"""
    TransposableField(field_in, FT = eltype(field_in); with_halos = false)

Construct a TransposableField object that containes the allocated memory and the ruleset required
for distributed transpositions. This includes:
- `xfield`: A field with an unpartitioned x-direction (x-local)
- `yfield`: A field with an unpartitioned y-direction (y-local)
- `zfield`: A field with an unpartitioned z-direction (z-local)
- one-dimensional buffers for performing communication between the different configurations, in particular:
    - `yzbuffer`: A buffer for communication between the z- and y-configurations
    - `xybuffer`: A buffer for communication between the y- and x-configurations
  These buffers are "packed" with the three dimensional data and then "unpacked" in the target configuration once
  received by the target rank.
- `counts`: The size of the chunks in the buffers to be sent and received
- `comms`: The MPI communicators for the yz and xy directions (different from MPI.COMM_WORLD!!!)

A `TransposableField` object is used to perform distributed transpositions between different configurations with the
`transpose_z_to_y!`, `transpose_y_to_x!`, `transpose_x_to_y!`, and `transpose_y_to_z!` functions.
In particular:
- `transpose_z_to_y!` copies data from the z-configuration (`zfield`) to the y-configuration (`yfield`)
- `transpose_y_to_x!` copies data from the y-configuration (`yfield`) to the x-configuration (`xfield`)
- `transpose_x_to_y!` copies data from the x-configuration (`xfield`) to the y-configuration (`yfield`)
- `transpose_y_to_z!` copies data from the y-configuration (`yfield`) to the z-configuration (`zfield`)

For more information on the transposition algorithm, see the docstring for the `transpose` functions.

# Arguments
- `field_in`: The input field. It needs to be in a _z-free_ configuration (i.e. ranks[3] == 1).
- `FT`: The element type of the field. Defaults to the element type of `field_in`.
- `with_halos`: A boolean indicating whether to include halos in the field. Defaults to `false`.
"""
function TransposableField(field_in, FT = eltype(field_in); with_halos = false)

    zgrid = field_in.grid # We support only a 2D partition in X and Y
    ygrid = twin_grid(zgrid; local_direction = :y)
    xgrid = twin_grid(zgrid; local_direction = :x)

    xN = size(xgrid)
    yN = size(ygrid)
    zN = size(zgrid)

    zarch = architecture(zgrid)
    yarch = architecture(ygrid)

    loc = location(field_in)

    Rx, Ry, _ = zarch.ranks
    if with_halos
        zfield = Field(loc, zgrid, FT)
        yfield = Ry == 1 ? zfield : Field(loc, ygrid, FT)
        xfield = Rx == 1 ? yfield : Field(loc, xgrid, FT)
    else
        zfield = Field(loc, zgrid, FT; indices = (1:zN[1], 1:zN[2], 1:zN[3]))
        yfield = Ry == 1 ? zfield : Field(loc, ygrid, FT; indices = (1:yN[1], 1:yN[2], 1:yN[3]))
        xfield = Rx == 1 ? yfield : Field(loc, xgrid, FT; indices = (1:xN[1], 1:xN[2], 1:xN[3]))
    end

    # One dimensional buffers to "pack" three-dimensional data in for communication
    yzbuffer = Ry == 1 ? nothing : (send = on_architecture(zarch, zeros(FT, prod(yN))),
                                    recv = on_architecture(zarch, zeros(FT, prod(zN))))
    xybuffer = Rx == 1 ? nothing : (send = on_architecture(zarch, zeros(FT, prod(xN))),
                                    recv = on_architecture(zarch, zeros(FT, prod(yN))))

    yzcomm = MPI.Comm_split(MPI.COMM_WORLD, zarch.local_index[1], zarch.local_index[1])
    xycomm = MPI.Comm_split(MPI.COMM_WORLD, yarch.local_index[3], yarch.local_index[3])

    zRx, zRy, zRz = ranks(zarch)
    yRx, yRy, yRz = ranks(yarch)

    # size of the chunks in the buffers to be sent and received
    # (see the docstring for the `transpose` algorithms)
    yzcounts = zeros(Int, zRy * zRz)
    xycounts = zeros(Int, yRx * yRy)

    yzrank = MPI.Comm_rank(yzcomm)
    xyrank = MPI.Comm_rank(xycomm)

    yzcounts[yzrank + 1] = yN[1] * zN[2] * yN[3]
    xycounts[xyrank + 1] = yN[1] * xN[2] * xN[3]

    MPI.Allreduce!(yzcounts, +, yzcomm)
    MPI.Allreduce!(xycounts, +, xycomm)

    return TransposableField(xfield, yfield, zfield,
                             yzbuffer, xybuffer,
                             (; yz = yzcounts, xy = xycounts),
                             (; yz = yzcomm,   xy = xycomm))
end

#####
##### Twin transposed grid
#####

"""
    twin_grid(grid::DistributedGrid; local_direction = :y)

Construct a "twin" grid based on the provided distributed `grid` object.
The twin grid is a grid that discretizes the same domain of the original grid, just with a
different partitioning strategy whereas the "local dimension" (i.e. the non-partitioned dimension)
is specified by the keyword argument `local_direction`. This could be either `:x` or `:y`.

Note that `local_direction = :z` will return the original grid as we do not allow partitioning in
the `z` direction.
"""
function twin_grid(grid::DistributedGrid; local_direction = :y)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    R = arch.ranks

    nx, ny, nz = n = size(grid)
    Nx, Ny, Nz = global_size(arch, n)

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, R[1], ri, rj, rk, arch)
    TY = reconstruct_global_topology(TY, R[2], rj, ri, rk, arch)
    TZ = reconstruct_global_topology(TZ, R[3], rk, ri, rj, arch)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    xG = R[1] == 1 ? x : assemble_coordinate(x, nx, arch, 1)
    yG = R[2] == 1 ? y : assemble_coordinate(y, ny, arch, 2)
    zG = R[3] == 1 ? z : assemble_coordinate(z, nz, arch, 3)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    if local_direction == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif local_direction == :x
        ranks = 1, R[1], R[2]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz ÷ ranks[3]

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif local_direction == :z
        #TODO: a warning here?
        return grid
    end

    new_arch  = Distributed(child_arch; partition = Partition(ranks...))
    global_sz = global_size(new_arch, (nnx, nny, nnz))
    global_sz = deflate_tuple(TX, TY, TZ, global_sz)
    global_hl = halo_size(grid)
    global_hl = deflate_tuple(TX, TY, TZ, global_hl)

    return construct_grid(grid, new_arch, FT;
                          size = global_sz,
                          halo = global_hl,
                          x = xG, y = yG, z = zG,
                          topology = (TX, TY, TZ))
end

function construct_grid(::RectilinearGrid, arch, FT; size, halo, x, y, z, topology)
    TX, TY, TZ = topology
    x = TX == Flat ? nothing : x
    y = TY == Flat ? nothing : y
    z = TZ == Flat ? nothing : z

    return RectilinearGrid(arch, FT; size,
                           halo,
                           x, y, z,
                           topology)
end

function construct_grid(::LatitudeLongitudeGrid, arch, FT; size, halo, x, y, z, topology)
    TX, TY, TZ = topology
    longitude = TX == Flat ? nothing : x
    latitude  = TY == Flat ? nothing : y
    z         = TZ == Flat ? nothing : z

    return LatitudeLongitudeGrid(arch, FT; size,
                                 halo,
                                 longitude, latitude, z,
                                 topology)
end
