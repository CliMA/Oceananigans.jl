using Test

import MPI

using Oceananigans
using Oceananigans: BCType
using Oceananigans.Grids: validate_tupled_argument

#####
##### Converting between index and MPI rank taking k as the fast index
#####

@inline index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

@inline function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1
end

#####
##### Connectivity graph
#####

const Connectivity = NamedTuple{(:east, :west, :north, :south, :top, :bottom)}

function construct_connectivity(index, ranks, boundary_conditions)
    i, j, k = index
    Rx, Ry, Rz = ranks

    i_east = i+1 > Rx ? nothing : i+1
    i_west = i-1 < 1  ? nothing : i-1

    j_north = j+1 > Ry ? nothing : j+1
    j_south = j-1 < 1  ? nothing : j-1

    k_top = k+1 > Rz ? nothing : k+1
    k_bot = k-1 < 1  ? nothing : k-1

    r_east = isnothing(i_east) ? nothing : index2rank(i_east, j, k, Rx, Ry, Rz)
    r_west = isnothing(i_west) ? nothing : index2rank(i_west, j, k, Rx, Ry, Rz)

    r_north = isnothing(j_north) ? nothing : index2rank(i, j_north, k, Rx, Ry, Rz)
    r_south = isnothing(j_south) ? nothing : index2rank(i, j_south, k, Rx, Ry, Rz)

    r_top = isnothing(k_top) ? nothing : index2rank(i, j, k_top, Rx, Ry, Rz)
    r_bot = isnothing(k_bot) ? nothing : index2rank(i, j, k_bot, Rx, Ry, Rz)

    return (east=r_east, west=r_west, north=r_north,
            south=r_south, top=r_top, bottom=r_bot)
end

#####
##### Communication boundary condition
#####

struct Communication <: BCType end

#####
##### Distributed model struct and constructor
#####

struct DistributedModel{A, R, G}
                 ranks :: R
                 model :: A
          connectivity :: G
end

"""
    DistributedModel(size, x, y, z, ranks, model_kwargs...)

size: Number of total grid points.
x, y, z: Left and right endpoints for each dimension.
ranks: Number of ranks in each dimension.
model_kwargs: Passed to `Model` constructor.
"""
function DistributedModel(; size, x, y, z, ranks, boundary_conditions, model_kwargs...)
    validate_tupled_argument(ranks, Int, "size")
    validate_tupled_argument(ranks, Int, "ranks")

    Nx, Ny, Nz = size

    # Pull out left and right endpoints for full model.
    xL, xR = x
    yL, yR = y
    zL, zR = z
    Lx, Ly, Lz = xR-xL, yR-yL, zR-zL

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    comm = MPI.COMM_WORLD

    mpi_ranks = MPI.Comm_size(comm)
    my_rank   = MPI.Comm_rank(comm)

    if total_ranks != mpi_ranks
        throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                            "with number of MPI ranks: $mpi_ranks. Exiting with code 1."))
        MPI.Finalize()
        exit(code=1)
    end

    i, j, k = index = rank2index(my_rank, Rx, Ry, Rz)
    @debug "Rank: $my_rank, index: $index"

    #####
    ##### Construct local grid
    #####

    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly
    z₁, z₂ = zL + (k-1)*lz, zL + k*lz

    @debug "Constructing local grid: n=($nx, $ny, $nz), x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
    grid = RegularCartesianGrid(size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂))

    #####
    ##### Construct local connectivity
    #####

    my_connectivity = construct_connectivity(index, ranks, boundary_conditions)

    @debug "Local connectivity: $my_connectivity"

    #####
    ##### Construct local model
    #####

    my_model = Model(grid=grid)

    return DistributedModel(ranks, my_model, my_connectivity)
end

#####
##### Script/test/whatever
#####

MPI.Init()

dm = DistributedModel(ranks=(2, 2, 2), size=(32, 32, 32),
                      x=(0, 1), y=(-0.5, 0.5), z=(-10, 0),
                      boundary_conditions=nothing)

my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
@info "Rank $my_rank: $(dm.connectivity), $(dm.model.grid.zF[end])"

MPI.Finalize()
