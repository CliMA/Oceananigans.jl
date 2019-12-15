import MPI

using Oceananigans

using Oceananigans.Grids: validate_tupled_argument

struct DistributedModel{A, R, G, C}
                 ranks :: R
                models :: A
    connectivity_graph :: G
              MPI_Comm :: C
end

const RankConnectivity = NamedTuple{(:east, :west, :north, :south, :top, :bottom)}

@inline index2rank(i, j, k, Rx, Ry, Rz) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)

@inline function rank2index(r, Rx, Ry, Rz)
    i = div(r, Ry*Rz)
    r -= i*Ry*Rz
    j = div(r, Rz)
    k = mod(r, Rz)
    return i+1, j+1, k+1
end

function DistributedModel(; size, length, ranks, model_kwargs...)
    validate_tupled_argument(ranks, Int, "size")
    validate_tupled_argument(ranks, Number, "length")
    validate_tupled_argument(ranks, Int, "ranks")

    Nx, Ny, Nz = size
    Lx, Ly, Lz = length

    Rx, Ry, Rz = ranks
    total_ranks = Rx*Ry*Rz

    MPI.Init()
    comm = MPI.COMM_WORLD

    mpi_ranks = MPI.Comm_size(comm)
    my_rank   = MPI.Comm_rank(comm)

    if my_rank == 0
        if total_ranks != mpi_ranks
            throw(ArgumentError("ranks=($Rx, $Ry, $Rz) [$total_ranks total] inconsistent " *
                                "with number of MPI ranks: $mpi_ranks. Exiting with code 1."))
            MPI.Finalize()
            exit(code=1)
        end
    end

    # Ensure that ranks 1:N don't go ahead if total_ranks != mpi_ranks.
    MPI.Barrier(comm)

    model_id = my_rank + 1
    index = rank2index(my_rank, Rx, Ry, Rz)
    rr = index2rank(index..., Rx, Ry, Rz)
    # @info "rank=$my_rank, index=$index, index2rank=$rr"

    i, j, k = rank2index(my_rank, Rx, Ry, Rz)
    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz
    x₁, x₂ = (i-1)*lx, i*lx
    y₁, y₂ = (j-1)*ly, j*ly
    z₁, z₂ = (k-1)*lz, k*lz
    @info "rank=$my_rank, x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
    grid = RegularCartesianGrid(size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂))

    return DistributedModel(ranks, nothing, nothing, comm)
end

dm = DistributedModel(size=(32, 32, 32), length=(1, 2, 5), ranks=(2, 2, 2))

MPI.Finalize()
