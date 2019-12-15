import MPI

using Oceananigans

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

function validate_tupled_argument(arg, argtype, argname)
    length(arg) == 3        || throw(ArgumentError("length($argname) must be 3."))
    all(isa.(arg, argtype)) || throw(ArgumentError("$argname=$arg must contain $argtype s."))
    all(arg .> 0)           || throw(ArgumentError("Elements of $argname=$arg must be > 0!"))
    return nothing
end

function DistributedModel(; ranks, model_kwargs...)
    validate_tupled_argument(ranks, Int, "ranks")
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
    println("Model #$my_rank reporting in")

    return DistributedModel(ranks, nothing, nothing, comm)
end

dm = DistributedModel(ranks=(2, 2, 1))
