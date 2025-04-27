using MPI

#####
##### Handle commands, typically downloading files
##### which should be executed by only one rank or distributed among ranks
#####

# Utilities to make the macro work importing only Oceananigans.DistributedComputations and not MPI
mpi_initialized()     = MPI.Initialized()
mpi_rank(comm)        = MPI.Comm_rank(comm)
mpi_size(comm)        = MPI.Comm_size(comm)
global_barrier(comm)  = MPI.Barrier(comm)
global_communicator() = MPI.COMM_WORLD

"""
    @root communicator exp...

Perform `exp` only on rank 0 in communicator, otherwise known as the "root" rank.
Other ranks will wait for the root rank to finish before continuing.
If `communicator` is not provided, `MPI.COMM_WORLD` is used.
"""
macro root(communicator, exp)
    command = quote
        if Oceananigans.DistributedComputations.mpi_initialized()
            rank = Oceananigans.DistributedComputations.mpi_rank($communicator)
            if rank == 0
                $exp
            end
            Oceananigans.DistributedComputations.global_barrier($communicator)
        else
            $exp
        end
    end
    return esc(command)
end

macro root(exp)
    command = quote
        @root Oceananigans.DistributedComputations.global_communicator() $exp
    end
    return esc(command)
end

"""
    @onrank communicator rank exp...

Perform `exp` only on rank `rank` (0-based index) in `communicator`.
Other ranks will wait for rank `rank` to finish before continuing.
The expression is run anyways if MPI in not initialized.
If `communicator` is not provided, `MPI.COMM_WORLD` is used.
"""
macro onrank(communicator, on_rank, exp)
    command = quote
        mpi_initialized = Oceananigans.DistributedComputations.mpi_initialized()
        if !mpi_initialized
            $exp
        else
            rank = Oceananigans.DistributedComputations.mpi_rank($communicator)
            if rank == $on_rank
                $exp
            end
            Oceananigans.DistributedComputations.global_barrier($communicator)
        end
    end

    return esc(command)
end

macro onrank(rank, exp)
    command = quote
        @onrank Oceananigans.DistributedComputations.global_communicator() $rank $exp
    end
    return esc(command)
end

"""
    @distribute communicator for i in iterable
        ...
    end

Distribute a `for` loop among different ranks in `communicator`.
If `communicator` is not provided, `MPI.COMM_WORLD` is used.
"""
macro distribute(communicator, exp)
    if exp.head != :for
        error("The `@distribute` macro expects a `for` loop")
    end

    iterable = exp.args[1].args[2]
    variable = exp.args[1].args[1]
    forbody  = exp.args[2]

    # Safety net if the iterable variable has the same name as the
    # reserved variable names (nprocs, counter, rank)
    nprocs  = ifelse(variable == :nprocs,  :othernprocs,  :nprocs)
    counter = ifelse(variable == :counter, :othercounter, :counter)
    rank    = ifelse(variable == :rank,    :otherrank,    :rank)

    new_loop = quote
        mpi_initialized = Oceananigans.DistributedComputations.mpi_initialized()
        if !mpi_initialized
            $exp
        else
            $rank   = Oceananigans.DistributedComputations.mpi_rank($communicator)
            $nprocs = Oceananigans.DistributedComputations.mpi_size($communicator)
            for ($counter, $variable) in enumerate($iterable)
                if ($counter - 1) % $nprocs == $rank
                    $forbody
                end
            end
            Oceananigans.DistributedComputations.global_barrier($communicator)
        end
    end

    return esc(new_loop)
end

macro distribute(exp)
    command = quote
        @distribute Oceananigans.DistributedComputations.global_communicator() $exp
    end
    return esc(command)
end

"""
    @handshake communicator exp...

perform `exp` on all ranks in `communicator`, but only one rank at a time, where
ranks `r2 > r1` wait for rank `r1` to finish before executing `exp`.
If `communicator` is not provided, `MPI.COMM_WORLD` is used.
"""
macro handshake(communicator, exp)
    command = quote
        mpi_initialized = Oceananigans.DistributedComputations.mpi_initialized()
        if !mpi_initialized
            $exp
        else
            rank   = Oceananigans.DistributedComputations.mpi_rank($communicator)
            nprocs = Oceananigans.DistributedComputations.mpi_size($communicator)
            for r in 0 : nprocs -1
                if rank == r
                    $exp
                end
                Oceananigans.DistributedComputations.global_barrier($communicator)
            end
        end
    end
    return esc(command)
end

macro handshake(exp)
    command = quote
        @handshake Oceananigans.DistributedComputations.global_communicator() $exp
    end
    return esc(command)
end
