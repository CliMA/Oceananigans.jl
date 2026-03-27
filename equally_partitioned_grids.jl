
using Oceananigans
using Oceananigans.DistributedComputations: Equal, barrier
using MPI
MPI.Init()

# Total number of ranks
Nr = MPI.Comm_size(MPI.COMM_WORLD)

# Allocate half the ranks to y, and the rest to x
Rx = Nr ÷ 2
partition = Partition(x=Rx, y=Equal())
arch = Distributed(CPU(); partition)

grid = RectilinearGrid(arch,
                       size = (48, 48, 16),
                       x = (0, 64),
                       y = (0, 64),
                       z = (0, 16),
                       topology = (Periodic, Periodic, Bounded))

# Let's see all the grids this time.
for r in 0:Nr-1
    if r == arch.local_rank
        msg = string("On rank ", r, ":", '
', '
',
                     arch, '
',
                     grid)
        @info msg
    end

    barrier(arch)
end
