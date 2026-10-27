# Distributed set! validation
#
# Run with:
#
#   mpiexec -n 2 julia --project distributed_set.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations

topology = (Periodic, Periodic, Flat)
arch = Distributed(CPU())

Nranks = MPI.Comm_size(arch.communicator)
rank = MPI.Comm_rank(arch.communicator)

grid = RectilinearGrid(arch; topology, size = (16 ÷ Nranks, 16), extent = (2π, 2π))

c = CenterField(grid)

# Test 1: set! with a function (note: 2 args because z is Flat)
f(x, y) = rand()
set!(c, f)
cmax = maximum(c)
@info "(function) rank $rank has max|c|: $cmax"

# Test 2: set! with an array
a = rand(size(c)...)
set!(c, a)
cmax = maximum(c)
@info "(array) rank $rank has max|c|: $cmax"
