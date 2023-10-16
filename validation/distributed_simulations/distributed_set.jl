using MPI
using Oceananigans
using Oceananigans.DistributedComputations

topology = (Periodic, Periodic, Flat)
arch = Distributed(CPU(); topology)

Nranks = MPI.Comm_size(arch.communicator)
grid = RectilinearGrid(arch; topology, size=(16 ÷ Nranks, 16), extent=(2π, 2π))

c = CenterField(grid)

f(x, y, z) = rand()
set!(c, f)
cmax = maximum(c)
@info "(function) rank $rank has max|c|: $cmax"

a = rand(size(c)...)
set!(c, a)
cmax = maximum(c)
@info "(array) rank $rank has max|c|: $cmax"

