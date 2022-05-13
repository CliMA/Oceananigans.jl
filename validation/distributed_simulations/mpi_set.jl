using MPI
using Oceananigans
using Oceananigans.Distributed

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(MPI.COMM_WORLD)

# Setup model
topology = (Periodic, Periodic, Flat)
arch = MultiArch(CPU(); topology, ranks=(1, Nranks, 1))
grid = RectilinearGrid(arch; topology, size=(16, 16), extent=(2π, 2π))
c = CenterField(grid)

f(x, y, z) = rand()
set!(c, f)
cmax = maximum(c)
@info "(function) rank $rank has max|c|: $cmax"

a = rand(size(c)...)
set!(c, a)
cmax = maximum(c)
@info "(array) rank $rank has max|c|: $cmax"

