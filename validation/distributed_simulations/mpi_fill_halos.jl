using MPI
using Oceananigans
using Oceananigans.Distributed
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: topology, architecture
using Oceananigans.Units: kilometers, meters
using Printf
using JLD2

MPI.Init()

comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

Rx = Nranks

topo = (Periodic, Periodic, Bounded)
arch = MultiArch(GPU(); topology = topo, ranks=(Nranks, 1, 1))

grid = RectilinearGrid(arch,
                       size = (Nranks * 2, 2, 1),
                       extent = (1, 1, 1),
                       topology = topo)

cf = CenterField(grid)
xf = XFaceField(grid)
yf = YFaceField(grid)

fill!(cf, rank)
fill!(xf, rank)
fill!(yf, rank)

@show rank, arch.connectivity

fill_halo_regions!((cf, xf, yf))

@show rank, view(cf.parent, 1+N+H:N+2H, :, :)
@show rank, view(cf.parent, 1:H, :, :)

@show rank, view(xf.parent, 1+N+H:N+2H, :, :)
@show rank, view(xf.parent, 1:H, :, :)

@show rank, view(yf.parent, 1+N+H:N+2H, :, :)
@show rank, view(yf.parent, 1:H, :, :)
