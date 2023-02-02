using MPI
using Oceananigans
using Oceananigans.Distributed
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: topology, architecture, halo_size
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

n = size(grid, 1)
h = halo_size(grid)[1]

fill!(cf, rank)
fill!(xf, rank)
fill!(yf, rank)

@show rank, arch.connectivity

fill_halo_regions!((cf, xf, yf))

@show rank, view(parent(cf), 1+n+h:n+2h, :, :)
@show rank, view(parent(cf), 1:h, :, :)

@show rank, view(parent(xf), 1+n+h:n+2h, :, :)
@show rank, view(parent(xf), 1:h, :, :)

@show rank, view(parent(yf), 1+n+h:n+2h, :, :)
@show rank, view(parent(yf), 1:h, :, :)
