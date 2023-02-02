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

Rx = Ry = Int(sqrt(Nranks))

topo = (Periodic, Bounded, Bounded)
arch = MultiArch(GPU(); topology = topo, ranks=(Rx, Ry, 1))

grid = RectilinearGrid(arch,
                       size = (Rx * 2, Ry * 2, 1),
                       extent = (1, 1, 1),
                       halo = (1, 1, 1),
                       topology = topo)

cf = CenterField(grid)
xf = XFaceField(grid)
yf = YFaceField(grid)

nx, ny = size(grid)[[1, 2]]
hx, hy = halo_size(grid)[[1, 2]]

fill!(cf, rank)
fill!(xf, rank)
fill!(yf, rank)

@show rank, arch.connectivity

fill_halo_regions!((cf, xf, yf))

@show rank, view(parent(cf), 1+nx+hx:nx+2hx, :, :)
@show rank, view(parent(cf), 1:hx, :, :)
@show rank, view(parent(cf), :, 1+ny+hy:ny+2hy, :)
@show rank, view(parent(cf), :, 1:hy, :)
