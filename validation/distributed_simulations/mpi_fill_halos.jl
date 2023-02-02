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

Rx = sqrt(Nranks)
Ry = sqrt(Nranks)

topo = (Periodic, Bounded, Bounded)
arch = MultiArch(GPU(); topology = topo, ranks=(Rx, Ry, 1))

grid = RectilinearGrid(arch,
                       size = (Rx * 2, Ry * 2, 1),
                       extent = (1, 1, 1),
                       halo = (1, 1, 1),
                       topology = topo)

model = HydrostaticFreeSurfaceModel(; grid, free_surface = SplitExplicitFreeSurface(; substeps = 10), 
                                    buoyancy = nothing, tracers = ())

η = model.free_surface.η
U = model.free_surface.state.U̅

nx, ny = size(η.grid)[[1, 2]]
hx, hy = halo_size(η.grid)[[1, 2]]

set!(η, rank)
set!(U, rank)

@show rank, arch.connectivity

fill_halo_regions!((η, U))

@show rank, view(parent(η), 1+nx+hx:nx+2hx, :, 1)
@show rank, view(parent(η), 1:hx, :, 1)
@show rank, view(parent(η), :, 1+ny+hy:ny+2hy, 1)
@show rank, view(parent(η), :, 1:hy, 1)



@show rank, view(parent(U), 1+nx+hx:nx+2hx, :, 1)
@show rank, view(parent(U), 1:hx, :, 1)
@show rank, view(parent(U), :, 1+ny+hy:ny+2hy, 1)
@show rank, view(parent(U), :, 1:hy, 1)
