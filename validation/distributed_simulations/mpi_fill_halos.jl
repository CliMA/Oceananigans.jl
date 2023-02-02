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
Ry = 1

topo = (Periodic, Bounded, Bounded)
arch = MultiArch(CPU(); topology = topo, ranks=(Rx, Ry, 1))

grid = LatitudeLongitudeGrid(arch,
                             size = (Rx * 20, Ry * 20, 1),
                             latitude = (0, 1),
                             longitude = (0, 1),
                             z = (0, 1),
                             halo = (1, 1, 1),
                             topology = topo)

model = HydrostaticFreeSurfaceModel(; grid, free_surface = SplitExplicitFreeSurface(; substeps = 10), 
                                    buoyancy = nothing, tracers = ())

η = model.free_surface.η
U = model.free_surface.state.U̅

time_step!(model, 1.0)
time_step!(model, 1.0)
time_step!(model, 1.0)

nx, ny = size(η.grid)[[1, 2]]
hx, hy = halo_size(η.grid)[[1, 2]]

@show rank, interior(η, :, :, 1)
@show rank, interior(U, :, :, 1)

fill_halo_regions!((η, U))

@show rank, η.data.parent[1+nx+hx:nx+2hx, :, 1]
@show rank, η.data.parent[1:hx, :, 1]
@show rank, η.data.parent[:, 1+ny+hy:ny+2hy, 1]
@show rank, η.data.parent[:, 1:hy, 1]

@show rank, U.data.parent[1+nx+hx:nx+2hx, :, 1]
@show rank, U.data.parent[1:hx, :, 1]
@show rank, U.data.parent[:, 1+ny+hy:ny+2hy, 1]
@show rank, U.data.parent[:, 1:hy, 1]
