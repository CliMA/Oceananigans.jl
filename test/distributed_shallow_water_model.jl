using Oceananigans
using Oceananigans.Distributed

MPI.Init()

      comm = MPI.COMM_WORLD
local_rank = MPI.Comm_rank(comm)
         R = MPI.Comm_size(comm)

Nx = pars(Int, ARGS[1])
Ny = pars(Int, ARGS[2])
 R = pars(Int, ARGS[3])

            topo = (Periodic, Periodic, Bounded)
distributed_grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, 1), extent=(1, 1, 1))
            arch = MultiCPU(grid=distributed_grid, ranks=(1, R, 1))
           model = DistributedShallowWaterModel(architecture=arch, grid=distributed_grid, gravitational_acceleration=1)
    
set!(model, h=model.grid.Lz)
time_step!(model, 1)

