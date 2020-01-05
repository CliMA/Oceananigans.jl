using Test

import MPI

using Oceananigans: interior, fill_halo_regions!

include("distributed_model.jl")

MPI.Init()

dm = DistributedModel(ranks = (2, 2, 2), size = (16, 16, 16),
                      x = (0, 1), y = (-0.5, 0.5), z = (-10, 0),
                      boundary_conditions = HorizontallyPeriodicSolutionBCs(),
                      poisson_solver = nothing)

my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
# @info "Rank $my_rank: $(dm.connectivity), $(dm.model.grid.zF[end])"
# @info "u.x BCs: $(dm.model.boundary_conditions.solution.u.x)"

interior(dm.model.velocities.u) .= rand(8, 8, 8)
fill_halo_regions!(dm.model.velocities.u.data, dm.model.boundary_conditions.solution.u, CPU(), dm.model.grid)
