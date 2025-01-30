using Oceananigans
using MPI
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField
using Printf
using Statistics
using Oceananigans.BoundaryConditions
using Oceananigans.DistributedComputations    
using Random
using JLD2

# Run with 
#
# ```julia 
#   mpiexec -n 4 julia --project distributed_hydrostatic_turbulence.jl
# ```

function run_simulation(nx, ny, arch; topology = (Periodic, Periodic, Bounded))
    grid = RectilinearGrid(arch; topology, size = (Nx, Ny, 10), extent=(4π, 4π, 0.5), halo=(8, 8, 8))
    
    bottom(x, y) = (x > π && x < 3π/2 && y > π/2 && y < 3π/2) ? 1.0 : - grid.Lz - 1.0
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = WENOVectorInvariant(),
                                        free_surface = SplitExplicitFreeSurface(grid, substeps=10),
                                        tracer_advection = WENO(),
                                        buoyancy = nothing,
                                        coriolis = FPlane(f = 1),
                                        tracers = :c)

    # Scale seed with rank to avoid symmetry
    local_rank = arch.local_rank

    Random.seed!(1234 * (local_rank + 1))

    uᵢ(x, y, z) = 1 - 2rand()
    cᵢ(x, y, z) = x > 3π/2 && x < 5π/2 && y > 3π/2 && y < 5π/2

    set!(model, u=uᵢ, v=uᵢ, c=cᵢ)

    outputs = merge(model.velocities, model.tracers)

    progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time), Δt: $(sim.Δt)"

    simulation = Simulation(model, Δt=0.01, stop_time=100.0)

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    filepath = "mpi_hydrostatic_turbulence_rank$(local_rank)"

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs, 
                                                          filename=filepath, 
                                                          schedule=TimeInterval(0.1),
                                                          overwrite_existing=true)

    run!(simulation)
end

Nx = 128
Ny = 128

arch = Distributed(CPU(), partition = Partition(2, 2)) 

# Run the simulation
run_simulation(Nx, Ny, arch)
