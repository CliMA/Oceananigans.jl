# Distributed hydrostatic turbulence validation
#
# Run with:
#
#   mpiexec -n 4 julia --project distributed_hydrostatic_turbulence.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Printf
using Statistics
using Random

function run_simulation(Nx, Ny, arch; topology = (Periodic, Periodic, Bounded))
    grid = RectilinearGrid(arch; topology, size = (Nx, Ny, 10), extent = (4π, 4π, 0.5), halo = (8, 8, 8))

    bottom(x, y) = (x > π && x < 3π/2 && y > π/2 && y < 3π/2) ? 1.0 : -grid.Lz - 1.0
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)

    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = VectorInvariant(vorticity_scheme = WENO(order = 9)),
                                        free_surface = SplitExplicitFreeSurface(grid, substeps = 10),
                                        tracer_advection = WENO(),
                                        buoyancy = nothing,
                                        coriolis = FPlane(f = 1),
                                        tracers = :c)

    # Scale seed with rank to avoid symmetry
    local_rank = MPI.Comm_rank(arch.communicator)
    Random.seed!(1234 * (local_rank + 1))

    set!(model, u = (x, y, z) -> 1 - 2rand(), v = (x, y, z) -> 1 - 2rand())

    mask(x, y, z) = x > 3π/2 && x < 5π/2 && y > 3π/2 && y < 5π/2
    set!(model.tracers.c, mask)

    set!(c, mask)

    u, v, _ = model.velocities
    # ζ = VerticalVorticityField(model)
    η = model.free_surface.displacement
    outputs = merge(model.velocities, model.tracers, (; η))
    simulation = Simulation(model, Δt=0.02, stop_time=100)
    conjure_time_step_wizard!(simulation, cfl = 0.2)
    progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time), Δt: $(sim.Δt)"
    add_callback!(simulation, progress, IterationInterval(10))

    run!(simulation)
    @info "Simulation completed on rank $local_rank"
    MPI.Barrier(arch.communicator)

    return nothing
end

Nx = 32
Ny = 32

arch = Distributed(CPU(), partition = Partition(2, 2))
run_simulation(Nx, Ny, arch)
