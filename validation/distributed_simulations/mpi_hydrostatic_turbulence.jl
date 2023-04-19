using Oceananigans
using MPI

MPI.Initialized() || MPI.Init()

     comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)

@assert mpi_ranks == 4

using Statistics
using Oceananigans
using Oceananigans.Distributed

ranks = (2, 2, 1)
topo  = (Periodic, Periodic, Bounded)
arch  = DistributedArch(CPU(), ranks=ranks, topology=topo, use_buffers=true)

grid  = RectilinearGrid(arch, topology=topo, size=(28, 28, 1), extent=(4π, 4π, 0.5), halo=(3, 3, 3))

local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

free_surface = SplitExplicitFreeSurface(; substeps = 30)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                     momentum_advection = WENO(),
                     tracer_advection = WENO(),
                     buoyancy = nothing,
                     coriolis = FPlane(f = 1),
                     tracers = :c)

using Random
Random.seed!(1234 * (local_rank +1))

set!(model, u = (x, y, z) -> rand(), v = (x, y, z) -> rand())

mask(x, y, z) = x > π && x < 2π && y > π && y < 2π ? 1.0 : 0.0
if local_rank == 0
    set!(model.tracers.c, mask)
end

u, v, _ = model.velocities
outputs = merge(model.velocities, model.tracers)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.001, stop_time=100.0)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

filepath = "mpi_hydrostatic_turbulence_rank$(local_rank)"
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs, filename=filepath, schedule=TimeInterval(0.1),
                     overwrite_existing=true)

MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)

if rank == 0
    using Printf
    using NCDatasets
    using GLMakie

    iter = Observable(1)

    z1 = FieldTimeSeries("mpi_hydrostatic_turbulence_rank0.jld2", "u")
    z2 = FieldTimeSeries("mpi_hydrostatic_turbulence_rank1.jld2", "u")
    z3 = FieldTimeSeries("mpi_hydrostatic_turbulence_rank2.jld2", "u")
    z4 = FieldTimeSeries("mpi_hydrostatic_turbulence_rank3.jld2", "u")

    ζ1 = @lift(interior(z1[$iter], 1:28, 1:28, 1))
    ζ2 = @lift(interior(z2[$iter], 1:28, 1:28, 1))
    ζ3 = @lift(interior(z3[$iter], 1:28, 1:28, 1))
    ζ4 = @lift(interior(z4[$iter], 1:28, 1:28, 1))

    x1, y1 = z1.grid.xᶠᵃᵃ[1:28], z1.grid.yᵃᶜᵃ[1:28]
    x2, y2 = z4.grid.xᶠᵃᵃ[1:28], z4.grid.yᵃᶜᵃ[1:28]

    fig = Figure()
    ax = Axis(fig[1, 1])
    heatmap!(ax, x1, y1, ζ1, colorrange = (-1.0, 1.0))
    heatmap!(ax, x1, y2, ζ2, colorrange = (-1.0, 1.0))
    heatmap!(ax, x2, y1, ζ3, colorrange = (-1.0, 1.0))
    heatmap!(ax, x2, y2, ζ4, colorrange = (-1.0, 1.0))
end