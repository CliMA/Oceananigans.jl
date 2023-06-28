using Oceananigans
using MPI
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField
using Printf
using CairoMakie

MPI.Initialized() || MPI.Init()

     comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)

@assert mpi_ranks == 16

using Statistics
using Oceananigans
using Oceananigans.Distributed

ranks = (4, 4, 1)
topo  = (Periodic, Periodic, Bounded)
arch  = DistributedArch(CPU(), ranks=ranks, topology=topo)

N = 28
nx, ny = N ÷ ranks[1], N ÷ ranks[2] 

grid  = RectilinearGrid(arch, topology=topo, size=(nx, ny, 1), extent=(4π, 4π, 0.5), halo=(3, 3, 3))

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
ζ = VerticalVorticityField(model)
outputs = merge(model.velocities, model.tracers, (; ζ))

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time), Δt: $(sim.Δt)"
simulation = Simulation(model, Δt=0.01, stop_time=100.0)

wizard = TimeStepWizard(cfl = 0.7, max_change = 1.2)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

filepath = "mpi_hydrostatic_turbulence_rank$(local_rank)"
simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs, filename=filepath, schedule=TimeInterval(0.1),
                     overwrite_existing=true)

MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)
MPI.Barrier(MPI.COMM_WORLD)

if rank == 0
    iter = Observable(1)

    vort = []
    ζ = []
    x = []
    y = []
    for i in 0:15
        push!(vort, FieldTimeSeries("mpi_hydrostatic_turbulence_rank$i.jld2", "u"))
        z1 = @lift(interior(vort[i][$iter], 1:nx, 1:ny, 1))
        push!(ζ, z1)

        push!(x, vort[i].grid.xᶠᵃᵃ[1:nx])
        push!(y, vort[i].grid.yᵃᶠᵃ[1:ny])
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    for i in 0:15
        heatmap!(ax, x[i], y[i], ζ[i], colorrange = (-1.0, 1.0))
    end

    CairoMakie.record(fig, "hydrostatic_test.mp4", iterations, framerate = 11) do i
        @info "step $i"; 
        iter[] = i; 
    end
end

MPI.Barrier(MPI.COMM_WORLD)