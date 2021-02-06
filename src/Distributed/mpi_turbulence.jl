include("distributed_model.jl")

using Statistics

using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

using Oceananigans.Solvers: calculate_pressure_right_hand_side!, copy_pressure!

import Oceananigans.Solvers: solve_for_pressure!

child_architecture(::CPU) = CPU()

function solve_for_pressure!(pressure, solver::DistributedFFTBasedPoissonSolver, arch, grid, Δt, U★)

    RHS = first(solver.storage)

    rhs_event = launch!(arch, grid, :xyz,
                        calculate_pressure_right_hand_side!, RHS, arch, grid, Δt, U★,
                        dependencies = Event(device(arch)))

    wait(device(arch), rhs_event)

    solve_poisson_equation!(solver)

    ϕ = first(solver.storage)

    copy_event = launch!(arch, grid, :xyz,
                         copy_pressure!, pressure, ϕ, arch, grid,
                         dependencies = Event(device(arch)))

    wait(device(arch), copy_event)

    return nothing
end

topo = (Periodic, Periodic, Periodic)
full_grid = RegularCartesianGrid(topology=topo, size=(128, 128, 1), extent=(2π, 2π, 1))
arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))
dm = DistributedModel(architecture=arch, grid=full_grid, closure=IsotropicDiffusivity(ν=1e-3))

model = dm.model
u₀ = rand(size(model.grid)...)
u₀ .-= mean(u₀)
set!(model, u=0.01u₀, v=0.01u₀)

# [time_step!(model, 0.1) for _ in 1:10]

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.1, stop_time=50, iteration_interval=1, progress=progress)

u, v, w = model.velocities
outputs = (ζ=ComputedField(∂x(v) - ∂y(u)),)
simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs, filepath="mpi_turbulence_rank$(arch.my_rank).nc", schedule=IterationInterval(1))

run!(simulation)

using Printf
using NCDatasets
using CairoMakie

if arch.my_rank == 0
    ranks = prod(arch.ranks)

    ds = [NCDataset("mpi_turbulence_rank$r.nc") for r in 0:ranks-1]

    frame = Node(1)
    title = @lift @sprintf("MPI turbulence t = %.2f", ds[1]["time"][$frame])
    ζ = [@lift ds[r]["ζ"][:, :, 1, $frame] for r in 1:ranks]

    fig = Figure(resolution=(1600, 1600))

    for r in 1:ranks
        ax = fig[0, 1] = Axis(fig, title="rank $r") # , xlabel="x", ylabel="y")
        hm = CairoMakie.heatmap!(ax, ds[r]["xC"], ds[r]["yC"], ζ[r], colormap=:balance, colorrange=(-0.01, 0.01))
        r == ranks && (cb1 = fig[:, 2] = Colorbar(fig, hm, width=30))
    end

    supertitle = fig[0, :] = Label(fig, title, textsize=30)

    record(fig, "mpi_turbulence.mp4", 1:10:length(ds[1]["time"]), framerate=15) do n
        @info "Animating MPI turbulence $var frame $n/$(length(ds[1]["time"]))..."
        frame[] = n
    end

    [close(d) for d in ds]
end
