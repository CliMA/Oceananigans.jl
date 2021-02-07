include("distributed_model.jl")

using MPI

MPI.Initialized() || MPI.Init()

using Statistics

using Oceananigans.Advection
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations
using Oceananigans.Utils

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
full_grid = RegularCartesianGrid(topology=topo, size=(512, 512, 1), extent=(4π, 4π, 1), halo=(3, 3, 3))
arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))

dm = DistributedModel(
    architecture = arch,
            grid = full_grid,
     timestepper = :RungeKutta3,
       advection = WENO5(),
         closure = IsotropicDiffusivity(ν=1e-5)
)

model = dm.model
u₀ = rand(size(model.grid)...);
u₀ .-= mean(u₀);
set!(model, u=u₀, v=u₀)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.05, stop_time=50, iteration_interval=1, progress=progress)

u, v, w = model.velocities
outputs = (ζ=ComputedField(∂x(v) - ∂y(u)),)
simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs, filepath="mpi_turbulence_rank$(arch.my_rank).nc", schedule=TimeInterval(0.1))

MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)

using Printf
using NCDatasets
using CairoMakie

if arch.my_rank == 0
    ranks = 4

    ds = [NCDataset("mpi_turbulence_rank$r.nc") for r in 0:ranks-1]

    frame = Node(1)
    plot_title = @lift @sprintf("Oceananigans.jl + MPI: 2D turbulence t = %.1f", ds[1]["time"][$frame])
    ζ = [@lift ds[r]["ζ"][:, :, 1, $frame] for r in 1:ranks]

    fig = Figure(resolution=(1600, 1200))

    for r in reverse(1:ranks)
        ax = fig[ranks-r+1, 1] = Axis(fig, ylabel="rank $(r-1)", xticks = MultiplesTicks(9, pi, "π"),  yticks = MultiplesTicks(3, pi, "π"))
        hm = CairoMakie.heatmap!(ax, ds[r]["xF"], ds[r]["yF"], ζ[r], colormap=:balance, colorrange=(-2, 2))
        r > 1 && hidexdecorations!(ax, grid=false)
        if r == 1
            cb = fig[:, 2] = Colorbar(fig, hm, label = "Vorticity ζ = ∂x(v) - ∂y(u)", width=30)
            cb.height = Relative(2/3)
        end
        xlims!(ax, [0, 4π])
        ylims!(ax, [(r-1)*π, r*π])
    end

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

    trim!(fig.layout)

    record(fig, "mpi_turbulence.mp4", 1:length(ds[1]["time"])-1, framerate=30) do n
        @info "Animating MPI turbulence frame $n/$(length(ds[1]["time"]))..."
        frame[] = n
    end

    [close(d) for d in ds]
end
