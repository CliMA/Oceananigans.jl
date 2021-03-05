using MPI

MPI.Initialized() || MPI.Init()

using Statistics
using Oceananigans
using Oceananigans.Distributed

topo = (Periodic, Periodic, Bounded)
full_grid = RegularRectilinearGrid(topology=topo, size=(128, 128, 1), extent=(4π, 4π, 1), halo=(3, 3, 3))
arch = MultiCPU(grid=full_grid, ranks=(1, 4, 1))

model = DistributedShallowWaterModel(
                  architecture = arch,
                          grid = full_grid,
                   timestepper = :RungeKutta3,
                     advection = UpwindBiasedFifthOrder(),
                       closure = IsotropicDiffusivity(ν=1e-5),
    gravitational_acceleration = 1.0
)

set!(model, h=model.grid.Lz)

uh₀ = rand(size(model.grid)...);
uh₀ .-= mean(uh₀);
set!(model, uh=uh₀, vh=uh₀)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.001, stop_time=2, iteration_interval=1, progress=progress)

uh, vh, h = model.solution
outputs = (ζ=ComputedField(∂x(vh/h) - ∂y(uh/h)),)
filepath = "mpi_shallow_water_turbulence_rank$(arch.my_rank).nc"
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath=filepath, schedule=TimeInterval(0.01), mode="c")

MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)

using Printf
using NCDatasets
using CairoMakie

if arch.my_rank == 0
    ranks = 4

    ds = [NCDataset("mpi_shallow_water_turbulence_rank$r.nc") for r in 0:ranks-1]

    frame = Node(1)
    plot_title = @lift @sprintf("Oceananigans.jl + MPI: 2D turbulence t = %.2f", ds[1]["time"][$frame])
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

    record(fig, "mpi_shallow_water_turbulence.mp4", 1:length(ds[1]["time"])-1, framerate=30) do n
        @info "Animating MPI turbulence frame $n/$(length(ds[1]["time"]))..."
        frame[] = n
    end

    [close(d) for d in ds]
end
