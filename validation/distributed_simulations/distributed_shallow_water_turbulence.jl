using MPI

comm = MPI.COMM_WORLD
mpi_ranks = MPI.Comm_size(comm)

@assert mpi_ranks == 4

using Statistics
using Oceananigans
using Oceananigans.DistributedComputations

ranks = (2, 2, 1)
topo = (Periodic, Periodic, Flat)
arch = Distributed(CPU(), ranks=ranks, topology=topo)
grid = RectilinearGrid(arch, topology=topo, size=(128 ÷ ranks[1], 128 ÷ ranks[2]), extent=(4π, 4π), halo=(3, 3))
local_rank = MPI.Comm_rank(MPI.COMM_WORLD)

model = ShallowWaterModel(grid = grid,
                          timestepper = :RungeKutta3,
                          momentum_advection = UpwindBiasedFifthOrder(),
                          gravitational_acceleration = 1)

set!(model, h=1)

uh₀ = rand(size(model.grid)...);
uh₀ .-= mean(uh₀);
set!(model, uh=uh₀, vh=uh₀)

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=0.001, stop_time=100.0)

uh, vh, h = model.solution
outputs = (ζ=Field(∂x(vh/h) - ∂y(uh/h)),)
filepath = "mpi_shallow_water_turbulence_rank$(local_rank).nc"
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, outputs, filepath=filepath, schedule=TimeInterval(1.0), mode="c")

MPI.Barrier(MPI.COMM_WORLD)

run!(simulation)

using Printf
using NCDatasets
using CairoMakie

nranks = prod(ranks)

if local_rank == 0

    ds = [NCDataset("mpi_shallow_water_turbulence_rank$r.nc") for r in 0:nranks-1]

    frame = Node(1)
    plot_title = @lift @sprintf("Oceananigans.jl + MPI: 2D turbulence t = %.2f", ds[1]["time"][$frame])
    ζ = [@lift ds[r]["ζ"][:, :, 1, $frame] for r in 1:nranks]
    
    fig = Figure(resolution=(1600, 1200))
    
    for rx in 1:ranks[1], ry in 1:ranks[2]
        ax = fig[rx, ry] = Axis(fig)
        r = (ry-1)*ranks[2] + rx
        hm = CairoMakie.heatmap!(ax, ds[r]["xF"], ds[r]["yF"], ζ[r], colormap=:balance, colorrange=(-2, 2))
    end
    
    record(fig, "mpi_shallow_water_turbulence.mp4", 1:length(ds[1]["time"])-1, framerate=30) do n
        @info "Animating MPI turbulence frame $n/$(length(ds[1]["time"]))..."
        frame[] = n
    end
    
    [close(d) for d in ds]
end
