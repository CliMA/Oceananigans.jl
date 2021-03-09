#using MPI

#MPI.Initialized() || MPI.Init()

using Printf
using NCDatasets
using CairoMakie

ranks = (2,2,1)
nranks = prod(ranks)

ds = [NCDataset("mpi_shallow_water_turbulence_rank$r.nc") for r in 0:nranks-1]

frame = Node(1)
plot_title = @lift @sprintf("Oceananigans.jl + MPI: 2D turbulence t = %.2f", ds[1]["time"][$frame])
ζ = [@lift ds[r]["ζ"][:, :, 1, $frame] for r in 1:nranks]

fig = Figure(resolution=(1600, 1200))

for rx in 1:ranks[1], ry in 1:ranks[2]
    ax = fig[ry, rx] = Axis(fig)
    r = (ry-1)*ranks[2] + rx - 1 + 1
    hm = CairoMakie.heatmap!(ax, ds[r]["xF"], ds[r]["yF"], ζ[r], colormap=:balance, colorrange=(-2, 2))
    #r > 1 && hidexdecorations!(ax, grid=false)
    #if r == 1
    #    cb = fig[:, 5] = Colorbar(fig, hm, label = "Vorticity ζ = ∂x(v) - ∂y(u)", width=30)
    #    cb.height = Relative(2/3)
    #end
    #xlims!(ax, [(r-1)*π, r*π])
    #ylims!(ax, [0, 4π])
end

#supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

#trim!(fig.layout)

record(fig, "mpi_shallow_water_turbulence.mp4", 1:length(ds[1]["time"])-1, framerate=30) do n
    @info "Animating MPI turbulence frame $n/$(length(ds[1]["time"]))..."
    frame[] = n
end

[close(d) for d in ds]