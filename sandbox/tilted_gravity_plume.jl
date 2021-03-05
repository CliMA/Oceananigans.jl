using Oceananigans
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils
using Oceananigans.Units

N = 64
L = 2000
topo = (Periodic, Bounded, Bounded)
grid = RegularRectilinearGrid(topology=topo, size=(1, N, N), extent=(L, L, L))

g_Earth = 9.80665
θ = 45
g = (0, g_Earth * sind(θ), g_Earth * cosd(θ))
buoyancy = SeawaterBuoyancy(gravitational_acceleration=g)

model = IncompressibleModel(
    grid = grid,
    timestepper = :RungeKutta3,
    buoyancy = buoyancy,
    advection = WENO5(),
    closure = IsotropicDiffusivity(ν=0, κ=0)
)

y₀, z₀ = L/2, -L/2
T₀(x, y, z) = 20 + 0.01 * exp(-100 * ((y - y₀)^2 + (z - z₀)^2) / (L^2 + L^2))
set!(model, T=T₀)

function print_progress(sim)
    @info "iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
    if any(isnan, model.velocities.w.data.parent)
        error("NaN found in w!")
    end
    return nothing
end
simulation = Simulation(model, Δt=10seconds, stop_time=4hours, progress=print_progress, iteration_interval=10)

fields = merge(model.velocities, model.tracers)
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filepath="tilted_gravity_plume.nc",
                       schedule=TimeInterval(5minutes),
                       mode="c",
                       )

run!(simulation)

pause
using NCDatasets, Plots

ds = NCDataset("tilted_gravity_plume.nc", "r")

Nt = length(ds["time"])
anim = @animate for n in 1:Nt
    @info "frame $n/$Nt"
    heatmap(ds["yC"], ds["zC"], ds["T"][1, :, :, n]', clims=(20, 20.01))
end
gif(anim, "tilted_gravity_plume.gif")
close(ds)
