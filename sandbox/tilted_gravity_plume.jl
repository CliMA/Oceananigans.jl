using Oceananigans
using Oceananigans.Units

N = 64
L = 2000
topo = (Periodic, Bounded, Bounded)
grid = RegularRectilinearGrid(topology=topo, size=(1, N, N), extent=(L, L, L))

use_buoyancy_tracer = false
θ = 45

g̃ = (0, sind(θ), cosd(θ))
if use_buoyancy_tracer
    buoyancy = Buoyancy(model=BuoyancyTracer(), gravitational_unit_vector=g̃)
    tracers = :b
else
    buoyancy = Buoyancy(model=SeawaterBuoyancy(), gravitational_unit_vector=g̃)
    tracers = :T, :S
end

model = IncompressibleModel(
           grid = grid,
    timestepper = :RungeKutta3,
       buoyancy = buoyancy,
        tracers = tracers,
      advection = UpwindBiasedFifthOrder(),
        closure = IsotropicDiffusivity(ν=0, κ=0)
)

y₀, z₀ = L/2, -L/2
anomaly(x, y, z) = 0.01 * exp(-100 * ((y - y₀)^2 + (z - z₀)^2) / (L^2 + L^2))

if use_buoyancy_tracer
    α = 2e-4; g₀ = 9.81
    b₀(x, y, z) = α * g₀ * anomaly(x, y, z)
    set!(model, b=b₀)
else
    T₀(x, y, z) = 10 + anomaly(x, y, z)
    set!(model, T=T₀)
end

print_progress(sim) = @info "iteration: $(sim.model.clock.iteration), time: $(sim.model.clock.time)"
simulation = Simulation(model, Δt=10seconds, stop_time=4hours, progress=print_progress, iteration_interval=10)

fields = merge(model.velocities, model.tracers)
simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filepath = "tilted_gravity_plume.nc",
                       schedule = TimeInterval(5minutes),
                       mode = "c")

run!(simulation)

pause
using NCDatasets
using Plots

ds = NCDataset("tilted_gravity_plume.nc", "r")

Nt = length(ds["time"])
anim = @animate for n in 1:Nt
    @info "frame $n/$Nt"
    heatmap(ds["yC"], ds["zC"], ds["b"][1, :, :, n]', clims=(20, 20.01))
end
gif(anim, "tilted_gravity_plume.gif")
close(ds)
