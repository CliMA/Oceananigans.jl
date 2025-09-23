using Oceananigans
using Printf
using CairoMakie
using NCDatasets
using Statistics

grid = RectilinearGrid(CPU(),
                      size=(32, 32), halo=(7, 7),
                      y = (0, 1),
                      z = (0, 1),
                      topology=(Flat, Bounded, Bounded))


const N² = 1.0
const fz = 0.1
const fy = 0.1
const β  = 0.0
const γ  = 0.0
const radius = 1.0

const shear_y = 1e-2
const shear_z = 1e-2

B(y, z) = -(shear_y*fz + shear_z*fy)*y + N²*z
U(y, z) =  (shear_y*y  + shear_z*z)

coriolis = NonTraditionalBetaPlane(fz=fz, fy=fy, β=β, γ=γ, radius=radius)
model = HydrostaticFreeSurfaceModel(; grid,
                              coriolis = coriolis,
                              buoyancy = BuoyancyTracer(),
                              momentum_advection = WENO(),
                              tracer_advection = WENO(),
                              tracers = (:b,))
set!(model, u = U, b = B)

xu, yu, zu = nodes(model.velocities.u)
xb, yb, zb = nodes(model.tracers.b)

function progress(sim)
    umax = maximum(abs, sim.model.velocities.u)
    bmax = maximum(sim.model.tracers.b)
    @info @sprintf("Iter: %d, time: %.2e, max|u|: %.2e. max b: %.2e",
       iteration(sim), time(sim), umax, bmax)

    return nothing
end

simulation = Simulation(model; Δt=1e-3, stop_time=10.0)
simulation.callbacks[:p] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities
b = model.tracers.b
outputs = (; u, b)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs;
                                                        filename = "linear_shear.nc",
                                                        schedule = TimeInterval(0.1),
                                                        overwrite_existing = true)

run!(simulation)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

fig = Figure(size = (800, 600))

axis_kwargs = (xlabel = "y",
               ylabel = "z",
               limits = ((0, grid.Lx), (0, grid.Lz)),
               )

ax_u = Axis(fig[2, 1]; title = "meridional velocity", axis_kwargs...)
ax_b = Axis(fig[3, 1]; title = "buoyancy - N²*z", axis_kwargs...)

n = Observable(1)

u = @lift ds["u"][:, :, $n]
v = @lift ds["b"][:, :, $n] - N²*zb

hm_u = heatmap!(ax_u, yu, zu, u, colormap = :balance)
Colorbar(fig[2, 2], hm_u)

hm_b = heatmap!(ax_b, yb, zb, b, colormap = :balance)
Colorbar(fig[3, 2], hm_b)

times = collect(ds["time"])
title = @lift "t = " * string(prettytime(times[$n]))
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

fig

frames = 1:length(times)

record(fig, "linear_shear_nontraditional.mp4", frames, framerate=12) do i
    n[] = i
end

close(ds)

