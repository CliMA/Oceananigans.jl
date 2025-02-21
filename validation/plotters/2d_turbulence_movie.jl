using Oceananigans
using CairoMakie

grid = RectilinearGrid(size=(32, 32), extent=(2π, 2π), topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(; grid,
                            advection = UpwindBiased(order=5))

using Statistics

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
add_callback!(simulation, Callback(wizard, IterationInterval(2)))

using Printf
progress_message(sim) = @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s\n",
                                       iteration(sim), time(sim), sim.Δt, maximum(abs, sim.model.velocities.u), prettytime(sim.run_wall_time))
add_callback!(simulation, progress_message, IterationInterval(100))


u, v, w = model.velocities
ω = ∂x(v) - ∂y(u)
s = sqrt(u^2 + v^2)

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

io = VideoStream(fig, format="mp4", framerate=12, compression=20)

function update_plot(sim, fig, io)
    heatmap!(fig[1, 1], Array(ω)[:, :, 1]; colormap = :balance, colorrange = (-2, 2))
    heatmap!(fig[1, 2], Array(s)[:, :, 1]; colormap = :balance, colorrange = (-2, 2))
    recordframe!(io)
end

maker = MovieMaker(update_plot; fig, filename="2d_turbulence.mp4", io)

# Write animation to disk after simulation is complete
import Oceananigans.Simulations: finalize!
finalize!(maker::MovieMaker, simulation) = save(maker.filename, maker.io)

add_callback!(simulation, maker, TimeInterval(0.6))

filename = "two_dimensional_turbulence"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)
run!(simulation)
