using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition
using Printf
using GLMakie

grid = RectilinearGrid(size=128, z=(-10, 10), topology=(Flat, Flat, Bounded))
sinking = AdvectiveForcing(WENO5(), w=-1)
rising = AdvectiveForcing(WENO5(), w=+1)

b_to_a(x, y, z, t, a, b) = + a * b
a_to_b(x, y, z, t, a, b) = - a * b 

a_reaction = Forcing(a_to_b, field_dependencies=(:a, :b))
b_reaction = Forcing(b_to_a, field_dependencies=(:a, :b))

model = HydrostaticFreeSurfaceModel(; grid,
                                    velocities = nothing,
                                    tracers = (:a, :b),
                                    buoyancy = nothing,
                                    closure = ScalarDiffusivity(κ=1e-2),
                                    forcing = (a=(a_reaction, sinking), b=(b_reaction, rising)))

aᵢ(x, y, z) = exp(-(z - 4)^2)
bᵢ(x, y, z) = exp(-(z + 4)^2)
set!(model, a=aᵢ, b=bᵢ)

simulation = Simulation(model; Δt=1e-2, stop_iteration=0)
progress(sim) = @info @sprintf("Iter: %d, time: %.2e", iteration(sim), time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

z = znodes(Center, grid)
a = interior(model.tracers.a, 1, 1, :)
b = interior(model.tracers.b, 1, 1, :)

fig = Figure()
ax = Axis(fig[1, 1],
          xlabel = "Reactant concentration",
          ylabel = "z",
          title = "Tracer reactions at t=0")
xlims!(ax, -1, 4)

ℓa = lines!(ax, a, z, label="a")
ℓb = lines!(ax, b, z, label="b")

axislegend(ax, position=:rb)

display(fig)

function update_plot!(sim)
    ℓa.input_args[1][] = interior(sim.model.tracers.a, 1, 1, :)
    ℓb.input_args[1][] = interior(sim.model.tracers.b, 1, 1, :)
    ax.title[] = @sprintf("Tracer reactions at t=%.2e", time(sim))
end

record(fig, "tracer_reactions.mp4", 1:100, framerate=24) do nn
    simulation.stop_iteration += 10
    run!(simulation)
    update_plot!(simulation)
end

