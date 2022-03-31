using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition
using Printf
using GLMakie

# Buoyancy profile
bᵢ(x, y, z) = 1e-5 * z
b_particle = bᵢ(0, 0, 0) # particle buoyancy
ν = 1e-4
κ_particle = 1e-3
R_particle = 1.0
Δt = 1e-2

grid = RectilinearGrid(size=128, z=(-4, 4), topology=(Flat, Flat, Bounded))

no_penetration = ImpenetrableBoundaryCondition()
slip_bcs = FieldBoundaryConditions(grid, (Center, Center, Face),
                                   top=no_penetration, bottom=no_penetration)
w_slip = ZFaceField(grid, boundary_conditions=slip_bcs)
sinking = AdvectiveForcing(WENO5(), w=w_slip)

model = NonhydrostaticModel(; grid,
                            tracers = (:b, :P),
                            buoyancy = BuoyancyTracer(),
                            closure = ScalarDiffusivity(κ=(b=0.0, P=κ_particle)),
                            forcing = (; P = sinking))

Pᵢ(x, y, z) = exp(-(z + 1)^2)
set!(model, b=bᵢ, P=Pᵢ)

simulation = Simulation(model; Δt, stop_iteration=0)

b = model.tracers.b
w_slip_op = 2/9 * (b - b_particle) / ν * R_particle^2

function compute_slip_velocity!(sim)
    w_slip .= w_slip_op
    fill_halo_regions!(w_slip)
    return nothing
end

simulation.callbacks[:slip] = Callback(compute_slip_velocity!)

progress(sim) = @info @sprintf("Iter: %d, time: %.2e, max|w_slip|: %.2e",
                               iteration(sim), time(sim), maximum(abs, w_slip))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

z = znodes(Center, grid)
P = interior(model.tracers.P, 1, 1, :)

fig = Figure()
ax = Axis(fig[1, 1],
          xlabel = "Particle concentration",
          ylabel = "z",
          title = "Settling diffusion particle concentration at t=0")
xlims!(ax, -1, 4)

ℓ = lines!(ax, P, z)
display(fig)

function update_plot!(sim)
    ℓ.input_args[1][] = interior(sim.model.tracers.P, 1, 1, :)
    ax.title[] = @sprintf("Particle concentration at t=%.2e", time(sim))
end

#simulation.callbacks[:plot] = Callback(update_plot!, IterationInterval(1000))

record(fig, "settling_diffusion.mp4", 1:100, framerate=24) do nn
    simulation.stop_iteration += 200
    run!(simulation)
    update_plot!(simulation)
end

