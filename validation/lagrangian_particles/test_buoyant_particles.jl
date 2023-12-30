using Oceananigans
using Oceananigans.Units
using StructArrays
using JLD2
using FileIO
using Printf
using Random
using Statistics
using Dates
using CUDA: CuArray
using Oceananigans.Models.LagrangianParticleTracking: ParticleVelocities, ParticleDiscreteForcing
using Oceananigans.Fields: TracerFields
using Oceananigans.Architectures: device, architecture
using CairoMakie
using KernelAbstractions

Random.seed!(123)

grid = RectilinearGrid(Oceananigans.GPU(), Float64,
                       size = (10, 10, 10),
                       halo = (5, 5, 5),
                       x = (0, 1),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

b_initial(x, y, z) = z

#%%
struct LagrangianPOC{T, B, BF}
    x :: T
    y :: T
    z :: T
    buoyancy :: B
    b :: BF
end

n_particles = 3

x₀ = CuArray(rand(n_particles))
y₀ = CuArray(rand(n_particles))
z₀ = CuArray(-rand(n_particles))

buoyancy₀ = CuArray(-rand(n_particles))
b₀ = CuArray(zeros(n_particles))

particles = StructArray{LagrangianPOC}((x₀, y₀, z₀, buoyancy₀, b₀))

@inline function w_buoyant(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    @inbounds begin
        b_fluid = particles[p].b
        b_particle = particles[p].buoyancy

    end
    return 1e-4 * (b_particle - b_fluid)
end

w_forcing  = ParticleDiscreteForcing(w_buoyant)
w_particle = ParticleVelocities(w=w_forcing)

tracers = TracerFields([:b], grid)

lagrangian_particles = LagrangianParticles(particles, advective_velocity=w_particle, tracked_fields=tracers)

#%%
model = NonhydrostaticModel(; 
            grid = grid,
            coriolis = FPlane(f=1e-4),
            buoyancy = BuoyancyTracer(),
            tracers = tracers,
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            particles = lagrangian_particles
            )

set!(model, b=b_initial)

simulation = Simulation(model, Δt=1e-2seconds, stop_time=0.5days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=10seconds, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, (<x>, <y>, <z>) (%6.3e, %6.3e, %6.3e), next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            mean(lagrangian_particles.properties.x),
            mean(lagrangian_particles.properties.y),
            mean(lagrangian_particles.properties.z),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

particle_outputs = (; model.particles)

simulation.output_writers[:particles] = JLD2OutputWriter(model, particle_outputs,
                                                          filename = "./particles.jld2",
                                                          schedule = TimeInterval(60seconds),
                                                          with_halos = true,
                                                          overwrite_existing = true)

run!(simulation)

#%%
times, particle_data = jldopen("./particles.jld2", "r") do file
    iters = keys(file["timeseries/t"])
    times = [file["timeseries/t/$(iter)"] for iter in iters]
    particle_timeseries = [file["timeseries/particles/$(iter)"] for iter in iters]
    return times, particle_timeseries
end

#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="z")
for i in 1:n_particles
    lines!(ax, times, [data.z[i] for data in particle_data], label="buoyancy = $(round(Array(buoyancy₀)[i], digits=2))")
end
axislegend(ax)
display(fig)
#%%