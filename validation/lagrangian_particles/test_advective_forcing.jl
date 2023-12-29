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
using Oceananigans.Fields: VelocityFields
using Oceananigans.Architectures: device, architecture
using CairoMakie
using KernelAbstractions

Random.seed!(123)

grid = RectilinearGrid(Oceananigans.GPU(), Float64,
                       size = (4, 4, 4),
                       halo = (5, 5, 5),
                       x = (0, 1),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

b_initial(x, y, z) = 1e-3 * rand()

#%%
struct LagrangianPOC{T, V}
    x :: T
    y :: T
    z :: T
    u :: V
    v :: V
    w :: V
    u_particle :: V
    v_particle :: V
    w_particle :: V
end

n_particles = 3

x₀ = CuArray(zeros(n_particles))
y₀ = CuArray(rand(n_particles))
z₀ = CuArray(-0.1 * rand(n_particles))

u₀ = CuArray(zeros(n_particles))
v₀ = CuArray(zeros(n_particles))
w₀ = CuArray(zeros(n_particles))

u₀_particle = deepcopy(u₀)
v₀_particle = deepcopy(v₀)
w₀_particle = CuArray(-1e-5 * rand(n_particles))

# x₀ = zeros(n_particles)
# y₀ = rand(n_particles)
# z₀ = -0.1 * rand(n_particles)

# u₀ = zeros(n_particles)
# v₀ = zeros(n_particles)
# w₀ = zeros(n_particles)

# u₀_particle = deepcopy(u₀)
# v₀_particle = deepcopy(v₀)
# w₀_particle = -1e-5 * rand(n_particles)

particles = StructArray{LagrangianPOC}((x₀, y₀, z₀, u₀, v₀, w₀, u₀_particle, v₀_particle, w₀_particle))

@inline function w_sinking(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    return particles[p].w_particle
end

@inline function sinking_dynamics(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    w₀ = particles[p].w_particle
    w = w₀ + 1 / (2 * 24 * 60^2) * (w_fluid - w₀) * Δt
    return w
end

w_forcing  = ParticleDiscreteForcing(w_sinking)
sinking = ParticleVelocities(w=w_forcing)

@kernel function update_particle_velocities!(particles, advective_velocity::ParticleVelocities, grid, clock, Δt, model_fields)
    p = @index(Global)
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]

        u_fluid = particles.u[p]
        v_fluid = particles.v[p]
        w_fluid = particles.w[p]

        particles.u_particle[p] = u_fluid
        particles.v_particle[p] = v_fluid
        particles.w_particle[p] = sinking_dynamics(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    end
end

function update_lagrangian_particle_velocities!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)
    model_fields = merge(model.velocities, model.tracers, model.auxiliary_fields)

    update_particle_velocities_kernel! = update_particle_velocities!(device(arch), workgroup, worksize)
    update_particle_velocities_kernel!(particles.properties, particles.advective_velocity, model.grid, model.clock, Δt, model_fields)

    return nothing
end
velocities = VelocityFields(grid)

lagrangian_particles = LagrangianParticles(particles, advective_velocity=sinking, tracked_fields=velocities, dynamics=update_lagrangian_particle_velocities!)

#%%
model = NonhydrostaticModel(; 
            grid = grid,
            velocities = velocities,
            closure = ScalarDiffusivity(ν=1e-5, κ=1e-5),
            coriolis = FPlane(f=1e-4),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            particles = lagrangian_particles
            )

set!(model, b=b_initial)

b = model.tracers.b
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1seconds, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=60seconds, cfl=0.6)
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

simulation.output_writers[:particles] = JLD2OutputWriter(model, (; model.particles),
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
    lines!(ax, times, [data.z[i] for data in particle_data])
end
display(fig)
#%%