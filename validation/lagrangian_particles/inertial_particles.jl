using StructArrays
using Oceananigans
using Oceananigans: architecture
using Oceananigans.Models.LagrangianParticleTracking: AbstractParticle
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!
import Oceananigans.Models.LagrangianParticleTracking: particle_u_velocity, particle_v_velocity, particle_w_velocity

struct InertialParticle{T} <: AbstractParticle
    x :: T
    y :: T
    z :: T
    u :: T
    v :: T
    w :: T
    particle_respose_time :: T
end

# 10 Particles with different inertia
x = ones(10)
y = ones(10)
z = ones(10)
u = zeros(10)
v = zeros(10)
w = zeros(10)

particle_respose_time = range(0.1, 1.0, length = 10)

properties = StructArray{InertialParticle}((x, y, z, u, v, w, particle_respose_time))
particles  = LagrangianParticles(properties)

grid = RectilinearGrid(size = (50, 50, 50), x = (0, 2), y = (0, 2), z = (0, 2), topology = (Periodic, Periodic, Periodic))

u_fluid = XFaceField(grid)
v_fluid = YFaceField(grid)
w_fluid = ZFaceField(grid)

@inline particles_u_velocity(u_fluid, particle, Δt) = particle.u + Δt / particles.particle_respose_time * (u_fluid - particle.u)
@inline particles_v_velocity(v_fluid, particle, Δt) = particle.v + Δt / particles.particle_respose_time * (v_fluid - particle.v)
@inline particles_w_velocity(w_fluid, particle, Δt) = particle.w + Δt / particles.particle_respose_time * (w_fluid - particle.w)

set!(u_fluid, (x, y, z) -> rand())
set!(v_fluid, (x, y, z) -> rand())

fill_halo_regions!((u_fluid, v_fluid))

compute_w_from_continuity!((; u = u_fluid, v = v_fluid, w = w_fluid), architecture(grid), grid)

velocities = PrescribedVelocityFields(; u = u_fluid, v = v_fluid, w = w_fluid)

model = HydrostaticFreeSurfaceModel(; grid, 
                                      tracers = (),
                                      buoyancy = nothing,
                                      particles,
                                      velocities)

simulation = Simulation(model, Δt = 1e-2, stop_time = 10)

particles_save = [deepcopy(properties)]

save_particles(sim) = 
    push!(particles_save, deepcopy(sim.model.particles.properties))

simulation.callbacks[:particles] = Callback(save_particles, IterationInterval(10))

run!(simulation)

