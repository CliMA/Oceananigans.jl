using Oceananigans, StructArrays, Printf, JLD2, KernelAbstractions
using Oceananigans.Architectures: device
using Oceananigans.Fields: TracerFields
using Oceananigans.LagrangianParticleTracking: calculate_particle_tendency_kernel!
# This example will not neccesarily keep tracers above zero because two particles may/will take from the same cell in the same timestep

grid = RectilinearGrid(;topology = (Bounded, Bounded, Periodic), size=(10, 10, 1), extent=(1, 1, 1))
struct Particle{T}
    x :: T
    y :: T
    z :: T
    A :: T
    A_sink :: T
    B_source :: T
end

# particle randomly walks around and turns tracer a into tracer b
function dynamics!(particles, model, Δt)
    particles.properties.x .+= (-1).^rand(Bool, length(particles)).*rand(length(particles))*Δt
    particles.properties.y .+= (-1).^rand(Bool, length(particles)).*rand(length(particles))*Δt

    #particles go round and turns tracer a into b
    particles.properties.A_sink .= -particles.properties.A ./ (50 .+ particles.properties.A)
    particles.properties.B_source .= particles.properties.A ./ (50 .+ particles.properties.A)

    workgroup = min(length(particles), 256)
    worksize = length(particles)

    arch = model.grid.architecture
    
    Gp_kernel! = calculate_particle_tendency_kernel!(device(arch), workgroup, worksize)

    model.auxiliary_fields.Gₚ.A .= 0
    model.auxiliary_fields.Gₚ.B .= 0

    Gp_event_A = Gp_kernel!(particles.properties.A_sink, model.auxiliary_fields.Gₚ.A, particles, model.grid, dependencies = Event(device(arch)))
    Gp_event_B = Gp_kernel!(particles.properties.B_source, model.auxiliary_fields.Gₚ.B, particles, model.grid, dependencies = Event(device(arch)))

    events=[Gp_event_A, Gp_event_B]

    wait(device(arch), MultiEvent(Tuple(events)))
end

P=2

xs = 0.5*ones(P)
ys = 0.5*ones(P)
zs = -0.5*ones(P)
as = zeros(P)
a_sink_s = zeros(P)
b_source_s = zeros(P)

particles = StructArray{Particle}((xs, ys, zs, as, a_sink_s, b_source_s))

Gₚ = TracerFields((:A, :B), grid)

lagrangian_particles = LagrangianParticles(particles; tracked_fields=(A=nothing, ), dynamics=dynamics!)
@info "Initialized Lagrangian particles"

@inline a_sink(i, j, k, clock, grid, c) = c.Gₚ.A[i, j, k]
@inline b_source(i, j, k, clock, grid, c) = c.Gₚ.B[i, j, k]

A_forcing = Forcing(a_sink, discrete_form=true)
B_forcing = Forcing(b_source, discrete_form=true)

model = NonhydrostaticModel(grid=grid, tracers=(:A, :B), forcing = (A=A_forcing, B=B_forcing), particles=lagrangian_particles, auxiliary_fields = (; Gₚ))

set!(model, A=1)

@info "Constructed a model"

sim = Simulation(model, Δt=1e-1, stop_time=20)

#run!(sim)
