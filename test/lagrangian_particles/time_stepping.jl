include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using StructArrays
using Oceananigans.Architectures: on_architecture
using Oceananigans.Models.LagrangianParticleTracking: ParticleTendencies

#####
##### Convergence of particle trajectories
#####

# Particles in solid body rotation, x = r cos(Ω t), y = r sin(Ω t), prescribed by background
# velocity fields that are linear in space, so that interpolation to particle positions is exact.
const Ω = 1

rotation_u(x, y, z, t) = - Ω * y
rotation_v(x, y, z, t) = + Ω * x

# NonhydrostaticModel supports RungeKutta3 and QuasiAdamsBashforth2, while
# HydrostaticFreeSurfaceModel supports SplitRungeKutta3 and QuasiAdamsBashforth2.
# The fields of the HydrostaticFreeSurfaceModel include a passive tracer, so that particles
# are advanced with the scheme of the model rather than with a forward Euler step.
function particle_test_model(grid, timestepper, particles, velocities)
    if timestepper == :SplitRungeKutta3
        return HydrostaticFreeSurfaceModel(grid; particles, timestepper, tracers=:c, buoyancy=nothing,
                                           velocities=PrescribedVelocityFields(; velocities...))
    else
        return NonhydrostaticModel(grid; particles, timestepper, advection=nothing, background_fields=velocities)
    end
end

function rotating_particle_error(arch, timestepper, Nt; stop_time = 1)
    grid = RectilinearGrid(arch; size=(4, 4, 1), x=(-2, 2), y=(-2, 2), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    x, y, z = on_architecture.(Ref(arch), ([1.0], [0.0], [0.5]))
    particles = LagrangianParticles(; x, y, z)
    model = particle_test_model(grid, timestepper, particles, (u = rotation_u, v = rotation_v))

    Δt = stop_time / Nt
    for _ in 1:Nt
        time_step!(model, Δt)
    end

    xᵖ = Array(model.particles.properties.x)[1]
    yᵖ = Array(model.particles.properties.y)[1]

    return sqrt((xᵖ - cos(Ω * stop_time))^2 + (yᵖ - sin(Ω * stop_time))^2)
end

convergence_rate(errors) = log2(errors[end-1] / errors[end])

@testset "Convergence of Lagrangian particle trajectories [$(typeof(arch))]" for arch in archs
    Nts = (16, 32, 64)

    # χ = 0 gives the second-order Adams-Bashforth scheme
    timesteppers_and_orders = ((:RungeKutta3, 3),
                               (QuasiAdamsBashforth2TimeStepper(χ=0.0), 2),
                               (:SplitRungeKutta3, 3))

    for (timestepper, order) in timesteppers_and_orders
        errors = [rotating_particle_error(arch, timestepper, Nt) for Nt in Nts]
        rate = convergence_rate(errors)
        @info "    Particle trajectory convergence with $(summary(timestepper)): errors = $errors, rate = $rate"
        @test abs(rate - order) < 0.2
    end
end

#####
##### Per-particle tendencies stays aligned with the particles
#####

function labeled_particles(arch, Np; seed=1234)
    rng = Xoshiro(seed)
    values = (x = rand(rng, Np), y = rand(rng, Np), z = rand(rng, Np), label = collect(1:Np))
    return StructArray(on_architecture(arch, values))
end

shear_u(x, y, z, t) = 0.3 * sin(2π * y) + 0.1 * z
shear_v(x, y, z, t) = 0.2 * cos(2π * x)
sinking_w(x, y, z, t) = - 0.2 - 0.1 * sin(2π * x) * cos(2π * y)

const shear_background_fields = (u = shear_u, v = shear_v, w = sinking_w)

function shear_flow_model(arch, timestepper, particles)
    grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    return particle_test_model(grid, timestepper, particles, shear_background_fields)
end

function particles_by_label(particles)
    properties = on_architecture(CPU(), particles.properties)
    label = Array(properties.label)
    order = sortperm(label)
    return (label = label[order],
            x = Array(properties.x)[order],
            y = Array(properties.y)[order],
            z = Array(properties.z)[order])
end

@testset "Sorting and removing Lagrangian particles during time stepping [$(typeof(arch)), $timestepper]" for arch in archs,
        timestepper in (:RungeKutta3, :QuasiAdamsBashforth2, :SplitRungeKutta3)

    Np = 200
    Nt = 10
    Δt = 0.1

    # Reference: no sorting, and particles reflect off the bottom
    reference = LagrangianParticles(labeled_particles(arch, Np))
    reference_model = shear_flow_model(arch, timestepper, reference)

    # Sorted particles that are absorbed at the bottom, and removed mid-run
    boundary_conditions = ParticleBoundaryConditions(bottom=AbsorbingBoundary())
    particles = LagrangianParticles(labeled_particles(arch, Np); sorting=CountingSort(), boundary_conditions)
    model = shear_flow_model(arch, timestepper, particles)

    @test model.particles.tendencies isa ParticleTendencies
    @test isnothing(particles.tendencies) # the particle tendencies is materialized by the model

    Np_history = Int[]

    for _ in 1:Nt
        time_step!(reference_model, Δt)
        time_step!(model, Δt)
        push!(Np_history, length(model.particles))
        @test all(array -> length(array) == length(model.particles), model.particles.tendencies.state)
    end

    # Particles are removed at several different time steps
    @test length(unique(Np_history)) > 2
    @test 0 < last(Np_history) < Np

    # Surviving particles never reach the bottom, so their trajectories match the reference
    survivors = particles_by_label(model.particles)
    expected = particles_by_label(reference_model.particles)
    kept = survivors.label

    @test expected.x[kept] ≈ survivors.x
    @test expected.y[kept] ≈ survivors.y
    @test expected.z[kept] ≈ survivors.z
end

#####
##### Checkpointing particle tendencies
#####

@testset "Checkpointing Lagrangian particle particle tendencies [$(typeof(arch))]" for arch in archs
    Np = 100
    Δt = 0.05
    boundary_conditions = ParticleBoundaryConditions(bottom=AbsorbingBoundary())
    timestepper = :QuasiAdamsBashforth2

    make_model() = shear_flow_model(arch, timestepper,
                                    LagrangianParticles(labeled_particles(arch, Np); boundary_conditions, sorting=CountingSort()))

    reference_model = make_model()
    reference_simulation = Simulation(reference_model; Δt, stop_iteration=10, verbose=false)
    run!(reference_simulation)

    dir = mktempdir()
    prefix = "particle_tendencies_$(typeof(arch))"

    model = make_model()
    simulation = Simulation(model; Δt, stop_iteration=5, verbose=false)
    simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=IterationInterval(5), dir, prefix)
    run!(simulation)

    # Some particles were removed before the checkpoint
    Np_checkpoint = length(model.particles)
    @test Np_checkpoint < Np

    state = Oceananigans.prognostic_state(model.particles)
    @test keys(state.tendencies) == (:u⁻, :v⁻, :w⁻)
    @test all(array -> length(array) == Np_checkpoint, state.tendencies)

    # Round trip through prognostic_state / restore_prognostic_state!
    new_model = make_model()
    Oceananigans.restore_prognostic_state!(new_model.particles, on_architecture(CPU(), state))
    @test length(new_model.particles) == Np_checkpoint
    for name in keys(state.tendencies)
        @test Array(new_model.particles.tendencies.state[name]) == Array(state.tendencies[name])
    end

    # Restarting from the checkpoint reproduces the reference run
    new_model = make_model()
    new_simulation = Simulation(new_model; Δt, stop_iteration=10, verbose=false)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model; schedule=IterationInterval(5), dir, prefix)
    set!(new_simulation; checkpoint=:latest)
    @test length(new_model.particles) == Np_checkpoint
    run!(new_simulation)

    restarted = particles_by_label(new_model.particles)
    expected = particles_by_label(reference_model.particles)
    @test restarted.label == expected.label
    @test restarted.x ≈ expected.x
    @test restarted.y ≈ expected.y
    @test restarted.z ≈ expected.z

    rm(dir; recursive=true)
end
