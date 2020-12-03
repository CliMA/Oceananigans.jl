using NCDatasets
using StructArrays

struct TestParticle{T}
    x :: T
    y :: T
    z :: T
    u :: T
    v :: T
    w :: T
    s :: T
end

function run_simple_particle_tracking_tests(arch, timestepper)
    topo = (Periodic, Periodic, Bounded)
    domain = (x=(-1, 1), y=(-1, 1), z=(-1, 1))
    grid = RegularCartesianGrid(topology=topo, size=(5, 5, 5); domain...)

    P = 10
    xs = convert(array_type(arch), zeros(P))
    ys = convert(array_type(arch), zeros(P))
    zs = convert(array_type(arch), 0.5*ones(P))

    # Test first constructor
    particles = LagrangianParticles(x=xs, y=ys, z=zs)
    @test particles isa LagrangianParticles

    us = convert(array_type(arch), zeros(P))
    vs = convert(array_type(arch), zeros(P))
    ws = convert(array_type(arch), zeros(P))
    ss = convert(array_type(arch), zeros(P))

    particles = StructArray{TestParticle}((xs, ys, zs, us, vs, ws, ss))

    velocities = VelocityFields(arch, grid)
    u, v, w = velocities
    speed = ComputedField(√(u^2 + v^2 + w^2))

    tracked_fields = merge(velocities, (s=speed,))
    lagrangian_particles = LagrangianParticles(particles; tracked_fields)

    model = IncompressibleModel(architecture=arch, grid=grid, timestepper=timestepper,
                                velocities=velocities, particles=lagrangian_particles)

    set!(model, u=1, v=1)

    sim = Simulation(model, Δt=1e-2, stop_iteration=1)

    test_output_file = "test_particles.nc"
    sim.output_writers[:particles] = NetCDFOutputWriter(model, model.particles,
                                                        filepath=test_output_file,
                                                        schedule=IterationInterval(1))

    sim.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1),
                                                     dir = ".", prefix = "particles_checkpoint")

    run!(sim)

    @test length(model.particles) == P
    @test size(model.particles) == (P,)
    @test propertynames(model.particles.particles) == (:x, :y, :z, :u, :v, :w, :s)

    x = convert(array_type(arch), model.particles.particles.x)
    y = convert(array_type(arch), model.particles.particles.y)
    z = convert(array_type(arch), model.particles.particles.z)
    u = convert(array_type(arch), model.particles.particles.u)
    v = convert(array_type(arch), model.particles.particles.v)
    w = convert(array_type(arch), model.particles.particles.w)
    s = convert(array_type(arch), model.particles.particles.s)

    @test size(x) == (P,)
    @test size(y) == (P,)
    @test size(z) == (P,)
    @test size(u) == (P,)
    @test size(v) == (P,)
    @test size(w) == (P,)
    @test size(s) == (P,)

    @test all(x .≈ 0.01)
    @test all(y .≈ 0.01)
    @test all(z .≈ 0.5)
    @test all(u .≈ 1)
    @test all(v .≈ 1)
    @test all(w .≈ 0)
    @test all(s .≈ √2)

    ds = NCDataset(test_output_file)
    x, y, z = ds["x"], ds["y"], ds["z"]
    u, v, w, s = ds["u"], ds["v"], ds["w"], ds["s"]

    @test size(x) == (P, 2)
    @test size(y) == (P, 2)
    @test size(z) == (P, 2)
    @test size(u) == (P, 2)
    @test size(v) == (P, 2)
    @test size(w) == (P, 2)
    @test size(s) == (P, 2)

    @test all(x[:, end] .≈ 0.01)
    @test all(y[:, end] .≈ 0.01)
    @test all(z[:, end] .≈ 0.5)
    @test all(u[:, end] .≈ 1)
    @test all(v[:, end] .≈ 1)
    @test all(w[:, end] .≈ 0)
    @test all(s[:, end] .≈ √2)

    close(ds)
    rm(test_output_file)

    # Test checkpoint of particle properties
    model.particles.particles.x .= 0
    model.particles.particles.y .= 0
    model.particles.particles.z .= 0
    model.particles.particles.u .= 0
    model.particles.particles.v .= 0
    model.particles.particles.w .= 0
    model.particles.particles.s .= 0

    set!(model, "particles_checkpoint_iteration1.jld2")

    x = convert(array_type(arch), model.particles.particles.x)
    y = convert(array_type(arch), model.particles.particles.y)
    z = convert(array_type(arch), model.particles.particles.z)
    u = convert(array_type(arch), model.particles.particles.u)
    v = convert(array_type(arch), model.particles.particles.v)
    w = convert(array_type(arch), model.particles.particles.w)
    s = convert(array_type(arch), model.particles.particles.s)

    @test model.particles.particles isa StructArray

    @test size(x) == (P,)
    @test size(y) == (P,)
    @test size(z) == (P,)
    @test size(u) == (P,)
    @test size(v) == (P,)
    @test size(w) == (P,)
    @test size(s) == (P,)

    @test all(x .≈ 0.01)
    @test all(y .≈ 0.01)
    @test all(z .≈ 0.5)
    @test all(u .≈ 1)
    @test all(v .≈ 1)
    @test all(w .≈ 0)
    @test all(s .≈ √2)

    return nothing
end

@testset "Lagrangian particle tracking" begin
    for arch in archs, timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @info "  Testing Lagrangian particle tacking [$(typeof(arch)), $timestepper]..."
        run_simple_particle_tracking_tests(arch, timestepper)
    end
end
