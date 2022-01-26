include("dependencies_for_runtests.jl")

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
    grid = RectilinearGrid(arch, topology=topo, size=(5, 5, 5); domain...)

    P = 10
    xs = arch_array(arch, zeros(P))
    ys = arch_array(arch, zeros(P))
    zs = arch_array(arch, 0.5*ones(P))

    # Test first constructor
    lagrangian_particles = LagrangianParticles(x=xs, y=ys, z=zs)
    @test lagrangian_particles isa LagrangianParticles

    us = convert(array_type(arch), zeros(P))
    vs = convert(array_type(arch), zeros(P))
    ws = convert(array_type(arch), zeros(P))
    ss = convert(array_type(arch), zeros(P))

    particles = StructArray{TestParticle}((xs, ys, zs, us, vs, ws, ss))

    velocities = VelocityFields(grid)
    u, v, w = velocities
    speed = Field(√(u*u + v*v + w*w))

    tracked_fields = merge(velocities, (; s=speed))

    # Test second constructor
    lagrangian_particles = LagrangianParticles(particles; tracked_fields)
    @test lagrangian_particles isa LagrangianParticles

    model = NonhydrostaticModel(grid=grid, timestepper=timestepper,
                                velocities=velocities, particles=lagrangian_particles)

    set!(model, u=1, v=1)

    sim = Simulation(model, Δt=1e-2, stop_iteration=1)

    jld2_filepath = "test_particles.jld2"
    sim.output_writers[:particles_jld2] =
        JLD2OutputWriter(model, (; particles=model.particles),
                         prefix="test_particles", schedule=IterationInterval(1))

    nc_filepath = "test_particles.nc"
    sim.output_writers[:particles_nc] =
        NetCDFOutputWriter(model, model.particles, filepath=nc_filepath, schedule=IterationInterval(1))

    sim.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1),
                                                     dir = ".", prefix = "particles_checkpoint")

    run!(sim)

    @test length(model.particles) == P
    @test size(model.particles) == tuple(P)
    @test propertynames(model.particles.properties) == (:x, :y, :z, :u, :v, :w, :s)

    x = convert(array_type(arch), model.particles.properties.x)
    y = convert(array_type(arch), model.particles.properties.y)
    z = convert(array_type(arch), model.particles.properties.z)
    u = convert(array_type(arch), model.particles.properties.u)
    v = convert(array_type(arch), model.particles.properties.v)
    w = convert(array_type(arch), model.particles.properties.w)
    s = convert(array_type(arch), model.particles.properties.s)

    @test size(x) == tuple(P)
    @test size(y) == tuple(P)
    @test size(z) == tuple(P)
    @test size(u) == tuple(P)
    @test size(v) == tuple(P)
    @test size(w) == tuple(P)
    @test size(s) == tuple(P)

    @test all(x .≈ 0.01)
    @test all(y .≈ 0.01)
    @test all(z .≈ 0.5)
    @test all(u .≈ 1)
    @test all(v .≈ 1)
    @test all(w .≈ 0)
    @test all(s .≈ √2)

    # Test NetCDF output is correct.
    ds = NCDataset(nc_filepath)
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
    rm(nc_filepath)

    # Test JLD2 output is correct
    file = jldopen(jld2_filepath)
    @test haskey(file["timeseries"], "particles")
    @test haskey(file["timeseries/particles"], "0")
    @test haskey(file["timeseries/particles"], "0")

    @test size(file["timeseries/particles/1"].x) == tuple(P)
    @test size(file["timeseries/particles/1"].y) == tuple(P)
    @test size(file["timeseries/particles/1"].z) == tuple(P)
    @test size(file["timeseries/particles/1"].u) == tuple(P)
    @test size(file["timeseries/particles/1"].v) == tuple(P)
    @test size(file["timeseries/particles/1"].w) == tuple(P)
    @test size(file["timeseries/particles/1"].s) == tuple(P)

    @test all(file["timeseries/particles/1"].x .≈ 0.01)
    @test all(file["timeseries/particles/1"].y .≈ 0.01)
    @test all(file["timeseries/particles/1"].z .≈ 0.5)
    @test all(file["timeseries/particles/1"].u .≈ 1)
    @test all(file["timeseries/particles/1"].v .≈ 1)
    @test all(file["timeseries/particles/1"].w .≈ 0)
    @test all(file["timeseries/particles/1"].s .≈ √2)

    close(file)
    rm(jld2_filepath)

    # Test checkpoint of particle properties
    model.particles.properties.x .= 0
    model.particles.properties.y .= 0
    model.particles.properties.z .= 0
    model.particles.properties.u .= 0
    model.particles.properties.v .= 0
    model.particles.properties.w .= 0
    model.particles.properties.s .= 0

    set!(model, "particles_checkpoint_iteration1.jld2")

    x = convert(array_type(arch), model.particles.properties.x)
    y = convert(array_type(arch), model.particles.properties.y)
    z = convert(array_type(arch), model.particles.properties.z)
    u = convert(array_type(arch), model.particles.properties.u)
    v = convert(array_type(arch), model.particles.properties.v)
    w = convert(array_type(arch), model.particles.properties.w)
    s = convert(array_type(arch), model.particles.properties.s)

    @test model.particles.properties isa StructArray

    @test size(x) == tuple(P)
    @test size(y) == tuple(P)
    @test size(z) == tuple(P)
    @test size(u) == tuple(P)
    @test size(v) == tuple(P)
    @test size(w) == tuple(P)
    @test size(s) == tuple(P)

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
