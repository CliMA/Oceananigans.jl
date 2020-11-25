using NCDatasets

function run_simple_particle_tracking_tests(arch)
    topo = (Periodic, Periodic, Bounded)
    domain = (x=(-1, 1), y=(-1, 1), z=(-1, 1))
    grid = RegularCartesianGrid(topology=topo, size=(5, 5, 5); domain...)

    P = 10
    xs = convert(array_type(arch), zeros(P))
    ys = convert(array_type(arch), zeros(P))
    zs = convert(array_type(arch), 0.5*ones(P))

    model = IncompressibleModel(architecture=arch, grid=grid,
                                particles=LagrangianParticles(x=xs, y=ys, z=zs))

    set!(model, u=1, v=1)

    sim = Simulation(model, Δt=1e-2, stop_iteration=1)

    test_output_file = "test_particles.nc"
    sim.output_writers[:particles] = NetCDFOutputWriter(model, model.particles,
                                                        filepath=test_output_file,
                                                        schedule=IterationInterval(1))

    run!(sim)

    @test length(model.particles) == P
    @test size(model.particles) == (P,)

    x = convert(array_type(arch), model.particles.particles.x)
    y = convert(array_type(arch), model.particles.particles.y)
    z = convert(array_type(arch), model.particles.particles.z)

    @test size(x) == (P,)
    @test size(y) == (P,)
    @test size(z) == (P,)

    @test all(x .≈ 0.01)
    @test all(y .≈ 0.01)
    @test all(z .≈ 0.5)

    ds = NCDataset(test_output_file)
    x, y, z = ds["x"], ds["y"], ds["z"]

    @test size(x) == (P, 2)
    @test size(y) == (P, 2)
    @test size(z) == (P, 2)

    @test all(x[:, end] .≈ 0.01)
    @test all(y[:, end] .≈ 0.01)
    @test all(z[:, end] .≈ 0.5)

    close(ds)
    rm(test_output_file)

    return nothing
end

@testset "Lagrangian particle tracking" begin
    for arch in archs
        @info "  Testing Lagrangian particle tacking [$(typeof(arch))]..."
        run_simple_particle_tracking_tests(arch)
    end
end
