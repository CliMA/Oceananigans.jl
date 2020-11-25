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

    test_output_file = "test_particles.nc"
    outputw = NetCDFOutputWriter(model, model.particles,
                                 filepath=test_output_file,
                                 schedule=IterationInterval(1))

    sim = Simulation(model, Δt=1e-2, stop_iteration=1)
    sim.output_writers[:particles] = outputw

    run!(sim)

    x = convert(array_type(arch), model.particles.x)
    y = convert(array_type(arch), model.particles.y)
    z = convert(array_type(arch), model.particles.z)

    @test all(x .≈ 0.01)
    @test all(y .≈ 0.01)
    @test all(z .≈ 0.5)

    ds = NCDataset(test_output_file)
    x, y, z = ds["x"], ds["y"], ds["z"]

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
