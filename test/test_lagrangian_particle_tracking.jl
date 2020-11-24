function run_simple_particle_tracking_tests(arch)
    topo = (Periodic, Periodic, Bounded)
    domain = (x=(0, 1), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topo, size=(5, 5, 5); domain...)

    P = 10
    xs = convert(array_type(arch), zeros(P))
    ys = convert(array_type(arch), zeros(P))
    zs = convert(array_type(arch), zeros(P))

    model = IncompressibleModel(architecture=arch, grid=grid,
                                particles=LagrangianParticles(x=xs, y=ys, z=zs))

    set!(model, u=1, v=1)

    time_step!(model, 1e-2)

    # Easy placeholder test!
    @test all(model.particles.x .≈ 0.01)
    @test all(model.particles.y .≈ 0.01)

    return nothing
end

@testset "Lagrangian particle tracking" begin
    for arch in archs
        @info "  Testing Lagrangian particle tacking [$(typeof(arch))]..."
        run_simple_particle_tracking_tests(arch)
    end
end
