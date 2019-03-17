function test_basic_timestepping()
    Nx, Ny, Nz = 4, 5, 6
    Lx, Ly, Lz = 1, 2, 3
    Nt, Δt = 10, 1
    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))
    time_step!(model, Nt, Δt)
    return typeof(model) == Model # Just testing that no errors happen.
end

function test_first_AB2_time_Step()
    Nx, Ny, Nz = 10, 12, 15
    Lx, Ly, Lz = 1, 2, 3
    Δt = 1

    model = Model((Nx, Ny, Nz), (Lx, Ly, Lz))

    add_ones(args...) = 1.0
    model.forcing = Forcing(nothing, nothing, nothing, add_ones, nothing)

    time_step!(model, 1, Δt)

    # Test that GT = 1 after first time step and that AB2 actually
    # reduced to forward Euler.
    @test all(model.G.Gu.data .≈ 0)
    @test all(model.G.Gv.data .≈ 0)
    @test all(model.G.Gw.data .≈ 0)
    @test all(model.G.GT.data .≈ 1.0)
    @test all(model.G.GS.data .≈ 0)

    return nothing
end
