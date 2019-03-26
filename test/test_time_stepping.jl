function time_stepping_works(arch, ft)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1, 2, 3
    Δt = 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=ft)
    time_step!(model, 1, Δt)

    # Just testing that no errors/crashes happen when time stepping.
    return true
end

function run_first_AB2_time_step_tests(arch, ft)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1, 2, 3
    Δt = 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=ft)

    add_ones(args...) = 1.0
    model.forcing = Forcing(nothing, nothing, nothing, add_ones, nothing)

    time_step!(model, 1, Δt)

    # Test that GT = 1 after first time step and that AB2 actually reduced to forward Euler.
    @test all(model.G.Gu.data .≈ 0)
    @test all(model.G.Gv.data .≈ 0)
    @test all(model.G.Gw.data .≈ 0)
    @test all(model.G.GT.data .≈ 1.0)
    @test all(model.G.GS.data .≈ 0)

    return nothing
end
