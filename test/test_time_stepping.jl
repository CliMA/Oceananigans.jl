using Oceananigans: velocity_div!

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

function incompressible_in_time(arch, ft, Nt)
    Nx, Ny, Nz = 32, 32, 32
    Lx, Ly, Lz = 10, 10, 10

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=ft)

    grid = model.grid
    u, v, w = model.velocities.u, model.velocities.v, model.velocities.w

    div_u = CellField(arch, model.grid)

    # Just add a temperature perturbation so we get some velocity field.
    @. model.tracers.T.data[8:24, 8:24, 8:24] += 0.01

    time_step!(model, Nt, 1)

    velocity_div!(grid, u, v, w, div_u)

    min_div = minimum(div_u)
    max_div = minimum(div_u)
    sum_div = sum(div_u)
    abs_sum_div = sum(abs.(div_u))
    @info "Velocity divergence after $Nt time steps ($arch, $ft): min=$min_div, max=$max_div, sum=$sum_div, abs_sum=$abs_sum_div"

    isapprox(abs_sum_div, 0; atol=1e-14)
end
