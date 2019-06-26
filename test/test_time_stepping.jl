using Oceananigans: velocity_div!, compute_w_from_continuity!


function time_stepping_works(arch, FT)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1, 2, 3
    Δt = 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=FT)
    time_step!(model, 1, Δt)

    # Just testing that no errors/crashes happen when time stepping.
    return true
end

function run_first_AB2_time_step_tests(arch, FT)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1, 2, 3
    Δt = 1

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=FT)

    add_ones(args...) = 1.0
    model.forcing = Forcing(nothing, nothing, nothing, add_ones, nothing)

    time_step!(model, 1, Δt)

    # Test that GT = 1 after first time step and that AB2 actually reduced to forward Euler.
    @test all(data(model.G.Gu) .≈ 0)
    @test all(data(model.G.Gv) .≈ 0)
    @test all(data(model.G.Gw) .≈ 0)
    @test all(data(model.G.GT) .≈ 1.0)
    @test all(data(model.G.GS) .≈ 0)

    return nothing
end

"""
    This test ensures that when we compute w from the continuity equation that the full velocity field
    is divergence-free.
"""
function compute_w_from_continuity(arch, FT)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 16, 16, 16

    grid = RegularCartesianGrid(FT, (Nx, Ny, Nz), (Lx, Ly, Lz))
    fbcs = DoublyPeriodicBCs()

    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    div_u = CellField(FT, arch, grid)

    data(u) .= rand(FT, Nx, Ny, Nz)
    data(v) .= rand(FT, Nx, Ny, Nz)

    fill_halo_regions!(grid, (:u, fbcs, u.data), (:v, fbcs, v.data))
    compute_w_from_continuity!(grid, u.data, v.data, w.data)

    fill_halo_regions!(grid, (:w, fbcs, w.data))
    velocity_div!(grid, u.data, v.data, w.data, div_u.data)

    # Set div_u to zero at the bottom because the initial velocity field is not divergence-free
    # so we end up some divergence at the bottom if we don't do this.
    data(div_u)[:, :, end] .= zero(FT)

    min_div = minimum(data(div_u))
    max_div = minimum(data(div_u))
    sum_div = sum(data(div_u))
    abs_sum_div = sum(abs.(data(div_u)))
    @info "Velocity divergence after recomputing w ($arch, $FT): min=$min_div, max=$max_div, sum=$sum_div, abs_sum=$abs_sum_div"

    all(isapprox.(data(div_u), 0; atol=5*eps(FT)))
end

"""
    This tests to make sure that the velocity field remains incompressible (or divergence-free) as the model is time
    stepped. It just initializes a cube shaped hot bubble perturbation in the center of the 3D domain to induce a
    velocity field.
"""
function incompressible_in_time(arch, FT, Nt)
    Nx, Ny, Nz = 32, 32, 32
    Lx, Ly, Lz = 10, 10, 10

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=arch, float_type=FT)

    grid = model.grid
    u, v, w = model.velocities.u, model.velocities.v, model.velocities.w

    div_u = CellField(FT, arch, model.grid)

    # Just add a temperature perturbation so we get some velocity field.
    @. model.tracers.T.data[8:24, 8:24, 8:24] += 0.01

    time_step!(model, Nt, 0.05)

    velocity_div!(grid, u, v, w, div_u)

    min_div = minimum(data(div_u))
    max_div = minimum(data(div_u))
    sum_div = sum(data(div_u))
    abs_sum_div = sum(abs.(data(div_u)))
    @info "Velocity divergence after $Nt time steps ($arch, $FT): min=$min_div, max=$max_div, sum=$sum_div, abs_sum=$abs_sum_div"

    # We are comparing with 0 so we use absolute tolerances. They are a bit larger than eps(Float64) and eps(Float32)
    # because we are summing over the absolute value of many machine epsilons. A better atol value may be
    # Nx*Ny*Nz*eps(FT) but it's much higher than the observed abs_sum_div.
    if FT == Float64
        return isapprox(abs_sum_div, 0; atol=5e-16)
    elseif FT == Float32
        return isapprox(abs_sum_div, 0; atol=1e-7)
    end
end
