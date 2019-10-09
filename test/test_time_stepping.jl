function time_stepping_works(arch, FT, Closure)
    model = BasicModel(N=(16, 16, 16), L=(1, 2, 3), architecture=arch, float_type=FT,
                       closure=Closure(FT))
    time_step!(model, 1, 1)
    return true # test that no errors/crashes happen when time stepping.
end

function run_first_AB2_time_step_tests(arch, FT)
    add_ones(args...) = 1.0
    model = BasicModel(N=(16, 16, 16), L=(1, 2, 3), architecture=arch, float_type=FT,
                       forcing=ModelForcing(T=add_ones))
    time_step!(model, 1, 1)

    # Test that GT = 1 after first time step and that AB2 actually reduced to forward Euler.
    @test all(data(model.timestepper.Gⁿ.Gu) .≈ 0)
    @test all(data(model.timestepper.Gⁿ.Gv) .≈ 0)
    @test all(data(model.timestepper.Gⁿ.Gw) .≈ 0)
    @test all(data(model.timestepper.Gⁿ.GT) .≈ 1.0)
    @test all(data(model.timestepper.Gⁿ.GS) .≈ 0)

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
    bcs = HorizontallyPeriodicSolutionBCs()

    u = FaceFieldX(FT, arch, grid)
    v = FaceFieldY(FT, arch, grid)
    w = FaceFieldZ(FT, arch, grid)
    div_u = CellField(FT, arch, grid)

    data(u) .= rand(FT, Nx, Ny, Nz)
    data(v) .= rand(FT, Nx, Ny, Nz)

    fill_halo_regions!(u.data, bcs.u, arch, grid)
    fill_halo_regions!(v.data, bcs.v, arch, grid)
    compute_w_from_continuity!(grid, (u=u.data, v=v.data, w=w.data))

    fill_halo_regions!(w.data, bcs.w, arch, grid)
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

    model = BasicModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), architecture=arch, float_type=FT)

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

"""
    tracer_conserved_in_channel(arch, FT, Nt)

Create a super-coarse eddying channel model with walls in the y and test that
temperature is conserved after `Nt` time steps.
"""
function tracer_conserved_in_channel(arch, FT, Nt)
    Nx, Ny, Nz = 16, 32, 16
    Lx, Ly, Lz = 160e3, 320e3, 1024

    α = (Lz/Nz)/(Lx/Nx) # Grid cell aspect ratio.
    νh, κh = 20.0, 20.0
    νv, κv = α*νh, α*κh

    model = ChannelModel(architecture = arch, float_type = FT,
                         grid = RegularCartesianGrid(N = (Nx, Ny, Nz), L = (Lx, Ly, Lz)),
                         closure = ConstantAnisotropicDiffusivity(νh=νh, νv=νv, κh=κh, κv=κv))

    Ty = 1e-4  # Meridional temperature gradient [K/m].
    Tz = 5e-3  # Vertical temperature gradient [K/m].

    # Initial temperature field [°C].
    T₀(x, y, z) = 10 + Ty*y + Tz*z + 0.0001*rand()
    set_ic!(model, T=T₀)

    Tavg0 = mean(data(model.tracers.T))

    time_step!(model; Nt=Nt, Δt=10*60)

    Tavg = mean(data(model.tracers.T))
    @info "Tracer conservation after $Nt time steps ($arch, $FT): ⟨T⟩-T₀=$(Tavg-Tavg0) °C"

    # Interestingly, it's very well conserved (almost to machine epsilon) for
    # Float64, but not as close for Float32... But it does seem constant in time
    # for Float32 so at least it is bounded.
    if FT == Float64
        return isapprox(Tavg, Tavg0; rtol=2e-14)
    elseif FT == Float32
        return isapprox(Tavg, Tavg0; rtol=2e-4)
    end
end

Closures = (ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity,
            ConstantSmagorinsky, AnisotropicMinimumDissipation)

@testset "Time stepping" begin
    println("Testing time stepping...")

    for arch in archs, FT in float_types, Closure in Closures
        @test time_stepping_works(arch, FT, Closure)
    end

    @testset "2nd-order Adams-Bashforth" begin
        println("  Testing 2nd-order Adams-Bashforth...")
        for arch in archs, FT in float_types
            run_first_AB2_time_step_tests(arch, FT)
        end
    end

    @testset "Recomputing w from continuity" begin
        println("  Testing recomputing w from continuity...")
        for arch in archs, FT in float_types
            @test compute_w_from_continuity(arch, FT)
        end
    end

    @testset "Incompressibility" begin
        println("  Testing incompressibility...")
        for arch in archs, FT in float_types, Nt in [1, 10, 100]
            @test incompressible_in_time(arch, FT, Nt)
        end
    end

    @testset "Tracer conservation in channel" begin
        println("  Testing tracer conservation in channel...")
        for arch in archs, FT in float_types
            @test tracer_conserved_in_channel(arch, FT, 10)
        end
    end
end
