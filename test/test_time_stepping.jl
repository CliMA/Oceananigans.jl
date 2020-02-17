function time_stepping_works_with_closure(arch, FT, Closure)
    # Use halos of size 2 to accomadate time stepping with AnisotropicBiharmonicDiffusivity.
    grid = RegularCartesianGrid(FT; size=(16, 16, 16), halo=(2, 2, 2), length=(1, 2, 3))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, closure=Closure(FT))
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_nonlinear_eos(arch, FT, eos_type)
    grid = RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3))

    eos = RoquetIdealizedNonlinearEquationOfState(eos_type)
    b = SeawaterBuoyancy(equation_of_state=eos)

    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid, buoyancy=b)
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function run_first_AB2_time_step_tests(arch, FT)
    add_ones(args...) = 1.0
    model = IncompressibleModel(grid=RegularCartesianGrid(FT; size=(16, 16, 16), length=(1, 2, 3)),
                                architecture=arch, float_type=FT, forcing=ModelForcing(T=add_ones))
    time_step!(model, 1, euler=true)

    # Test that GT = 1 after first time step and that AB2 actually reduced to forward Euler.
    @test all(interior(model.timestepper.Gⁿ.u) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.v) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.w) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.T) .≈ 1.0)
    @test all(interior(model.timestepper.Gⁿ.S) .≈ 0)

    return nothing
end

"""
    This test ensures that when we compute w from the continuity equation that the full velocity field
    is divergence-free.
"""
function compute_w_from_continuity(arch, FT)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 16, 16, 16

    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    U = VelocityFields(arch, grid)
    div_U = CellField(FT, arch, grid, TracerBoundaryConditions(grid))

    interior(U.u) .= rand(FT, Nx, Ny, Nz)
    interior(U.v) .= rand(FT, Nx, Ny, Nz)

    fill_halo_regions!(U, arch)

    @launch(device(arch), config=launch_config(grid, :xy),
            _compute_w_from_continuity!((u=U.u.data, v=U.v.data, w=U.w.data), grid))

    fill_halo_regions!(U, arch)
    velocity_div!(grid, U.u.data, U.v.data, U.w.data, div_U.data)

    # Set div_U to zero at the top because the initial velocity field is not
    # divergence-free so we end up some divergence at the top if we don't do this.
    interior(div_U)[:, :, Nz] .= zero(FT)

    min_div = minimum(interior(div_U))
    max_div = maximum(interior(div_U))
    sum_div = sum(interior(div_U))
    abs_sum_div = sum(abs.(interior(div_U)))
    @info "Velocity divergence after recomputing w [$(typeof(arch)), $FT]: " *
          "min=$min_div, max=$max_div, sum=$sum_div, abs_sum=$abs_sum_div"

    return all(isapprox.(interior(div_U), 0, atol=5*eps(FT)))
end

"""
    This tests to make sure that the velocity field remains incompressible (or divergence-free) as the model is time
    stepped. It just initializes a cube shaped hot bubble perturbation in the center of the 3D domain to induce a
    velocity field.
"""
function incompressible_in_time(arch, FT, Nt)
    Nx, Ny, Nz = 32, 32, 32
    Lx, Ly, Lz = 10, 10, 10

    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    grid = model.grid
    u, v, w = model.velocities

    div_U = CellField(FT, arch, grid, TracerBoundaryConditions(grid))

    # Just add a temperature perturbation so we get some velocity field.
    @. model.tracers.T.data[8:24, 8:24, 8:24] += 0.01

    for n in 1:Nt
        time_step!(model, 0.05, euler = n==1)
    end

    velocity_div!(grid, u, v, w, div_U)

    min_div = minimum(interior(div_U))
    max_div = minimum(interior(div_U))
    sum_div = sum(interior(div_U))
    abs_sum_div = sum(abs.(interior(div_U)))
    @info "Velocity divergence after $Nt time steps [$(typeof(arch)), $FT]: " *
          "min=$min_div, max=$max_div, sum=$sum_div, abs_sum=$abs_sum_div"

    # We are comparing with 0 so we use absolute tolerances. They are a bit larger than eps(Float64) and eps(Float32)
    # because we are summing over the absolute value of many machine epsilons. A better atol value may be
    # Nx*Ny*Nz*eps(FT) but it's much higher than the observed abs_sum_div.
    FT == Float64 && return isapprox(abs_sum_div, 0, atol=5e-16)
    FT == Float32 && return isapprox(abs_sum_div, 0, atol=1e-7)
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

    topology = (Periodic, Bounded, Bounded)
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))
    model = IncompressibleModel(architecture = arch, float_type = FT, grid = grid,
                                closure = ConstantAnisotropicDiffusivity(νh=νh, νv=νv, κh=κh, κv=κv))

    Ty = 1e-4  # Meridional temperature gradient [K/m].
    Tz = 5e-3  # Vertical temperature gradient [K/m].

    # Initial temperature field [°C].
    T₀(x, y, z) = 10 + Ty*y + Tz*z + 0.0001*rand()
    set!(model, T=T₀)

    Tavg0 = mean(interior(model.tracers.T))

    for n in 1:Nt
        time_step!(model, 600, euler= n==1)
    end

    Tavg = mean(interior(model.tracers.T))
    @info "Tracer conservation after $Nt time steps [$(typeof(arch)), $FT]: " *
          "⟨T⟩-T₀=$(Tavg-Tavg0) °C"

    # Interestingly, it's very well conserved (almost to machine epsilon) for
    # Float64, but not as close for Float32... But it does seem constant in time
    # for Float32 so at least it is bounded.
    FT == Float64 && return isapprox(Tavg, Tavg0, rtol=2e-14)
    FT == Float32 && return isapprox(Tavg, Tavg0, rtol=2e-4)
end

Closures = (ConstantIsotropicDiffusivity, ConstantAnisotropicDiffusivity,
            AnisotropicBiharmonicDiffusivity, TwoDimensionalLeith,
            SmagorinskyLilly, BlasiusSmagorinsky,
            AnisotropicMinimumDissipation, RozemaAnisotropicMinimumDissipation)

@testset "Time stepping" begin
    @info "Testing time stepping..."

    @testset "Turbulence closures" begin
        for arch in archs, FT in [Float64], Closure in Closures
            @info "  Testing that time stepping works [$(typeof(arch)), $FT, $Closure]..."
            if Closure === TwoDimensionalLeith
                # This test is extremely slow; skipping for now.
                @test_skip time_stepping_works_with_closure(arch, FT, Closure)
            else
                @test time_stepping_works_with_closure(arch, FT, Closure)
            end
        end
    end

    @testset "Idealized nonlinear equation of state" begin
        for arch in archs, FT in [Float64]
            for eos_type in keys(Oceananigans.Buoyancy.optimized_roquet_coeffs)
                @info "  Testing that time stepping works with " *
                        "RoquetIdealizedNonlinearEquationOfState [$(typeof(arch)), $FT, $eos_type]"
                @test time_stepping_works_with_nonlinear_eos(arch, FT, eos_type)
            end
        end
    end

    @testset "2nd-order Adams-Bashforth" begin
        @info "  Testing 2nd-order Adams-Bashforth..."
        for arch in archs, FT in float_types
            run_first_AB2_time_step_tests(arch, FT)
        end
    end

    @testset "Recomputing w from continuity" begin
        @info "  Testing recomputing w from continuity..."
        for arch in archs, FT in float_types
            @test compute_w_from_continuity(arch, FT)
        end
    end

    @testset "Incompressibility" begin
        @info "  Testing incompressibility..."
        for arch in archs, FT in float_types, Nt in [1, 10, 100]
            @test incompressible_in_time(arch, FT, Nt)
        end
    end

    @testset "Tracer conservation in channel" begin
        @info "  Testing tracer conservation in channel..."
        for arch in archs, FT in float_types
            @test tracer_conserved_in_channel(arch, FT, 10)
        end
    end
end
