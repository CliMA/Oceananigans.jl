using Oceananigans.Grids: topological_tuple_length, total_size
using Oceananigans.Fields: BackgroundField
using Oceananigans.TimeSteppers: Clock

function time_stepping_works_with_flat_dimensions(arch, topology)
    size = Tuple(1 for i = 1:topological_tuple_length(topology...))
    extent = Tuple(1 for i = 1:topological_tuple_length(topology...))
    grid = RegularRectilinearGrid(size=size, extent=extent, topology=topology)
    model = IncompressibleModel(grid=grid, architecture=arch)
    time_step!(model, 1, euler=true)
    return true # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_coriolis(arch, FT, Coriolis)
    grid = RegularRectilinearGrid(FT, size=(1, 1, 1), extent=(1, 2, 3))
    c = Coriolis(FT, latitude=45)
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, coriolis=c)

    time_step!(model, 1, euler=true)

    return true # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_closure(arch, FT, Closure)
    # Use halos of size 2 to accomadate time stepping with AnisotropicBiharmonicDiffusivity.
    grid = RegularRectilinearGrid(FT; size=(1, 1, 1), halo=(2, 2, 2), extent=(1, 2, 3))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, closure=Closure(FT))
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_advection_scheme(arch, advection_scheme)
    # Use halo=(3, 3, 3) to accomodate WENO-5 advection scheme
    grid = RegularRectilinearGrid(size=(1, 1, 1), halo=(3, 3, 3), extent=(1, 2, 3))
    model = IncompressibleModel(grid=grid, architecture=arch, advection=advection_scheme)
    time_step!(model, 1, euler=true)
    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_nothing_closure(arch, FT)
    grid = RegularRectilinearGrid(FT; size=(1, 1, 1), extent=(1, 2, 3))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, closure=nothing)
    time_step!(model, 1, euler=true)
    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_nonlinear_eos(arch, FT, EOS)
    grid = RegularRectilinearGrid(FT; size=(1, 1, 1), extent=(1, 2, 3))

    eos = EOS()
    b = SeawaterBuoyancy(equation_of_state=eos)

    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid, buoyancy=b)
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function run_first_AB2_time_step_tests(arch, FT)
    add_ones(args...) = 1.0

    # Weird grid size to catch https://github.com/CliMA/Oceananigans.jl/issues/780
    grid = RegularRectilinearGrid(FT, size=(13, 17, 19), extent=(1, 2, 3))

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, forcing=(T=add_ones,))
    time_step!(model, 1, euler=true)

    # Test that GT = 1, T = 1 after 1 time step and that AB2 actually reduced to forward Euler.
    @test all(interior(model.timestepper.Gⁿ.u) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.v) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.w) .≈ 0)
    @test all(interior(model.timestepper.Gⁿ.T) .≈ 1.0)
    @test all(interior(model.timestepper.Gⁿ.S) .≈ 0)

    @test all(interior(model.velocities.u) .≈ 0)
    @test all(interior(model.velocities.v) .≈ 0)
    @test all(interior(model.velocities.w) .≈ 0)
    @test all(interior(model.tracers.T)    .≈ 1.0)
    @test all(interior(model.tracers.S)    .≈ 0)

    return nothing
end

"""
    This tests to make sure that the velocity field remains incompressible (or divergence-free) as the model is time
    stepped. It just initializes a cube shaped hot bubble perturbation in the center of the 3D domain to induce a
    velocity field.
"""
function incompressible_in_time(arch, FT, Nt, timestepper)
    Nx, Ny, Nz = 32, 32, 32
    Lx, Ly, Lz = 10, 10, 10

    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT, timestepper=timestepper)

    grid = model.grid
    u, v, w = model.velocities

    div_U = CenterField(FT, arch, grid, TracerBoundaryConditions(grid))

    # Just add a temperature perturbation so we get some velocity field.
    @. model.tracers.T.data[8:24, 8:24, 8:24] += 0.01

    for n in 1:Nt
        ab2_or_rk3_time_step!(model, 0.05, n)
    end

    event = launch!(arch, grid, :xyz, divergence!, grid, u.data, v.data, w.data, div_U.data, dependencies=Event(device(arch)))
    wait(device(arch), event)

    min_div = minimum(interior(div_U))
    max_div = maximum(interior(div_U))
    max_abs_div = maximum(abs, interior(div_U))
    sum_div = sum(interior(div_U))
    sum_abs_div = sum(abs, interior(div_U))
    @info "Velocity divergence after $Nt time steps [$(typeof(arch)), $FT, $timestepper]: " *
          "min=$min_div, max=$max_div, max_abs_div=$max_abs_div, sum=$sum_div, abs_sum=$sum_abs_div"

    # We are comparing with 0 so we use absolute tolerances. They are a bit larger than eps(Float64) and eps(Float32)
    # because we are summing over the absolute value of many machine epsilons. A better atol value may be
    # Nx*Ny*Nz*eps(FT) but it's much higher than the observed abs_sum_div.
    return isapprox(max_abs_div, 0, atol=5e-8)
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
    νz, κz = α*νh, α*κh

    topology = (Periodic, Bounded, Bounded)
    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = IncompressibleModel(architecture = arch, float_type = FT, grid = grid,
                                closure = AnisotropicDiffusivity(νh=νh, νz=νz, κh=κh, κz=κz))

    Ty = 1e-4  # Meridional temperature gradient [K/m].
    Tz = 5e-3  # Vertical temperature gradient [K/m].

    # Initial temperature field [°C].
    T₀(x, y, z) = 10 + Ty*y + Tz*z + 0.0001*rand()
    set!(model, T=T₀)

    Tavg0 = mean(interior(model.tracers.T))

    for n in 1:Nt
        ab2_or_rk3_time_step!(model, 600, n)
    end

    Tavg = mean(interior(model.tracers.T))
    @info "Tracer conservation after $Nt time steps [$(typeof(arch)), $FT]: " *
          "⟨T⟩-T₀=$(Tavg-Tavg0) °C"

    return isapprox(Tavg, Tavg0, atol=Nx*Ny*Nz*eps(FT))
end

function time_stepping_with_background_fields(arch)

    grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

    background_u(x, y, z, t) = π
    background_v(x, y, z, t) = sin(x) * cos(y) * exp(t)

    background_w_func(x, y, z, t, p) = p.α * x + p.β * exp(z / p.λ)
    background_w = BackgroundField(background_w_func, parameters=(α=1.2, β=0.2, λ=43))

    background_T(x, y, z, t) = background_u(x, y, z, t)

    background_S_func(x, y, z, t, α) = α * y
    background_S = BackgroundField(background_S_func, parameters=1.2)

    model = IncompressibleModel(grid=grid, background_fields=(u=background_u, v=background_v, w=background_w,
                                                              T=background_T, S=background_S))

    time_step!(model, 1, euler=true)

    return location(model.background_fields.velocities.u) === (Face, Center, Center) &&
           location(model.background_fields.velocities.v) === (Center, Face, Center) &&
           location(model.background_fields.velocities.w) === (Center, Center, Face) &&
           location(model.background_fields.tracers.T) === (Center, Center, Center) &&
           location(model.background_fields.tracers.S) === (Center, Center, Center)
end

Planes = (FPlane, NonTraditionalFPlane, BetaPlane, NonTraditionalBetaPlane)

Closures = (IsotropicDiffusivity, AnisotropicDiffusivity,
            AnisotropicBiharmonicDiffusivity, TwoDimensionalLeith,
            SmagorinskyLilly, AnisotropicMinimumDissipation)

advection_schemes = (nothing,
                     CenteredSecondOrder(),
                     UpwindBiasedThirdOrder(),
                     CenteredFourthOrder(),
                     UpwindBiasedFifthOrder(),
                     WENO5())

timesteppers = (:QuasiAdamsBashforth2, :RungeKutta3)

@testset "Time stepping" begin
    @info "Testing time stepping..."

    for arch in archs, FT in float_types
        @testset "Time stepping with DateTimes [$(typeof(arch)), $FT]" begin
            @info "  Testing time stepping with datetime clocks [$(typeof(arch)), $FT]"

            model = IncompressibleModel(grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)),
                                        clock = Clock(time=DateTime(2020)))

            time_step!(model, 7.883)
            @test model.clock.time == DateTime("2020-01-01T00:00:07.883")

            model = IncompressibleModel(grid = RegularRectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1)),
                                        clock = Clock(time=TimeDate(2020)))

            time_step!(model, 123e-9)  # 123 nanoseconds
            @test model.clock.time == TimeDate("2020-01-01T00:00:00.000000123")
        end
    end

   @testset "Flat dimensions" begin
        for arch in archs
            for topology in ((Flat, Periodic, Periodic),
                             (Periodic, Flat, Periodic),
                             (Periodic, Periodic, Flat),
                             (Flat, Flat, Bounded))

                TX, TY, TZ = topology
                @info "  Testing that time stepping works with flat dimensions [$(typeof(arch)), $TX, $TY, $TZ]..."
                @test time_stepping_works_with_flat_dimensions(arch, topology)
            end
        end
    end

    @testset "Coriolis" begin
        for arch in archs, FT in [Float64], Coriolis in Planes
            @info "  Testing that time stepping works [$(typeof(arch)), $FT, $Coriolis]..."
            @test time_stepping_works_with_coriolis(arch, FT, Coriolis)
        end
    end

    @testset "Advection schemes" begin
        for arch in archs, advection_scheme in advection_schemes
            @info "  Testing time stepping with advection schemes [$(typeof(arch)), $(typeof(advection_scheme))]"
            @test time_stepping_works_with_advection_scheme(arch, advection_scheme)
        end
    end

    @testset "BackgroundFields" begin
        for arch in archs
            @info "  Testing that time stepping works with background fields [$(typeof(arch))]..."
            @test time_stepping_with_background_fields(arch)
        end
    end

    @testset "Turbulence closures" begin
        for arch in archs, FT in [Float64]

            @info "  Testing that time stepping works [$(typeof(arch)), $FT, nothing]..."
            @test time_stepping_works_with_nothing_closure(arch, FT)

            for Closure in Closures
                @info "  Testing that time stepping works [$(typeof(arch)), $FT, $Closure]..."
                if Closure === TwoDimensionalLeith && arch isa CPU
                    # This test is extremely slow so we skip.
                    # See: https://github.com/CliMA/Oceananigans.jl/pull/1074
                    @test_skip time_stepping_works_with_closure(arch, FT, Closure)
                else
                    @test time_stepping_works_with_closure(arch, FT, Closure)
                end
            end
        end
    end

    @testset "Idealized nonlinear equation of state" begin
        for arch in archs, FT in [Float64]
            for eos_type in (SeawaterPolynomials.RoquetEquationOfState, SeawaterPolynomials.TEOS10EquationOfState)
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

    @testset "Incompressibility" begin
        @info "  Testing incompressibility..."
        for arch in archs, FT in float_types, Nt in [1, 10, 100], timestepper in timesteppers
            @test incompressible_in_time(arch, FT, Nt, timestepper)
        end
    end

    @testset "Tracer conservation in channel" begin
        @info "  Testing tracer conservation in channel..."
        for arch in archs, FT in float_types
            @test tracer_conserved_in_channel(arch, FT, 10)
        end
    end
end
