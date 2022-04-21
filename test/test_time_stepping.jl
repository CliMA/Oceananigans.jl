include("dependencies_for_runtests.jl")

using Oceananigans.Grids: topological_tuple_length, total_size
using Oceananigans.Fields: BackgroundField
using Oceananigans.TimeSteppers: Clock
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

function time_stepping_works_with_flat_dimensions(arch, topology)
    size = Tuple(1 for i = 1:topological_tuple_length(topology...))
    extent = Tuple(1 for i = 1:topological_tuple_length(topology...))
    grid = RectilinearGrid(arch, size=size, extent=extent, topology=topology)
    model = NonhydrostaticModel(grid=grid)
    time_step!(model, 1, euler=true)
    return true # Test that no errors/crashes happen when time stepping.
end

function euler_time_stepping_doesnt_propagate_NaNs(arch)
    model = HydrostaticFreeSurfaceModel(grid=RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 2, 3)),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b)

    CUDA.@allowscalar model.timestepper.G⁻.u[1, 1, 1] = NaN
    time_step!(model, 1, euler=true)
    u111 = CUDA.@allowscalar model.velocities.u[1, 1, 1]
    
    return !isnan(u111)
end

function time_stepping_works_with_coriolis(arch, FT, Coriolis)
    grid = RectilinearGrid(arch, FT, size=(1, 1, 1), extent=(1, 2, 3))
    c = Coriolis(FT, latitude=45)
    model = NonhydrostaticModel(grid=grid, coriolis=c)

    time_step!(model, 1, euler=true)

    return true # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_closure(arch, FT, Closure; buoyancy=Buoyancy(model=SeawaterBuoyancy(FT)))
    # Add TKE tracer "e" to tracers when using CATKEVerticalDiffusivity
    tracers = [:T, :S]
    Closure === CATKEVerticalDiffusivity && push!(tracers, :e)

    # Use halos of size 3 to be conservative
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), halo=(3, 3, 3), extent=(1, 2, 3))
    closure = Closure(FT)
    model = NonhydrostaticModel(; grid, closure, tracers, buoyancy)
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_advection_scheme(arch, advection_scheme)
    # Use halo=(3, 3, 3) to accomodate WENO-5 advection scheme
    grid = RectilinearGrid(arch, size=(1, 1, 1), halo=(3, 3, 3), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid, advection=advection_scheme)
    time_step!(model, 1, euler=true)
    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_nothing_closure(arch, FT)
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(1, 2, 3))
    model = NonhydrostaticModel(grid=grid, closure=nothing)
    time_step!(model, 1, euler=true)
    return true  # Test that no errors/crashes happen when time stepping.
end

function time_stepping_works_with_nonlinear_eos(arch, FT, EOS)
    grid = RectilinearGrid(arch, FT; size=(1, 1, 1), extent=(1, 2, 3))

    eos = EOS()
    b = SeawaterBuoyancy(equation_of_state=eos)

    model = NonhydrostaticModel(grid=grid, buoyancy=b,
                                tracers=(:T, :S))
    time_step!(model, 1, euler=true)

    return true  # Test that no errors/crashes happen when time stepping.
end

function run_first_AB2_time_step_tests(arch, FT)
    add_ones(args...) = 1.0

    # Weird grid size to catch https://github.com/CliMA/Oceananigans.jl/issues/780
    grid = RectilinearGrid(arch, FT, size=(13, 17, 19), extent=(1, 2, 3))

    model = NonhydrostaticModel(grid=grid, forcing=(T=add_ones,),
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
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
function incompressible_in_time(grid, Nt, timestepper)
    model = NonhydrostaticModel(grid=grid, timestepper=timestepper,
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    grid = model.grid
    u, v, w = model.velocities

    div_U = CenterField(grid)

    # Just add a temperature perturbation so we get some velocity field.
    CUDA.@allowscalar interior(model.tracers.T)[8:24, 8:24, 8:24] .+= 0.01

    update_state!(model)
    for n in 1:Nt
        time_step!(model, 0.05)
    end

    arch = architecture(grid)
    event = launch!(arch, grid, :xyz, divergence!, grid, u.data, v.data, w.data, div_U.data, dependencies=Event(device(arch)))
    wait(device(arch), event)

    min_div = CUDA.@allowscalar minimum(interior(div_U))
    max_div = CUDA.@allowscalar maximum(interior(div_U))
    max_abs_div = CUDA.@allowscalar maximum(abs, interior(div_U))
    sum_div = CUDA.@allowscalar sum(interior(div_U))
    sum_abs_div = CUDA.@allowscalar sum(abs, interior(div_U))

    @info "Velocity divergence after $Nt time steps [$(typeof(arch)), $(typeof(grid)), $timestepper]: " *
          "min=$min_div, max=$max_div, max_abs_div=$max_abs_div, sum=$sum_div, abs_sum=$sum_abs_div"

    # We are comparing with 0 so we use absolute tolerances. They are a bit larger than eps(Float64) and eps(Float32)
    # because we are summing over the absolute value of many machine epsilons. A better atol value may be
    # Nx*Ny*Nz*eps(eltype(grid)) but it's much higher than the observed max_abs_div, so out of a general abundance of caution
    # we manually insert a smaller tolerance than we might need for this test.
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
    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = NonhydrostaticModel(grid = grid,
                                closure = (HorizontalScalarDiffusivity(ν=νh, κ=κh), 
                                           VerticalScalarDiffusivity(ν=νz, κ=κz)),
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

    Ty = 1e-4  # Meridional temperature gradient [K/m].
    Tz = 5e-3  # Vertical temperature gradient [K/m].

    # Initial temperature field [°C].
    T₀(x, y, z) = 10 + Ty*y + Tz*z + 0.0001*rand()
    set!(model, T=T₀)

    Tavg0 = CUDA.@allowscalar mean(interior(model.tracers.T))

    update_state!(model)
    for n in 1:Nt
        time_step!(model, 600)
    end

    Tavg = CUDA.@allowscalar mean(interior(model.tracers.T))
    @info "Tracer conservation after $Nt time steps [$(typeof(arch)), $FT]: " *
          "⟨T⟩-T₀=$(Tavg-Tavg0) °C"

    return isapprox(Tavg, Tavg0, atol=Nx*Ny*Nz*eps(FT))
end

function time_stepping_with_background_fields(arch)

    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))

    background_u(x, y, z, t) = π
    background_v(x, y, z, t) = sin(x) * cos(y) * exp(t)

    background_w_func(x, y, z, t, p) = p.α * x + p.β * exp(z / p.λ)
    background_w = BackgroundField(background_w_func, parameters=(α=1.2, β=0.2, λ=43))

    background_T(x, y, z, t) = background_u(x, y, z, t)

    background_S_func(x, y, z, t, α) = α * y
    background_S = BackgroundField(background_S_func, parameters=1.2)

    model = NonhydrostaticModel(grid=grid, background_fields=(u=background_u, v=background_v, w=background_w,
                                                              T=background_T, S=background_S),
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))

    time_step!(model, 1, euler=true)

    return location(model.background_fields.velocities.u) === (Face, Center, Center) &&
           location(model.background_fields.velocities.v) === (Center, Face, Center) &&
           location(model.background_fields.velocities.w) === (Center, Center, Face) &&
           location(model.background_fields.tracers.T) === (Center, Center, Center) &&
           location(model.background_fields.tracers.S) === (Center, Center, Center)
end

Planes = (FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane)

BuoyancyModifiedAnisotropicMinimumDissipation(FT) = AnisotropicMinimumDissipation(FT, Cb=1.0)

Closures = (ScalarDiffusivity,
            ScalarBiharmonicDiffusivity,
            TwoDimensionalLeith,
            IsopycnalSkewSymmetricDiffusivity,
            SmagorinskyLilly,
            AnisotropicMinimumDissipation,
            BuoyancyModifiedAnisotropicMinimumDissipation,
            CATKEVerticalDiffusivity)

advection_schemes = (nothing,
                     UpwindBiasedFirstOrder(),
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

            model = NonhydrostaticModel(grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1)),
                                        clock = Clock(time=DateTime(2020)))

            time_step!(model, 7.883)
            @test model.clock.time == DateTime("2020-01-01T00:00:07.883")

            model = NonhydrostaticModel(grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1)),
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

    @testset "Euler time stepping propagate NaNs in previous tendency G⁻" begin
        for arch in archs
            @info "  Testing that Euler time stepping doesn't propagate NaNs found in previous tendency G⁻ [$(typeof(arch))]..."
            @test euler_time_stepping_doesnt_propagate_NaNs(arch)
        end
    end

    @testset "Turbulence closures" begin
        for arch in archs, FT in [Float64]

            @info "  Testing that time stepping works [$(typeof(arch)), $FT, nothing]..."
            @test time_stepping_works_with_nothing_closure(arch, FT)

            for Closure in Closures
                @info "  Testing that time stepping works [$(typeof(arch)), $FT, $Closure]..."
                if Closure === TwoDimensionalLeith
                    # TwoDimensionalLeith is slow on the CPU and doesn't compile right now on the GPU.
                    # See: https://github.com/CliMA/Oceananigans.jl/pull/1074
                    @test_skip time_stepping_works_with_closure(arch, FT, Closure)
                elseif Closure === CATKEVerticalDiffusivity
                    # CATKE isn't supported with NonhydrostaticModel yet
                    @test_skip time_stepping_works_with_closure(arch, FT, Closure)
                else
                    @test time_stepping_works_with_closure(arch, FT, Closure)
                end
            end

            # AnisotropicMinimumDissipation can depend on buoyancy...
            @test time_stepping_works_with_closure(arch, FT, AnisotropicMinimumDissipation; buoyancy=nothing)
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
        for FT in float_types, arch in archs
            Nx, Ny, Nz = 32, 32, 32

            regular_grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z=(-1, 1))

            S = 1.3 # Stretching factor
            hyperbolically_spaced_nodes(k) = tanh(S * (2 * (k - 1) / Nz - 1)) / tanh(S)
            hyperbolic_vs_grid = RectilinearGrid(arch, FT,
                                             size = (Nx, Ny, Nz),
                                                x = (0, 1),
                                                y = (0, 1),
                                                z = hyperbolically_spaced_nodes)

            regular_vs_grid = RectilinearGrid(arch, FT,
                                             size = (Nx, Ny, Nz),
                                                x = (0, 1),
                                                y = (0, 1),
                                                z = collect(range(0, stop=1, length=Nz+1)))

            for grid in (regular_grid, hyperbolic_vs_grid, regular_vs_grid)
                @info "  Testing incompressibility [$FT, $(typeof(grid).name.wrapper)]..."

                for Nt in [1, 10, 100], timestepper in timesteppers
                    @test incompressible_in_time(grid, Nt, timestepper)
                end
            end
        end
    end

    @testset "Tracer conservation in channel" begin
        @info "  Testing tracer conservation in channel..."
        for arch in archs, FT in float_types
            @test tracer_conserved_in_channel(arch, FT, 10)
        end
    end
end
