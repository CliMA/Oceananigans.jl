include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "dependencies_for_poisson_solvers.jl"))

using Metal
using Oceananigans.Fields: interpolate!
using Oceananigans.OutputReaders: TimeInterpolator
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

Oceananigans.defaults.FloatType = Float32

# Note that these tests are run on a virtualization framework
# via github actions runners and may break in the future.
# More about that:
# * https://github.com/CliMA/Oceananigans.jl/pull/4124#discussion_r1976449272
# * https://github.com/CliMA/Oceananigans.jl/pull/4152


@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa MtlArray
    @test parent(grid.xᶜᵃᵃ) isa MtlArray
    @test eltype(grid) == Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(grid;
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.b) isa MtlArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end

@testset "MetalGPU: ImmersedBoundaryGrid" begin
    arch = GPU(Metal.MetalBackend())
    FT = Float32

    Lx, Ly, Lz = FT(5000meters), FT(5000meters), FT(20meters)
    Nx, Ny, Nz = 16, 16, 8

    underlying_grid = RectilinearGrid(arch;
                                      size=(Nx, Ny, Nz),
                                      x=(0, Lx),
                                      y=(0, Ly),
                                      z=(-Lz, 0),
                                      topology=(Bounded, Bounded, Bounded),
                                      halo=(5, 5, 5))

    @test eltype(underlying_grid) == Float32

    @inline function depth(x, y)
        a² = (0.45*Lx)^2  # semi-major squared axis
        b² = (0.45*Ly)^2  # semi-minor squared axis
        x₀ = Lx / 2
        y₀ = Ly / 2
        Δx = x - x₀
        Δy = y - y₀
        r² = (Δx * Δx) / a² + (Δy * Δy) / b²
        return r² ≤ 1 ? -Lz * (1 - r²) : 1
    end

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(depth); active_cells_map=true)

    @test eltype(grid) == Float32

    Qᵀ = -1e-4 # downward heat flux
    T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵀ))

    Qᵘ = -1e-4
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵘ))

    model = HydrostaticFreeSurfaceModel(grid;
                                        coriolis=FPlane(latitude=60),
                                        tracers=(:T, :S),
                                        buoyancy=SeawaterBuoyancy(),
                                        momentum_advection=WENO(),
                                        tracer_advection=WENO(),
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=30), # default works on MetalGPU as well
                                        closure=ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, background_κz=1e-3),
                                        boundary_conditions=(u=u_bcs, T=T_bcs))

    for field in merge(model.velocities, model.tracers)
        @test parent(field) isa MtlArray
    end

    simulation = Simulation(model, Δt=1, stop_iteration=100)
    run!(simulation)

    @test iteration(simulation) == 100
    @test time(simulation) == 100seconds
    @test maximum(simulation.model.tracers.T) > 0.01
    @test maximum(simulation.model.velocities.u) > 0.01
end

@testset "MetalGPU: test for reductions" begin
    arch = GPU(Metal.MetalBackend())
    grid = RectilinearGrid(arch, size=(32, 32, 32), extent=(1, 1, 1))

    # Test reduction of Field
    c = CenterField(grid)
    set!(c, 1)
    @test minimum(c) == 1

    # Test reduction of KernelFunctionOperation
    add_2(i, j, k, grid, c) = @inbounds c[i, j, k] + 2
    add_2_kfo = KernelFunctionOperation{Center,Center,Center}(add_2, grid, c)
    @test minimum(add_2_kfo) == 3
end

@testset "MetalGPU: set! from a Float64 array" begin
    arch = GPU(Metal.MetalBackend())
    grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))
    c = CenterField(grid)
    a = rand(Float64, 4, 4, 4)
    set!(c, a)
    @test Array(interior(c)) == Float32.(a)
end

@testset "MetalGPU: Float64 numbers in kernel arguments" begin
    wind_stress(x, y, t, p) = p.τ * cos(t / p.T)
    heating(x, y, z, t, p) = p.Q * exp(z / p.h)

    function stepped_model(arch)
        grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
        u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(wind_stress, parameters=(τ=1e-4, T=1day)))
        buoyancy_forcing = (Forcing(heating, parameters=(Q=1e-8, h=0.5)),
                            Relaxation(rate=1/hour, target=1e-3, mask=GaussianMask{:z}(center=-1.0, width=0.2)))

        model = HydrostaticFreeSurfaceModel(grid;
                                            buoyancy = BuoyancyTracer(),
                                            tracers = :b,
                                            boundary_conditions = (; u=u_bcs),
                                            forcing = (; b=buoyancy_forcing))

        for _ in 1:3
            time_step!(model, 1minute)
        end

        return model
    end

    metal_model = stepped_model(GPU(Metal.MetalBackend()))
    cpu_model = stepped_model(CPU())

    @test Array(interior(metal_model.velocities.u)) ≈ interior(cpu_model.velocities.u)
    @test Array(interior(metal_model.tracers.b)) ≈ interior(cpu_model.tracers.b)

    c = CenterField(metal_model.grid)
    set!(c, 1)
    c .= 2.5 .* c .+ 1e-3
    @test all(Array(interior(c)) .≈ 2.501f0)
end

@kernel function _interpolate_in_time!(c, time_series, time_interpolator)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k] = time_series[i, j, k, time_interpolator]
end

@testset "MetalGPU: FieldTimeSeries in kernels" begin
    times = [0, 1hour, 3hours]

    function stepped_tracer(arch)
        grid = RectilinearGrid(arch; size=(4, 4, 4), x=(0, 1000), y=(0, 1000), z=(-100, 0),
                               topology=(Periodic, Periodic, Bounded))

        u = FieldTimeSeries{Face, Center, Center}(grid, times)
        surface_flux = FieldTimeSeries{Center, Center, Nothing}(grid, times)

        for n in eachindex(times)
            set!(u[n], n / 100)
            set!(surface_flux[n], n / 10000)
        end

        c_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(surface_flux))

        model = HydrostaticFreeSurfaceModel(grid;
                                            buoyancy = nothing,
                                            tracers = :c,
                                            velocities = PrescribedVelocityFields(; u),
                                            boundary_conditions = (; c=c_bcs))

        set!(model, c = (x, y, z) -> sin(2π * x / 1000))

        for _ in 1:3
            time_step!(model, 10minutes)
        end

        return Array(interior(model.tracers.c))
    end

    @test stepped_tracer(GPU(Metal.MetalBackend())) ≈ stepped_tracer(CPU())

    # https://github.com/CliMA/Oceananigans.jl/issues/6572
    grid = RectilinearGrid(GPU(Metal.MetalBackend()); size=(4, 4, 1), extent=(1, 1, 1))
    time_series = FieldTimeSeries{Center, Center, Center}(grid, [0, 1hour])
    set!(time_series[1], 1)
    set!(time_series[2], 3)

    c = CenterField(grid)
    time_interpolator = TimeInterpolator(time_series, 30minutes)
    launch!(architecture(grid), grid, :xyz, _interpolate_in_time!, c, time_series, time_interpolator)

    @test all(Array(interior(c)) .== 2)
end

@testset "MetalGPU: TimeStepWizard" begin
    arch = GPU(Metal.MetalBackend())

    grid = RectilinearGrid(arch; size=(64, 64, 16), x=(0, 5000), y=(0, 5000), z=(-20, 0))
    @test eltype(grid) == Float32

    model = HydrostaticFreeSurfaceModel(grid; momentum_advection=WENO(), tracer_advection=WENO(),
                                        free_surface=SplitExplicitFreeSurface(grid; substeps=30))

    sim = Simulation(model, Δt=5, stop_iteration=20)

    wizard = TimeStepWizard(cfl=0.7, min_Δt=1, max_Δt=15)
    sim.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

    run!(sim)
    @test time(sim) > 100seconds
end

@testset "MetalGPU: Nonhydrostatic model" begin
    arch = GPU(Metal.MetalBackend());
    grid = RectilinearGrid(arch; size=(32, 32, 8), x=(0, 5000), y=(0, 5000), z=(-20, 0))
    @test eltype(grid) == Float32

    model = NonhydrostaticModel(grid)
    @test model.pressure_solver isa FFTBasedPoissonSolver

    sim = Simulation(model, Δt=5, stop_iteration=2)
    run!(sim)
    @test time(sim) == 10seconds

    topo = (TX, TY, TZ) =  topology(grid)
    λx = poisson_eigenvalues(grid, grid.Nx, grid.Lx, 1, TX())
    @test eltype(λx) == eltype(grid)
end

@testset "MetalGPU: time step model configurations" begin
    arch = GPU(Metal.MetalBackend())
    grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1))
    implicit_diffusion = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1e-3, κ=1e-3)

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3),
        free_surface in (ExplicitFreeSurface(), ImplicitFreeSurface(), SplitExplicitFreeSurface(grid; substeps=10)),
        closure in (implicit_diffusion, CATKEVerticalDiffusivity())

        model = HydrostaticFreeSurfaceModel(grid; timestepper, free_surface, closure,
                                            buoyancy=BuoyancyTracer(), tracers=:b)
        time_step!(model, 1.0)
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        model = NonhydrostaticModel(grid; timestepper, closure=implicit_diffusion,
                                    buoyancy=BuoyancyTracer(), tracers=:b)
        time_step!(model, 1.0)
    end
end

@testset "MetalGPU: FFT-based Poisson solver" begin
    arch = GPU(Metal.MetalBackend())

    topologies = [(Periodic, Periodic, Periodic),
                  (Periodic, Periodic, Bounded),
                  (Periodic, Bounded, Bounded),
                  (Bounded, Bounded, Bounded)]

    for topo in topologies
        grid = RectilinearGrid(arch; topology=topo, size=(8, 8, 8), extent=(1, 1, 1))
        @test divergence_free_poisson_solution(grid)
    end
end

@testset "MetalGPU: Fourier-tridiagonal Poisson solver" begin
    arch = GPU(Metal.MetalBackend())
    faces = collect(0:8) .^ 1.2
    @test stretched_poisson_solver_correct_answer(Float32, arch, (Periodic, Periodic, Bounded), 8, 8, faces)
end

@testset "MetalGPU: CATKEVerticalDiffusivity" begin
    # https://github.com/CliMA/Oceananigans.jl/issues/5939
    arch = GPU(Metal.MetalBackend())
    grid = RectilinearGrid(arch; size=(8, 8, 16), x=(0, 128), y=(0, 128), z=(-64, 0),
                           topology=(Periodic, Periodic, Bounded))

    @test eltype(grid) == Float32

    Tbcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-4)) # surface cooling

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = WENO(),
                                        tracer_advection = WENO(),
                                        buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
                                        tracers = (:T, :S),
                                        closure = CATKEVerticalDiffusivity(),
                                        boundary_conditions = (; T=Tbcs))

    @test model.clock isa Clock{Float64}
    @test Oceananigans.TimeSteppers.kernel_time_type(model.clock) == Float32

    set!(model, T=(x, y, z) -> 20f0 + 0.01f0 * z, S=35f0)
    set!(model, e=1f-6)

    @test maximum(model.tracers.e) == 1f-6

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes

    @test maximum(model.tracers.e) > 1f-6
    @test maximum(model.tracers.T) < 20
end

@testset "MetalGPU: interpolate! on LatitudeLongitudeGrid" begin
    arch = GPU(Metal.MetalBackend())
    source_grid = LatitudeLongitudeGrid(arch; size=(8, 12, 1), longitude=(-180, 180), latitude=(-60, 60), z=(0, 1))
    target_grid = LatitudeLongitudeGrid(arch; size=(6, 8, 1), longitude=(0, 360), latitude=(-60, 60), z=(0, 1))

    source = CenterField(source_grid)
    target = CenterField(target_grid)
    expected = CenterField(target_grid)
    set!(source, (λ, φ, z) -> φ)
    set!(expected, (λ, φ, z) -> φ)

    interpolate!(target, source)
    @test Array(interior(target)) ≈ Array(interior(expected))
end
