include("dependencies_for_runtests.jl")

using oneAPI
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# Float64 model kernels exceed Intel's 2 KiB kernel-argument limit (and Arc GPUs
# only emulate Float64), so unlike the AMDGPU tests these are Float32-only
oneapi_float_types = (Float32,)

function build_and_timestep_simulation(model)
    FT = eltype(model)

    for field in merge(model.velocities, model.tracers)
        @test parent(field) isa oneArray
    end

    simulation = Simulation(model, Δt=1minute, stop_iteration=3, verbose=false)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) ≈ FT(3minutes)

    return nothing
end

@testset "oneAPI device selection" begin
    arch = GPU(oneAPI.oneAPIBackend())
    original_device = oneAPI.device()

    try
        Oceananigans.Architectures.device!(arch, 0)
        @test oneAPI.device() == first(oneAPI.devices())
    finally
        oneAPI.device!(original_device)
    end
end

@testset "oneAPI on RectilinearGrids" begin
    oneapi = oneAPI.oneAPIBackend()
    arch = GPU(oneapi)

    for FT in oneapi_float_types
        Oceananigans.defaults.FloatType = FT
        @info "Testing grids on $arch with $FT..."

        regular_grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=(0, 4), y=(0, 1), z=(0, 16))
        horizontally_stretched_grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))
        vertically_stretched_grid = RectilinearGrid(arch, FT, size=(16, 8, 4), x=(0, 16), y=(0, 1), z=[0, 1, 2, 3, 4])

        # oneMKL FFTs are full-dimension only, so the default FFT-based pressure
        # solver requires triple periodicity
        triply_periodic_grid = RectilinearGrid(arch, FT, size=(4, 8, 16), x=(0, 4), y=(0, 1), z=(0, 16),
                                               topology=(Periodic, Periodic, Periodic))

        @test parent(horizontally_stretched_grid.xᶠᵃᵃ) isa oneArray
        @test parent(horizontally_stretched_grid.xᶜᵃᵃ) isa oneArray

        @test parent(vertically_stretched_grid.z.cᵃᵃᶠ) isa oneArray
        @test parent(vertically_stretched_grid.z.cᵃᵃᶜ) isa oneArray
        @test parent(vertically_stretched_grid.z.Δᵃᵃᶠ) isa oneArray
        @test parent(vertically_stretched_grid.z.Δᵃᵃᶜ) isa oneArray

        for grid in (regular_grid, horizontally_stretched_grid, vertically_stretched_grid, triply_periodic_grid)
            @test eltype(grid) == FT
            @test architecture(grid) isa GPU
        end

        @info "Testing HydrostaticFreeSurfaceModel on $arch with $FT..."

        coriolis = FPlane(latitude=45)
        buoyancy = BuoyancyTracer()
        tracers = :b
        advection = WENO(order=5)

        for grid in (regular_grid, horizontally_stretched_grid, vertically_stretched_grid)
            momentum_advection = tracer_advection = advection

            free_surface = SplitExplicitFreeSurface(grid; substeps=60)

            model = HydrostaticFreeSurfaceModel(grid; free_surface,
                                                coriolis, buoyancy, tracers,
                                                momentum_advection, tracer_advection)

            build_and_timestep_simulation(model)
        end

        @info "Testing NonhydrostaticModel on $arch with $FT..."

        model_kw = (; coriolis, buoyancy, tracers, advection)

        # With default pressure solver
        model = NonhydrostaticModel(triply_periodic_grid; model_kw...)
        build_and_timestep_simulation(model)

        # With CG pressure solver
        for grid in (regular_grid, vertically_stretched_grid)
            cg_solver = Oceananigans.Solvers.ConjugateGradientPoissonSolver(grid;
                maxiter=10, reltol=1e-7, abstol=1e-7, preconditioner=nothing)

            model = NonhydrostaticModel(grid; pressure_solver=cg_solver, model_kw...)
            build_and_timestep_simulation(model)
        end

        # oneMKL cannot plan partial-dimension FFTs: once this stops throwing, the
        # topology restriction above can be lifted
        @test_throws ErrorException NonhydrostaticModel(regular_grid; model_kw...)
    end

    Oceananigans.defaults.FloatType = Float64
end

@testset "oneAPI on LatitudeLongitudeGrid with HydrostaticFreeSurfaceModel" begin
    oneapi = oneAPI.oneAPIBackend()
    arch = GPU(oneapi)

    for FT in oneapi_float_types
        Oceananigans.defaults.FloatType = FT
        @info "    Testing on $arch with $FT"

        grid = LatitudeLongitudeGrid(arch, FT, size=(4, 8, 16), longitude=(-60, 60), latitude=(0, 60), z=(0, 1))

        @test parent(grid.Δxᶜᶜᵃ) isa oneArray
        @test parent(grid.Δxᶠᶜᵃ) isa oneArray
        @test parent(grid.Δxᶜᶠᵃ) isa oneArray
        @test parent(grid.Δxᶠᶠᵃ) isa oneArray
        @test parent(grid.Azᶜᶜᵃ) isa oneArray
        @test parent(grid.Azᶠᶜᵃ) isa oneArray
        @test parent(grid.Azᶜᶠᵃ) isa oneArray
        @test parent(grid.Azᶠᶠᵃ) isa oneArray
        @test eltype(grid) == FT
        @test architecture(grid) isa GPU

        equation_of_state = TEOS10EquationOfState(FT)
        buoyancy = SeawaterBuoyancy(; equation_of_state)

        # Fewer substeps than the AMDGPU test: the substep-weight tuple is a kernel
        # argument, and with 60 substeps this model overflows Intel's 2 KiB limit
        model = HydrostaticFreeSurfaceModel(grid; buoyancy,
                                            coriolis = FPlane(latitude=45),
                                            tracers = (:T, :S),
                                            momentum_advection = WENO(order=5),
                                            tracer_advection = WENO(order=5),
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=10))

        build_and_timestep_simulation(model)
    end

    Oceananigans.defaults.FloatType = Float64
end
