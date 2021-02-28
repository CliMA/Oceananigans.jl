using Oceananigans.Diagnostics

function nan_checker_aborts_simulation(arch)
    grid = RegularRectilinearGrid(size=(4, 2, 1), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch)
    simulation = Simulation(model, Δt=1, stop_iteration=1)

    model.velocities.u[1, 1, 1] = NaN

    run!(simulation)

    return nothing
end

TestModel(::GPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularRectilinearGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = GPU(),
    float_type = FT
)

TestModel(::CPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularRectilinearGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = CPU(),
    float_type = FT
)

function advective_cfl_diagnostic_is_correct(arch, FT)
    model = TestModel(arch, FT)

    Δt = FT(1.3e-6)
    Δx = FT(model.grid.Δx)
    u₀ = FT(1.2)
    CFL_by_hand = Δt * u₀ / Δx

    model.velocities.u.data.parent .= u₀
    cfl = AdvectiveCFL(FT(Δt))

    return cfl(model) ≈ CFL_by_hand
end

function diffusive_cfl_diagnostic_is_correct(arch, FT)
    Δt = FT(1.3e-6)
    Δx = FT(0.5)
    ν = FT(1.2)
    CFL_by_hand = Δt * ν / Δx^2

    model = TestModel(arch, FT, ν, Δx)
    cfl = DiffusiveCFL(FT(Δt))

    return cfl(model) ≈ CFL_by_hand
end

get_iteration(model) = model.clock.iteration
get_time(model) = model.clock.time

function diagnostics_getindex(arch, FT)
    model = TestModel(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)
    nc = NaNChecker(model, schedule=IterationInterval(1), fields=model.velocities)
    simulation.diagnostics[:nc] = nc

    # The first diagnostic is the NaN checker.
    return simulation.diagnostics[2] == nc
end

function diagnostics_setindex(arch, FT)
    model = TestModel(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)

    nc1 = NaNChecker(model, schedule=IterationInterval(1), fields=model.velocities)
    nc2 = NaNChecker(model, schedule=IterationInterval(2), fields=model.velocities)
    nc3 = NaNChecker(model, schedule=IterationInterval(3), fields=model.velocities)

    push!(simulation.diagnostics, nc1, nc2)
    simulation.diagnostics[2] = nc3

    return simulation.diagnostics[:diag2] == nc3
end

@testset "Diagnostics" begin
    @info "Testing diagnostics..."

    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            @info "  Testing NaN Checker [$(typeof(arch))]"
            @test_throws ErrorException nan_checker_aborts_simulation(arch)
        end
    end

    for arch in archs
        @testset "Miscellaneous timeseries diagnostics [$(typeof(arch))]" begin
            @info "  Testing miscellaneous timeseries diagnostics [$(typeof(arch))]"
            for FT in float_types
                @test diffusive_cfl_diagnostic_is_correct(arch, FT)
                @test advective_cfl_diagnostic_is_correct(arch, FT)
                @test diagnostics_getindex(arch, FT)
                @test diagnostics_setindex(arch, FT)
            end
        end
    end
end
