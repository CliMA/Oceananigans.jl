using Oceananigans.Diagnostics

function nan_checker_aborts_simulation(arch, FT)
    grid = RegularCartesianGrid(size=(4, 2, 1), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; schedule=IterationInterval(1), fields=Dict(:w => model.velocities.w.data.parent))
    push!(model.diagnostics, nc)

    model.velocities.w[3, 2, 1] = NaN

    time_step!(model, 1, 1)
end

TestModel(::GPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = GPU(),
    float_type = FT
)

TestModel(::CPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = IsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = CPU(),
    float_type = FT
)

function max_abs_field_diagnostic_is_correct(arch, FT)
    model = TestModel(arch, FT)
    set!(model.velocities.u, rand(size(model.grid)))
    u_max = FieldMaximum(abs, model.velocities.u)
    return u_max(model) == maximum(abs, model.velocities.u.data.parent)
end

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
    nc = NaNChecker(model; schedule=IterationInterval(1), fields=Dict(:w => model.velocities.w.data.parent))
    simulation.diagnostics[:nc] = nc
    return simulation.diagnostics[1] == nc
end

function diagnostics_setindex(arch, FT)
    model = TestModel(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)

    nc1 = NaNChecker(model; schedule=IterationInterval(1), fields=Dict(:w => model.velocities.w.data.parent))
    nc2 = NaNChecker(model; schedule=IterationInterval(2), fields=Dict(:u => model.velocities.u.data.parent))
    nc3 = NaNChecker(model; schedule=IterationInterval(3), fields=Dict(:v => model.velocities.v.data.parent))

    push!(simulation.diagnostics, nc1, nc2)
    simulation.diagnostics[2] = nc3

    return simulation.diagnostics[:diag2] == nc3
end

@testset "Diagnostics" begin
    @info "Testing diagnostics..."

    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            @info "  Testing NaN Checker [$(typeof(arch))]"
            for FT in float_types
                @test_throws ErrorException nan_checker_aborts_simulation(arch, FT)
            end
        end
    end

    for arch in archs
        @testset "Miscellaneous timeseries diagnostics [$(typeof(arch))]" begin
            @info "  Testing miscellaneous timeseries diagnostics [$(typeof(arch))]"
            for FT in float_types
                @test diffusive_cfl_diagnostic_is_correct(arch, FT)
                @test advective_cfl_diagnostic_is_correct(arch, FT)
                @test max_abs_field_diagnostic_is_correct(arch, FT)
                @test diagnostics_getindex(arch, FT)
                @test diagnostics_setindex(arch, FT)
            end
        end
    end
end
