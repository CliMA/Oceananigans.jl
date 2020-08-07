function horizontal_average_is_correct(arch, FT)
    Nx = Ny = Nz = 16
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(100, 100, 100))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = z
    set!(model; T=T₀)

    T̅ = Average(model.tracers.T, dims=(1, 2), time_interval=0.5second)
    computed_profile = dropdims(T̅(model), dims=(1, 2))

    zC = znodes(Cell, grid)
    correct_profile = zC

    return all(computed_profile[2:end-1] .≈ correct_profile)
end

function zonal_average_is_correct(arch, FT)
    Nx = Ny = Nz = 16
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(100, 100, 100))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = z
    set!(model; T=T₀)

    T̅ = Average(model.tracers.T, dims=1, time_interval=0.5second)
    computed_slice = dropdims(T̅(model), dims=(1,))
    zC = znodes(Cell, grid)
    return all([all(computed_slice[i, 2:end-1] .≈ zC) for i in 2:Ny+1])
end

function volume_average_is_correct(arch, FT)
    Nx = Ny = Nz = 16
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(100, 100, 100))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = z
    set!(model; T=T₀)

    T̅ = Average(model.tracers.T, dims=(1, 2, 3), time_interval=0.5second)
    return all(T̅(model) .≈ -50.0)
end

function nan_checker_aborts_simulation(arch, FT)
    grid = RegularCartesianGrid(size=(16, 16, 2), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; iteration_interval=1, fields=Dict(:w => model.velocities.w.data.parent))
    push!(model.diagnostics, nc)

    model.velocities.w[4, 3, 2] = NaN

    time_step!(model, 1, 1)
end

TestModel(::GPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(16, 16, 16), extent=(16Δx, 16Δx, 16Δx)),
       closure = ConstantIsotropicDiffusivity(FT, ν=ν, κ=ν),
  architecture = GPU(),
    float_type = FT
)

TestModel(::CPU, FT, ν=1.0, Δx=0.5) =
    IncompressibleModel(
          grid = RegularCartesianGrid(FT, size=(3, 3, 3), extent=(3Δx, 3Δx, 3Δx)),
       closure = ConstantIsotropicDiffusivity(FT, ν=ν, κ=ν),
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

function timeseries_diagnostic_works(arch, FT)
    model = TestModel(arch, FT)
    Δt = FT(1e-16)
    simulation = Simulation(model, Δt=Δt, stop_iteration=1)
    iter_diag = TimeSeries(get_iteration, model, iteration_interval=1)
    push!(simulation.diagnostics, iter_diag)
    run!(simulation)
    return iter_diag.time[end] == Δt && iter_diag.data[end] == 1
end

function timeseries_diagnostic_tuples(arch, FT)
    model = TestModel(arch, FT)
    Δt = FT(1e-16)
    simulation = Simulation(model, Δt=Δt, stop_iteration=2)
    timeseries = TimeSeries((iters=get_iteration, itertimes=get_time), model, iteration_interval=2)
    simulation.diagnostics[:timeseries] = timeseries
    run!(simulation)
    return timeseries.iters[end] == 2 && timeseries.itertimes[end] == 2Δt
end

function diagnostics_getindex(arch, FT)
    model = TestModel(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)
    iter_timeseries = TimeSeries(get_iteration, model)
    time_timeseries = TimeSeries(get_time, model)
    simulation.diagnostics[:iters] = iter_timeseries
    simulation.diagnostics[:times] = time_timeseries
    return simulation.diagnostics[2] == time_timeseries
end

function diagnostics_setindex(arch, FT)
    model = TestModel(arch, FT)
    simulation = Simulation(model, Δt=0, stop_iteration=0)

    iter_timeseries = TimeSeries(get_iteration, model)
    time_timeseries = TimeSeries(get_time, model)
    max_abs_u_timeseries = TimeSeries(FieldMaximum(abs, model.velocities.u), model, iteration_interval=1)

    push!(simulation.diagnostics, iter_timeseries, time_timeseries)
    simulation.diagnostics[2] = max_abs_u_timeseries

    return simulation.diagnostics[:diag2] == max_abs_u_timeseries
end


@testset "Diagnostics" begin
    @info "Testing diagnostics..."

    for arch in archs
        @testset "Horizontal average [$(typeof(arch))]" begin
            @info "  Testing horizontal average [$(typeof(arch))]"
            for FT in float_types
                @test horizontal_average_is_correct(arch, FT)
            end
        end

        @testset "Zonal average [$(typeof(arch))]" begin
            @info "  Testing zonal average [$(typeof(arch))]"
            for FT in float_types
                @test zonal_average_is_correct(arch, FT)
            end
        end

        @testset "Volume average [$(typeof(arch))]" begin
            @info "  Testing volume average [$(typeof(arch))]"
            for FT in float_types
                @test zonal_average_is_correct(arch, FT)
            end
        end
    end

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
                @test timeseries_diagnostic_works(arch, FT)
                @test timeseries_diagnostic_tuples(arch, FT)
                @test diagnostics_getindex(arch, FT)
                @test diagnostics_setindex(arch, FT)
            end
        end
    end
end
