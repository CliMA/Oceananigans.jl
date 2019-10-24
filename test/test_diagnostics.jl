function horizontal_average_is_correct(arch, FT)
    grid = RegularCartesianGrid(size=(16, 16, 16), length=(100, 100, 100))
    model = Model(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = 20 + 0.01*z
    set!(model; T=T₀)

    T̅ = HorizontalAverage(model.tracers.T; interval=0.5second)
    computed_profile = T̅(model)
    correct_profile = @. 20 + 0.01 * model.grid.zC

    return all(computed_profile[:][2:end-1] .≈ correct_profile)
end

function nan_checker_aborts_simulation(arch, FT)
    grid=RegularCartesianGrid(size=(16, 16, 2), length=(1, 1, 1))
    model = Model(grid=grid, architecture=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; frequency=1, fields=Dict(:w => model.velocities.w.data.parent))
    push!(model.diagnostics, nc)

    model.velocities.w[4, 3, 2] = NaN

    time_step!(model, 1, 1);
end

TestModel(::GPU, FT, ν=1.0, Δx=0.5) = Model(grid = RegularCartesianGrid(FT; size=(16, 16, 16), length=(16Δx, 16Δx, 16Δx)),
                                         closure = ConstantIsotropicDiffusivity(FT; ν=ν, κ=ν),
                                    architecture = GPU(),
                                      float_type = FT)

TestModel(::CPU, FT, ν=1.0, Δx=0.5) = Model(grid = RegularCartesianGrid(FT; size=(3, 3, 3), length=(3Δx, 3Δx, 3Δx)),
                                         closure = ConstantIsotropicDiffusivity(FT; ν=ν, κ=ν),
                                    architecture = CPU(),
                                      float_type = FT)

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
    iter_diag = Timeseries(get_iteration, model; frequency=1)
    push!(model.diagnostics, iter_diag)
    Δt = FT(1e-16)
    time_step!(model, 1, Δt)

    return iter_diag.time[end] == Δt && iter_diag.data[end] == 1
end

function timeseries_diagnostic_tuples(arch, FT)
    model = TestModel(arch, FT)
    timeseries = Timeseries((iters=get_iteration, itertimes=get_time), model; frequency=2)
    model.diagnostics[:timeseries] = timeseries
    Δt = FT(1e-16)
    time_step!(model, 2, Δt)
    return timeseries.iters[end] == 2 && timeseries.itertimes[end] == 2Δt
end

function diagnostics_getindex(arch, FT)
    model = TestModel(arch, FT)
    iter_timeseries = Timeseries(get_iteration, model)
    time_timeseries = Timeseries(get_time, model)
    model.diagnostics[:iters] = iter_timeseries
    model.diagnostics[:times] = time_timeseries
    return model.diagnostics[2] == time_timeseries
end

function diagnostics_setindex(arch, FT)
    model = TestModel(arch, FT)
    iter_timeseries = Timeseries(get_iteration, model)
    time_timeseries = Timeseries(get_time, model)
    max_abs_u_timeseries = Timeseries(FieldMaximum(abs, model.velocities.u), model; frequency=1)

    push!(model.diagnostics, iter_timeseries, time_timeseries)
    model.diagnostics[2] = max_abs_u_timeseries

    return model.diagnostics[:diag2] == max_abs_u_timeseries
end


@testset "Diagnostics" begin
    println("Testing diagnostics...")

    for arch in archs
        @testset "Horizontal average [$(typeof(arch))]" begin
            println("  Testing horizontal average [$(typeof(arch))]")
            for FT in float_types
                @test horizontal_average_is_correct(arch, FT)
            end
        end
    end

    for arch in archs
        @testset "NaN Checker [$(typeof(arch))]" begin
            println("  Testing NaN Checker [$(typeof(arch))]")
            for FT in float_types
                @test_throws ErrorException nan_checker_aborts_simulation(arch, FT)
            end
        end
    end

    for arch in archs
        @testset "Miscellaneous timeseries diagnostics [$(typeof(arch))]" begin
            println("  Testing miscellaneous timeseries diagnostics [$(typeof(arch))]")
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
