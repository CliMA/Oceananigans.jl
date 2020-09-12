using Oceananigans.Grids: halo_size

function run_horizontal_average_tests(arch, FT)
    topo = (Periodic, Periodic, Bounded)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    Hx, Hy, Hz = halo_size(grid)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    linear(x, y, z) = z
    set!(model, T=linear, w=linear)

    T̅ = Average(model.tracers.T, dims=(1, 2), with_halos=false)
    computed_T_profile = T̅(model)
    @test_skip size(computed_T_profile) == (1, 1, Nz)
    @test_skip computed_T_profile ≈ znodes(Cell, grid, reshape=true)

    T̅ = Average(model.tracers.T, dims=(1, 2), with_halos=true)
    computed_T_profile_with_halos = T̅(model)
    @test_skip size(computed_T_profile_with_halos) == (1, 1, Nz+2Hz)
    @test_skip computed_T_profile_with_halos[1+Hz:Nz+Hz] ≈ znodes(Cell, grid)

    w̅ = Average(model.velocities.w, dims=(1, 2), with_halos=false)
    computed_w_profile = w̅(model)
    @test_skip size(computed_w_profile) == (1, 1, Nz+1)
    @test_skip computed_w_profile ≈ znodes(Face, grid, reshape=true)

    w̅ = Average(model.velocities.w, dims=(1, 2), with_halos=true)
    computed_w_profile_with_halos = w̅(model)
    @test_skip size(computed_w_profile_with_halos) == (1, 1, Nz+1+2Hz)
    @test_skip computed_w_profile_with_halos[1+Hz:Nz+1+Hz] ≈ znodes(Face, grid)
end

function run_zonal_average_tests(arch, FT)
    topo = (Periodic, Bounded, Bounded)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(topology=topo, size=(Nx, Ny, Nz), extent=(100, 100, 100))
    Hx, Hy, Hz = halo_size(grid)

    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    linear(x, y, z) = z
    set!(model, T=linear, v=linear)

    T̅ = Average(model.tracers.T, dims=1, with_halos=false)
    computed_T_slice = T̅(model)
    @test_skip size(computed_T_slice) == (1, Ny, Nz)

    computed_T_slice = dropdims(computed_T_slice, dims=1)
    zC = znodes(Cell, grid)
    @test_skip all(computed_T_slice[j, :] ≈ zC for j in 1:Ny)

    T̅ = Average(model.tracers.T, dims=1, with_halos=true)
    computed_T_slice_with_halos = T̅(model)
    @test_skip size(computed_T_slice_with_halos) == (1, Ny+2Hy, Nz+2Hz)

    computed_T_slice_with_halos = dropdims(computed_T_slice_with_halos, dims=1)
    @test_skip computed_T_slice_with_halos[1+Hy:Ny+Hy, 1+Hz:Nz+Hz] ≈ computed_T_slice

    v̅ = Average(model.velocities.v, dims=1, with_halos=false)
    computed_v_slice = v̅(model)
    @test_skip size(computed_v_slice) == (1, Ny+1, Nz)

    computed_v_slice = dropdims(computed_v_slice, dims=1)
    zC = znodes(Cell, grid)
    @test_skip all(computed_v_slice[j, :] ≈ zC for j in 1:Ny)

    v̅ = Average(model.velocities.v, dims=1, with_halos=true)
    computed_v_slice_with_halos = v̅(model)
    @test_skip size(computed_v_slice_with_halos) == (1, Ny+1+2Hy, Nz+2Hz)

    computed_v_slice_with_halos = dropdims(computed_v_slice_with_halos, dims=1)
    @test_skip computed_v_slice_with_halos[1+Hy:Ny+1+Hy, 1+Hz:Nz+Hz] ≈ computed_v_slice
end

function run_volume_average_tests(arch, FT)
    Nx = Ny = Nz = 4
    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(100, 100, 100))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    T₀(x, y, z) = z
    set!(model, T=T₀)

    T̅ = Average(model.tracers.T, dims=(1, 2, 3), time_interval=0.5second, with_halos=false)
    computed_scalar = T̅(model)
    @test_skip size(computed_scalar) == (1, 1, 1)
    @test_skip all(computed_scalar .≈ -50.0)

    T̅ = Average(model.tracers.T, dims=(1, 2, 3), time_interval=0.5second, with_halos=true)
    computed_scalar_with_halos = T̅(model)
    @test_skip size(computed_scalar_with_halos) == (1, 1, 1)
    @test_skip all(computed_scalar_with_halos .≈ -50.0)
end

function nan_checker_aborts_simulation(arch, FT)
    grid = RegularCartesianGrid(size=(4, 2, 1), extent=(1, 1, 1))
    model = IncompressibleModel(grid=grid, architecture=arch, float_type=FT)

    # It checks for NaNs in w by default.
    nc = NaNChecker(model; iteration_interval=1, fields=Dict(:w => model.velocities.w.data.parent))
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
        @testset "Average [$(typeof(arch))]" begin
            @info "  Testing averages [$(typeof(arch))]"
            for FT in float_types
                run_horizontal_average_tests(arch, FT)
                run_zonal_average_tests(arch, FT)
                run_volume_average_tests(arch, FT)
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
