using Random
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

function run_cubed_sphere_simulation_test(grid, grid_suffix, FT, arch, cm, cm_suffix)
    if grid_suffix == "UG"
        @info "  Testing simulation on conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
    else
        @info "  Testing simulation on immersed boundary conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
    end

    momentum_advection = WENOVectorInvariant(FT; order=5)
    tracer_advection   = WENO(FT; order=5)
    free_surface       = SplitExplicitFreeSurface(grid; substeps=12)
    coriolis           = HydrostaticSphericalCoriolis(FT)
    tracers            = (:T, :S)
    buoyancy           = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState())

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection,
                                        tracer_advection,
                                        free_surface,
                                        coriolis,
                                        tracers,
                                        buoyancy)

    Random.seed!(1234)
    Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
    Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()
    set!(model, T=Tᵢ, S=Sᵢ)

    simulation = Simulation(model, Δt=1minute, stop_time=10minutes)

    save_fields_interval = 2minute
    checkpointer_interval = 4minutes

    filename_checkpointer =
        "cubed_sphere_checkpointer_$(FT)_$(typeof(arch))_" * cm_suffix * "_" * grid_suffix
    filename_output_writer = "cubed_sphere_output_$(FT)_$(typeof(arch))_" * cm_suffix * "_" * grid_suffix

    # If previous run produced these files, remove them now to ensure a clean test.
    for f in readdir(".")
        if f == filename_output_writer * ".jld2" || occursin(r"^" * filename_checkpointer * r"_.*\.jld2$", f)
            rm(f; force=true)
        end
    end

    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = TimeInterval(checkpointer_interval),
                                                            prefix = filename_checkpointer,
                                                            overwrite_existing = true)

    b = buoyancy_field(model)
    outputs = merge(fields(model), (; b))
    simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                    schedule = TimeInterval(save_fields_interval),
                                                    filename = filename_output_writer,
                                                    verbose = false,
                                                    overwrite_existing = true)

    run!(simulation)

    u = simulation.model.velocities.u
    v = simulation.model.velocities.v
    T = simulation.model.tracers.T
    S = simulation.model.tracers.S
    b = buoyancy_field(simulation.model)

    free_surface_no_halos = SplitExplicitFreeSurface(grid; substeps=12, extend_halos=false)
    model_no_halos = HydrostaticFreeSurfaceModel(grid;
                                                 momentum_advection,
                                                 tracer_advection,
                                                 free_surface = free_surface_no_halos,
                                                 coriolis,
                                                 tracers,
                                                 buoyancy)

    Random.seed!(1234)
    Tᵢ_no_halos(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
    Sᵢ_no_halos(λ, φ, z) = 28 - 5e-3 * z + rand()
    set!(model_no_halos, T=Tᵢ_no_halos, S=Sᵢ_no_halos)

    simulation_no_halos = Simulation(model_no_halos, Δt=1minute, stop_time=10minutes)

    run!(simulation_no_halos)

    u_no_halos = simulation_no_halos.model.velocities.u
    v_no_halos = simulation_no_halos.model.velocities.v
    T_no_halos = simulation_no_halos.model.tracers.T
    S_no_halos = simulation_no_halos.model.tracers.S
    b_no_halos = buoyancy_field(simulation_no_halos.model)

    @apply_regionally regional_comparison = interior(u) ≈ interior(u_no_halos)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(v) ≈ interior(v_no_halos)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(T) ≈ interior(T_no_halos)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(S) ≈ interior(S_no_halos)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(b) ≈ interior(b_no_halos)
    @test all(regional_comparison.regional_objects)

    @test iteration(simulation) == 10
    @test time(simulation) == 10minutes

    u_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "u"; architecture = CPU())
    v_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "v"; architecture = CPU())
    T_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "T"; architecture = CPU())
    S_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "S"; architecture = CPU())
    b_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "b"; architecture = CPU())
    u_end = u_timeseries[end]
    v_end = v_timeseries[end]
    T_end = T_timeseries[end]
    S_end = S_timeseries[end]
    b_end = b_timeseries[end]

    if grid_suffix == "UG"
        @info "  Restarting simulation from pickup file on conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
    else
        @info "  Restarting simulation from pickup file on immersed boundary conformal cubed sphere grid [$FT, $(typeof(arch)), $cm]..."
    end

    simulation = Simulation(model, Δt=1minute, stop_time=10minutes)

    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = TimeInterval(checkpointer_interval),
                                                            prefix = filename_checkpointer,
                                                            overwrite_existing = true)

    b = buoyancy_field(model)
    outputs = merge(fields(model), (; b))
    simulation.output_writers[:fields] = JLD2Writer(model, outputs;
                                                    schedule = TimeInterval(save_fields_interval),
                                                    filename = filename_output_writer,
                                                    verbose = false,
                                                    overwrite_existing = true)

    run!(simulation, pickup = 4)

    @test iteration(simulation) == 10
    @test time(simulation) == 10minutes

    u_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "u"; architecture = CPU())
    v_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "v"; architecture = CPU())
    T_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "T"; architecture = CPU())
    S_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "S"; architecture = CPU())
    b_timeseries = FieldTimeSeries(filename_output_writer * ".jld2", "b"; architecture = CPU())
    u_end_checkpointed_run = u_timeseries[end]
    v_end_checkpointed_run = v_timeseries[end]
    T_end_checkpointed_run = T_timeseries[end]
    S_end_checkpointed_run = S_timeseries[end]
    b_end_checkpointed_run = b_timeseries[end]

    @apply_regionally regional_comparison = interior(u_end) ≈ interior(u_end_checkpointed_run)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(v_end) ≈ interior(v_end_checkpointed_run)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(T_end) ≈ interior(T_end_checkpointed_run)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(S_end) ≈ interior(S_end_checkpointed_run)
    @test all(regional_comparison.regional_objects)
    @apply_regionally regional_comparison = interior(b_end) ≈ interior(b_end_checkpointed_run)
    @test all(regional_comparison.regional_objects)

    return nothing
end
