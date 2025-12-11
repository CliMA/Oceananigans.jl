include("dependencies_for_runtests.jl")

using Glob

using Oceananigans: restore_prognostic_state!, prognostic_fields
using Oceananigans.Models.ShallowWaterModels: ShallowWaterScalarDiffusivity
using Oceananigans.OutputWriters: load_checkpoint_state

function test_model_equality(test_model, true_model)
    # Test prognostic field equality
    test_model_fields = prognostic_fields(test_model)
    true_model_fields = prognostic_fields(true_model)
    field_names = keys(test_model_fields)

    for name in field_names
        @test all(test_model_fields[name].data .≈ true_model_fields[name].data)

        if name ∈ keys(test_model.timestepper.Gⁿ)
            @test all(test_model.timestepper.Gⁿ[name].data .≈ true_model.timestepper.Gⁿ[name].data)

            if hasfield(typeof(test_model.timestepper), :G⁻)
                @test all(test_model.timestepper.G⁻[name].data .≈ true_model.timestepper.G⁻[name].data)
            end

            if hasfield(typeof(test_model.timestepper), :Ψ⁻)
                @test all(test_model.timestepper.Ψ⁻[name].data .≈ true_model.timestepper.Ψ⁻[name].data)
            end
        end
    end

    # Test particle equality
    if hasproperty(test_model, :particles) && !isnothing(test_model.particles)
        for name in propertynames(test_model.particles.properties)
            test_prop = getproperty(test_model.particles.properties, name)
            true_prop = getproperty(true_model.particles.properties, name)
            @test all(Array(test_prop) .≈ Array(true_prop))
        end
    end

    # Test free surface equality
    if hasproperty(test_model, :free_surface) && test_model.free_surface isa SplitExplicitFreeSurface
        fs_test = test_model.free_surface
        fs_true = true_model.free_surface
        @test all(interior(fs_test.barotropic_velocities.U) .≈ interior(fs_true.barotropic_velocities.U))
        @test all(interior(fs_test.barotropic_velocities.V) .≈ interior(fs_true.barotropic_velocities.V))
        @test all(interior(fs_test.filtered_state.η̅)        .≈ interior(fs_true.filtered_state.η̅))
        @test all(interior(fs_test.filtered_state.U̅)        .≈ interior(fs_true.filtered_state.U̅))
        @test all(interior(fs_test.filtered_state.V̅)        .≈ interior(fs_true.filtered_state.V̅))

        # Check free surface timestepper fields (for AdamsBashforth3Scheme)
        if hasproperty(fs_test.timestepper, :ηᵐ)
            ts_test = fs_test.timestepper
            ts_true = fs_true.timestepper
            @test all(interior(ts_test.ηᵐ)   .≈ interior(ts_true.ηᵐ))
            @test all(interior(ts_test.ηᵐ⁻¹) .≈ interior(ts_true.ηᵐ⁻¹))
            @test all(interior(ts_test.ηᵐ⁻²) .≈ interior(ts_true.ηᵐ⁻²))
            @test all(interior(ts_test.Uᵐ⁻¹) .≈ interior(ts_true.Uᵐ⁻¹))
            @test all(interior(ts_test.Uᵐ⁻²) .≈ interior(ts_true.Uᵐ⁻²))
            @test all(interior(ts_test.Vᵐ⁻¹) .≈ interior(ts_true.Vᵐ⁻¹))
            @test all(interior(ts_test.Vᵐ⁻²) .≈ interior(ts_true.Vᵐ⁻²))
        end
    end

    # Test auxiliary fields equality
    if hasproperty(test_model, :auxiliary_fields) && length(test_model.auxiliary_fields) > 0
        for name in keys(test_model.auxiliary_fields)
            @test all(interior(test_model.auxiliary_fields[name]) .≈ interior(true_model.auxiliary_fields[name]))
        end
    end

    return nothing
end

function test_minimal_restore_nonhydrostatic(arch, FT, pickup_method)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
        size = (N, N, N),
        topology = (Periodic, Bounded, Bounded),
        extent = (L, L, L)
    )

    model = NonhydrostaticModel(; grid)
    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(
        model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    @test isfile("$(prefix)_iteration0.jld2")
    @test isfile("$(prefix)_iteration1.jld2")
    @test isfile("$(prefix)_iteration2.jld2")
    @test isfile("$(prefix)_iteration3.jld2")

    grid = nothing
    model = nothing
    simulation = nothing
    checkpointer = nothing

    new_grid = RectilinearGrid(arch, FT,
        size = (N, N, N),
        topology = (Periodic, Bounded, Bounded),
        extent = (L, L, L)
    )

    new_model = NonhydrostaticModel(; grid=new_grid)
    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(
        new_model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        pickup = true
    elseif pickup_method == :iteration
        pickup = 3
    elseif pickup_method == :filepath
        pickup = "$(prefix)_iteration3.jld2"
    end

    @test_nowarn set!(new_simulation, pickup)

    @test iteration(new_simulation) == 3
    @test time(new_simulation) == 3.0

    @test new_checkpointer.schedule.actuations == 3

    rm.(glob("$(prefix)_iteration*.jld2"))

    return nothing
end

function test_checkpointer_cleanup(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(; grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=0.2, stop_iteration=10)

    prefix = "checkpointer_cleanup_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(
        model;
        schedule = IterationInterval(3),
        prefix = prefix,
        cleanup = true
    )

    @test_nowarn run!(simulation)

    [@test !isfile("$(prefix)_iteration$i.jld2") for i in 1:10 if i != 9]
    @test isfile("$(prefix)_iteration9.jld2")

    rm("$(prefix)_iteration9.jld2", force=true)

    return nothing
end

function test_thermal_bubble_checkpointing_nonhydrostatic(arch, timestepper)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = NonhydrostaticModel(; grid, timestepper,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model, T=bubble, S=bubble)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    checkpointer = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = "thermal_bubble_checkpointing_$(typeof(arch))"
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = NonhydrostaticModel(; timestepper,
        grid = new_grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_checkpointer = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = "thermal_bubble_checkpointing_$(typeof(arch))"
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    return nothing
end

function test_minimal_restore_hydrostatic(arch, FT, pickup_method)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
        size = (N, N, N),
        topology = (Periodic, Bounded, Bounded),
        extent = (L, L, L)
    )

    model = HydrostaticFreeSurfaceModel(; grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_hydrostatic_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(
        model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    @test isfile("$(prefix)_iteration0.jld2")
    @test isfile("$(prefix)_iteration1.jld2")
    @test isfile("$(prefix)_iteration2.jld2")
    @test isfile("$(prefix)_iteration3.jld2")

    grid = nothing
    model = nothing
    simulation = nothing
    checkpointer = nothing

    new_grid = RectilinearGrid(arch, FT,
        size = (N, N, N),
        topology = (Periodic, Bounded, Bounded),
        extent = (L, L, L)
    )

    new_model = HydrostaticFreeSurfaceModel(; grid=new_grid, buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(
        new_model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        pickup = true
    elseif pickup_method == :iteration
        pickup = 3
    elseif pickup_method == :filepath
        pickup = "$(prefix)_iteration3.jld2"
    end

    @test_nowarn set!(new_simulation, pickup)

    @test iteration(new_simulation) == 3
    @test time(new_simulation) == 3.0

    @test new_checkpointer.schedule.actuations == 3

    rm.(glob("$(prefix)_iteration*.jld2"))

    return nothing
end

function test_thermal_bubble_checkpointing_hydrostatic(arch, timestepper)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = HydrostaticFreeSurfaceModel(; grid, timestepper,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model, T=bubble, S=bubble)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    checkpointer = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = "thermal_bubble_checkpointing_hydrostatic_$(typeof(arch))"
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = HydrostaticFreeSurfaceModel(; timestepper,
        grid = new_grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_checkpointer = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = "thermal_bubble_checkpointing_hydrostatic_$(typeof(arch))"
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    return nothing
end

function test_minimal_restore_shallow_water(arch, FT, pickup_method)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
        size = (N, N),
        topology = (Periodic, Periodic, Flat),
        extent = (L, L)
    )

    model = ShallowWaterModel(; grid, gravitational_acceleration=1)
    set!(model, h=1)
    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_shallow_water_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(
        model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    @test isfile("$(prefix)_iteration0.jld2")
    @test isfile("$(prefix)_iteration1.jld2")
    @test isfile("$(prefix)_iteration2.jld2")
    @test isfile("$(prefix)_iteration3.jld2")

    grid = nothing
    model = nothing
    simulation = nothing
    checkpointer = nothing

    new_grid = RectilinearGrid(arch, FT,
        size = (N, N),
        topology = (Periodic, Periodic, Flat),
        extent = (L, L)
    )

    new_model = ShallowWaterModel(; grid=new_grid, gravitational_acceleration=1)
    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(
        new_model;
        schedule = TimeInterval(1.0),
        prefix = prefix,
        cleanup = false,
        verbose = true
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        pickup = true
    elseif pickup_method == :iteration
        pickup = 3
    elseif pickup_method == :filepath
        pickup = "$(prefix)_iteration3.jld2"
    end

    @test_nowarn set!(new_simulation, pickup)

    @test iteration(new_simulation) == 3
    @test time(new_simulation) == 3.0

    @test new_checkpointer.schedule.actuations == 3

    rm.(glob("$(prefix)_iteration*.jld2"))

    return nothing
end

function test_height_perturbation_checkpointing_shallow_water(arch, timestepper)
    Nx, Ny = 16, 16
    Lx, Ly = 100, 100
    Δt = 6

    grid = RectilinearGrid(arch, size=(Nx, Ny), extent=(Lx, Ly), topology=(Periodic, Periodic, Flat))
    model = ShallowWaterModel(; grid, timestepper,
        gravitational_acceleration = 1,
        closure = ShallowWaterScalarDiffusivity(ν=4e-2, ξ=0)
    )

    # Gaussian height perturbation (analogous to thermal bubble)
    perturbation(x, y) = 1 + 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2) / (Lx^2 + Ly^2))
    set!(model, h=perturbation)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    checkpointer = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = "height_perturbation_checkpointing_shallow_water_$(typeof(arch))"
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(Nx, Ny), extent=(Lx, Ly), topology=(Periodic, Periodic, Flat))
    new_model = ShallowWaterModel(; timestepper,
        grid = new_grid,
        gravitational_acceleration = 1,
        closure = ShallowWaterScalarDiffusivity(ν=4e-2, ξ=0)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_checkpointer = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = "height_perturbation_checkpointing_shallow_water_$(typeof(arch))"
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    return nothing
end

function test_checkpointing_split_explicit_free_surface(arch, timestepper)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1000, 1000, 100
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    free_surface = SplitExplicitFreeSurface(grid; substeps=30)

    model = HydrostaticFreeSurfaceModel(; grid, timestepper, free_surface,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model, T=bubble, S=bubble)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "split_explicit_checkpointing_$(typeof(arch))_$(timestepper)"
    checkpointer = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    simulation.output_writers[:checkpointer] = checkpointer

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_free_surface = SplitExplicitFreeSurface(new_grid; substeps=30)

    new_model = HydrostaticFreeSurfaceModel(; timestepper,
        grid = new_grid,
        free_surface = new_free_surface,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_checkpointer = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

        end
    end


    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper]..."
            test_checkpointing_split_explicit_free_surface(arch, timestepper)
        end
    end

