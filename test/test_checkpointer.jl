include("dependencies_for_runtests.jl")

using Glob
using NCDatasets

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
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Create new model with same setup
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_free_surface = SplitExplicitFreeSurface(new_grid; substeps=30)

    new_model = HydrostaticFreeSurfaceModel(; timestepper,
        grid = new_grid,
        free_surface = new_free_surface,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_implicit_free_surface(arch, solver_method)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1000, 1000, 1000
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz),
                          topology=(Bounded, Bounded, Bounded))

    free_surface = ImplicitFreeSurface(solver_method=solver_method)

    model = HydrostaticFreeSurfaceModel(; grid, free_surface,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model, T=bubble, S=bubble)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "implicit_free_surface_checkpointing_$(typeof(arch))_$(solver_method)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Create new model with same setup
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz),
                              topology=(Bounded, Bounded, Bounded))

    new_free_surface = ImplicitFreeSurface(solver_method=solver_method)

    new_model = HydrostaticFreeSurfaceModel(;
        grid = new_grid,
        free_surface = new_free_surface,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefixdefault_included_properties)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_lagrangian_particles(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.01

    grid = RectilinearGrid(arch,
        size = (Nx, Ny, Nz),
        extent = (Lx, Ly, Lz),
        topology = (Periodic, Periodic, Bounded)
    )

    P = 10  # number of particles
    xs = on_architecture(arch, 0.5 * ones(P))
    ys = on_architecture(arch, 0.5 * ones(P))
    zs = on_architecture(arch, -0.5 * ones(P))

    particles = LagrangianParticles(x=xs, y=ys, z=zs)

    model = NonhydrostaticModel(; grid, timestepper, particles,
        closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)
    )

    # Set some initial velocity to move particles
    set!(model, u=1, v=0.5, w=0)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "lagrangian_particles_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Create new model with same setup
    new_grid = RectilinearGrid(arch,
        size = (Nx, Ny, Nz),
        extent = (Lx, Ly, Lz),
        topology = (Periodic, Periodic, Bounded)
    )

    new_xs = on_architecture(arch, 0.5 * ones(P))
    new_ys = on_architecture(arch, 0.5 * ones(P))
    new_zs = on_architecture(arch, -0.5 * ones(P))
    new_particles = LagrangianParticles(x=new_xs, y=new_ys, z=new_zs)

    new_model = NonhydrostaticModel(; timestepper,
        grid = new_grid,
        particles = new_particles,
        closure = ScalarDiffusivity(ν=1e-4, κ=1e-4)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_immersed_boundary_grid(arch, boundary_type)
    Nx, Ny, Nz = 16, 16, 8
    Lx, Ly, Lz = 100, 100, 50
    Δt = 0.1

    underlying_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

    bottom(x, y) = -40 + 10 * sin(2π * x / Lx)

    if boundary_type == :GridFittedBottom
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
    elseif boundary_type == :PartialCellBottom
        grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))
    end

    model = NonhydrostaticModel(; grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z + Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model, T=bubble, S=bubble)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "immersed_boundary_checkpointing_$(typeof(arch))_$(boundary_type)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_underlying_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    if boundary_type == :GridFittedBottom
        new_grid = ImmersedBoundaryGrid(new_underlying_grid, GridFittedBottom(bottom))
    elseif boundary_type == :PartialCellBottom
        new_grid = ImmersedBoundaryGrid(new_underlying_grid, PartialCellBottom(bottom))
    end

    new_model = NonhydrostaticModel(;
        grid = new_grid,
        closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_latitude_longitude_grid(arch)
    # Use parameters that ensure numerical stability
    Nx, Ny, Nz = 8, 8, 4
    Δt = 300  # 5 minute timestep

    grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz),
                                 longitude=(0, 60),
                                 latitude=(-30, 30),
                                 z=(-1000, 0))

    free_surface = SplitExplicitFreeSurface(grid; substeps=30)

    model = HydrostaticFreeSurfaceModel(; grid, free_surface,
        coriolis = HydrostaticSphericalCoriolis(),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    # Stable initial conditions: linear temperature profile
    T_init(λ, φ, z) = 20 + 5 * (z + 1000) / 1000
    set!(model, T=T_init, S=35)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "lat_lon_grid_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz),
                                     longitude=(0, 60),
                                     latitude=(-30, 30),
                                     z=(-1000, 0))

    new_free_surface = SplitExplicitFreeSurface(new_grid; substeps=30)

    new_model = HydrostaticFreeSurfaceModel(;
        grid = new_grid,
        free_surface = new_free_surface,
        coriolis = HydrostaticSphericalCoriolis(),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_float32(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, Float32, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)

    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Float32(Δt), stop_iteration=5)

    prefix = "float32_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, Float32, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid)

    new_simulation = Simulation(new_model, Δt=Float32(Δt), stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_auxiliary_fields(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))

    auxiliary_fields = (custom_field = CenterField(grid),)

    model = NonhydrostaticModel(; grid, auxiliary_fields)

    # Set custom_field data
    set!(model.auxiliary_fields.custom_field, (x, y, z) -> x + y + z)
    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "auxiliary_fields_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_auxiliary_fields = (custom_field = CenterField(new_grid),)
    new_model = NonhydrostaticModel(; grid=new_grid, auxiliary_fields=new_auxiliary_fields)

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_closure_fields(arch)
    N = 8
    L = 1
    Δt = 0.01

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))

    closure = SmagorinskyLilly()

    model = NonhydrostaticModel(; grid, closure,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    # Set initial conditions to generate turbulent closure fields
    u₀(x, y, z) = sin(2π*x)
    v₀(x, y, z) = cos(2π*y)
    T₀(x, y, z) = 20
    S₀(x, y, z) = 35
    set!(model, u=u₀, v=v₀, T=T₀, S=S₀)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "closure_fields_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(;
        grid = new_grid,
        closure = SmagorinskyLilly(),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_catke_closure(arch)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 100, 100, 100
    Δt = 60

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = CATKEVerticalDiffusivity()

    model = HydrostaticFreeSurfaceModel(; grid, closure,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S, :e)
    )

    # Linear stratification + noisy velocity to generate TKE
    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.01 * randn()
    set!(model, T=T_init, S=35, u=u_init)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "catke_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Create new model and restore
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = HydrostaticFreeSurfaceModel(;
        grid = new_grid,
        closure = CATKEVerticalDiffusivity(),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S, :e)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_dynamic_smagorinsky_closure(arch)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.001

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = DynamicSmagorinsky()

    model = NonhydrostaticModel(; grid, closure)

    # Sheared flow to generate non-trivial dynamic coefficients
    u_init(x, y, z) = sin(2π * z / Lz)
    set!(model, u=u_init)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "dynamic_smagorinsky_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Get original closure field state
    original_cf = model.closure_fields
    original_previous_time = original_cf.previous_compute_time[]
    original_𝒥ᴸᴹ = copy(Array(interior(original_cf.𝒥ᴸᴹ)))
    original_𝒥ᴹᴹ = copy(Array(interior(original_cf.𝒥ᴹᴹ)))

    # Create new model and restore
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = NonhydrostaticModel(; grid=new_grid, closure=DynamicSmagorinsky())

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    # Verify closure field state was restored
    new_cf = new_model.closure_fields
    @test new_cf.previous_compute_time[] ≈ original_previous_time
    @test all(Array(interior(new_cf.𝒥ᴸᴹ)) .≈ original_𝒥ᴸᴹ)
    @test all(Array(interior(new_cf.𝒥ᴹᴹ)) .≈ original_𝒥ᴹᴹ)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_ri_based_closure(arch)
    Nx, Ny, Nz = 8, 8, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 60

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = RiBasedVerticalDiffusivity(Cᵃᵛ=0.6)  # Time averaging enabled

    model = HydrostaticFreeSurfaceModel(; grid, closure,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    # Stratified with shear to generate non-trivial Ri-based diffusivities
    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.1 * z / Lz
    set!(model, T=T_init, S=35, u=u_init)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "ri_based_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Get original closure field state
    original_κc = copy(Array(interior(model.closure_fields.κc)))
    original_κu = copy(Array(interior(model.closure_fields.κu)))

    # Create new model and restore
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = HydrostaticFreeSurfaceModel(;
        grid = new_grid,
        closure = RiBasedVerticalDiffusivity(Cᵃᵛ=0.6),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    # Verify closure field state was restored
    @test all(Array(interior(new_model.closure_fields.κc)) .≈ original_κc)
    @test all(Array(interior(new_model.closure_fields.κu)) .≈ original_κu)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_tke_dissipation_closure(arch)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 100, 100, 100
    Δt = 60

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = TKEDissipationVerticalDiffusivity()

    model = HydrostaticFreeSurfaceModel(; grid, closure,
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S, :e, :ϵ)
    )

    # Linear stratification + noisy velocity to generate turbulence
    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.01 * randn()
    set!(model, T=T_init, S=35, u=u_init)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "tke_dissipation_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Store original previous_velocities state
    original_u⁻ = copy(Array(interior(model.closure_fields.previous_velocities.u)))
    original_v⁻ = copy(Array(interior(model.closure_fields.previous_velocities.v)))

    # Create new model and restore
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = HydrostaticFreeSurfaceModel(;
        grid = new_grid,
        closure = TKEDissipationVerticalDiffusivity(),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S, :e, :ϵ)
    )

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    # Verify previous_velocities state was restored
    @test all(Array(interior(new_model.closure_fields.previous_velocities.u)) .≈ original_u⁻)
    @test all(Array(interior(new_model.closure_fields.previous_velocities.v)) .≈ original_v⁻)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_continuation_matches_direct(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.01

    # Run A: Direct run for 10 iterations
    grid_A = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_A = NonhydrostaticModel(; grid=grid_A, timestepper,
        closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model_A, T=bubble, S=bubble, u=0.1)

    simulation_A = Simulation(model_A, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(simulation_A)

    # Run B: Run 5 iterations, checkpoint, restore, run 5 more
    grid_B = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B = NonhydrostaticModel(; grid=grid_B, timestepper,
        closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    set!(model_B, T=bubble, S=bubble, u=0.1)

    simulation_B = Simulation(model_B, Δt=Δt, stop_iteration=5)

    prefix = "continuation_test_$(typeof(arch))_$(timestepper)"
    simulation_B.output_writers[:checkpointer] = Checkpointer(model_B,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation_B)

    # Create fresh model and restore from checkpoint
    grid_B_new = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B_new = NonhydrostaticModel(; grid=grid_B_new, timestepper,
        closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
        buoyancy = SeawaterBuoyancy(),
        tracers = (:T, :S)
    )

    simulation_B_new = Simulation(model_B_new, Δt=Δt, stop_iteration=10)

    simulation_B_new.output_writers[:checkpointer] = Checkpointer(model_B_new,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(simulation_B_new, true)

    # Continue running for 5 more iterations (to iteration 10)
    @test_nowarn run!(simulation_B_new)

    # Verify both models have the same final state
    @test iteration(simulation_A) == iteration(simulation_B_new) == 10

    fields_A = prognostic_fields(model_A)
    fields_B = prognostic_fields(model_B_new)

    for name in keys(fields_A)
        @test all(fields_A[name].data .≈ fields_B[name].data)
    end

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_empty_tracers(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid, tracers=())

    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "empty_tracers_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid, tracers=())

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=5)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, true)

    test_model_equality(new_model, model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_missing_file_warning(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    # Use a unique prefix that doesn't have any checkpoint files
    prefix = "nonexistent_checkpoint_$(typeof(arch))_$(rand(1:100000))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(5),
        prefix = prefix
    )

    # Should warn but not error when no checkpoint files exist
    @test_logs (:warn,) set!(simulation, true)

    # Simulation should still be at iteration 0
    @test iteration(simulation) == 0

    return nothing
end

function test_stateful_schedule_checkpointing(arch, schedule_type)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)
    set!(model, u=1, v=0.5)

    if schedule_type == :SpecifiedTimes
        schedule = SpecifiedTimes(0.5, 1.0, 1.5, 2.0)
    elseif schedule_type == :ConsecutiveIterations
        schedule = ConsecutiveIterations(TimeInterval(0.5))
    elseif schedule_type == :TimeInterval
        schedule = TimeInterval(0.5)
    elseif schedule_type == :WallTimeInterval
        schedule = WallTimeInterval(0.5)  # 0.5 seconds
    end

    simulation = Simulation(model, Δt=Δt, stop_iteration=15)

    prefix = "schedule_checkpointing_$(typeof(arch))_$(schedule_type)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(10),
        prefix = prefix
    )

    # We will test the schedule via a callback that does nothing.
    simulation.callbacks[:test_schedule] = Callback(_ -> nothing, schedule)

    @test_nowarn run!(simulation)

    original_schedule = simulation.callbacks[:test_schedule].schedule

    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid)

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=15)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(10),
        prefix = prefix
    )

    # Add callback with fresh schedule
    if schedule_type == :SpecifiedTimes
        new_schedule = SpecifiedTimes(0.5, 1.0, 1.5, 2.0)
    elseif schedule_type == :ConsecutiveIterations
        new_schedule = ConsecutiveIterations(TimeInterval(0.5))
    elseif schedule_type == :TimeInterval
        new_schedule = TimeInterval(0.5)
    elseif schedule_type == :WallTimeInterval
        new_schedule = WallTimeInterval(0.5)
    end

    new_simulation.callbacks[:test_schedule] = Callback(_ -> nothing, new_schedule)

    @test_nowarn set!(new_simulation, true)

    # Run the restored simulation to completion
    @test_nowarn run!(new_simulation)

    # Both should be at iteration 15
    original_schedule = simulation.callbacks[:test_schedule].schedule
    restored_schedule = new_simulation.callbacks[:test_schedule].schedule

    if schedule_type == :SpecifiedTimes
        @test restored_schedule.previous_actuation == original_schedule.previous_actuation
    elseif schedule_type == :ConsecutiveIterations
        @test restored_schedule.previous_parent_actuation_iteration == original_schedule.previous_parent_actuation_iteration
        @test restored_schedule.parent.actuations == original_schedule.parent.actuations
    elseif schedule_type == :TimeInterval
        @test restored_schedule.first_actuation_time == original_schedule.first_actuation_time
        @test restored_schedule.actuations == original_schedule.actuations
    elseif schedule_type == :WallTimeInterval
        @test restored_schedule.interval == original_schedule.interval
    end

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_windowed_time_average_checkpointing(arch, WriterType)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model = NonhydrostaticModel(; grid)

    # Set up initial conditions that will produce non-trivial averages
    u_init(x, y, z) = sin(2π * x / Lx)
    set!(model, u=u_init)

    simulation = Simulation(model, Δt=Δt, stop_iteration=8)

    # Writer-specific settings
    if WriterType == JLD2Writer
        prefix = "wta_checkpointing_jld2_$(typeof(arch))"
        ext = ".jld2"
        output_key = :u
    else  # NetCDFWriter
        prefix = "wta_checkpointing_netcdf_$(typeof(arch))"
        ext = ".nc"
        output_key = "u"
    end

    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(8),
        prefix = prefix
    )

    simulation.output_writers[:averaged] = WriterType(model, model.velocities,
        schedule = AveragedTimeInterval(1.0, window=0.5),
        filename = "$(prefix)_averaged$(ext)",
        overwrite_existing = true
    )

    @test_nowarn run!(simulation)

    # Store reference state at iteration 8
    writer = simulation.output_writers[:averaged]
    wta_u = writer.outputs[output_key]

    original_result = copy(Array(wta_u.result))
    original_window_start_time = wta_u.window_start_time
    original_window_start_iteration = wta_u.window_start_iteration
    original_previous_collection_time = wta_u.previous_collection_time
    original_collecting = wta_u.schedule.collecting
    original_actuations = wta_u.schedule.actuations

    # Create new simulation and restore from checkpoint at iteration 8
    new_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    new_model = NonhydrostaticModel(; grid=new_grid)

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=15)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(8),
        prefix = prefix
    )

    new_simulation.output_writers[:averaged] = WriterType(new_model, new_model.velocities,
        schedule = AveragedTimeInterval(1.0, window=0.5),
        filename = "$(prefix)_averaged_restored$(ext)",
        overwrite_existing = true
    )

    # Restore from checkpoint at iteration 8
    @test_nowarn set!(new_simulation, 8)

    # Verify WindowedTimeAverage state was restored
    new_writer = new_simulation.output_writers[:averaged]
    new_wta_u = new_writer.outputs[output_key]

    @test new_wta_u.window_start_time == original_window_start_time
    @test new_wta_u.window_start_iteration == original_window_start_iteration
    @test new_wta_u.previous_collection_time == original_previous_collection_time
    @test new_wta_u.schedule.collecting == original_collecting
    @test new_wta_u.schedule.actuations == original_actuations

    # Most importantly: verify the accumulated result data was restored
    @test all(Array(new_wta_u.result) .≈ original_result)

    rm.(glob("$(prefix)*$(ext)"), force=true)

    return nothing
end

# This test verifies that checkpointing in the middle of a time-averaging window
# produces the exact same result as a continuous (uninterrupted) run.
#
# With AveragedTimeInterval(1.0, window=0.5) and Δt=0.1:
# - Output is written every 1.0 time units
# - Each output is the time-average over the preceding 0.5 time window
# - The first averaging window spans time 0.5 → 1.0 (iterations 5-10)
#
# Timeline:  0.0 -------- 0.5 ======== 0.7 ======== 1.0
#                          ↑            ↑            ↑
#                     window starts  CHECKPOINT   window ends
#                     collecting=true              (output written)
#
# Test strategy:
# 1. Run A (continuous): Runs from iteration 0 → 10 without interruption
# 2. Run B (checkpointed):
#    - Runs from iteration 0 → 7, checkpoints at iteration 7 (time 0.7)
#    - At this point, the WTA is actively collecting (collecting=true)
#    - The result field contains a partial average (accumulated from time 0.5 to 0.7)
# 3. Run B_new (restored): Creates a fresh simulation, restores from checkpoint,
#    continues to iteration 10
#
# At iteration 10, the test verifies:
# - Both runs completed exactly 1 averaging window (actuations == 1)
# - The final time-averaged results are identical (wta_A.result ≈ wta_B.result)
#
# If the WindowedTimeAverage state isn't properly checkpointed/restored, the restored
# simulation would lose the partial accumulated average and produce incorrect output.
function test_windowed_time_average_continuation_correctness(arch, WriterType)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.1

    if WriterType == JLD2Writer
        prefix_A = "wta_continuous_jld2_$(typeof(arch))"
        prefix_B = "wta_checkpoint_jld2_$(typeof(arch))"
        ext = ".jld2"
        output_key = :u
    elseif WriterType == NetCDFWriter
        prefix_A = "wta_continuous_netcdf_$(typeof(arch))"
        prefix_B = "wta_checkpoint_netcdf_$(typeof(arch))"
        ext = ".nc"
        output_key = "u"
    end

    # Run A: Continuous run from 0 to iteration 10
    grid_A = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_A = NonhydrostaticModel(; grid=grid_A)
    u_init(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    set!(model_A, u=u_init, v=0.1)

    simulation_A = Simulation(model_A, Δt=Δt, stop_iteration=10)

    simulation_A.output_writers[:averaged] = WriterType(model_A, model_A.velocities,
        schedule = AveragedTimeInterval(1.0, window=0.5),
        filename = "$(prefix_A)$(ext)",
        overwrite_existing = true
    )

    @test_nowarn run!(simulation_A)

    # Run B: From 0 to iteration 7, checkpoint in middle of first window
    grid_B = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B = NonhydrostaticModel(; grid=grid_B)
    set!(model_B, u=u_init, v=0.1)

    simulation_B = Simulation(model_B, Δt=Δt, stop_iteration=7)

    simulation_B.output_writers[:checkpointer] = Checkpointer(model_B,
        schedule = IterationInterval(7),
        prefix = prefix_B
    )
    simulation_B.output_writers[:averaged] = WriterType(model_B, model_B.velocities,
        schedule = AveragedTimeInterval(1.0, window=0.5),
        filename = "$(prefix_B)$(ext)",
        overwrite_existing = true
    )

    @test_nowarn run!(simulation_B)

    # Verify checkpoint was taken during active collection
    wta_B_at_checkpoint = simulation_B.output_writers[:averaged].outputs[output_key]
    @test wta_B_at_checkpoint.schedule.collecting == true

    # Run B_new: Restore from iteration 7, continue to iteration 10
    grid_B_new = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B_new = NonhydrostaticModel(; grid=grid_B_new)

    simulation_B_new = Simulation(model_B_new, Δt=Δt, stop_iteration=10)

    simulation_B_new.output_writers[:checkpointer] = Checkpointer(model_B_new,
        schedule = IterationInterval(7),
        prefix = prefix_B
    )
    simulation_B_new.output_writers[:averaged] = WriterType(model_B_new, model_B_new.velocities,
        schedule = AveragedTimeInterval(1.0, window=0.5),
        filename = "$(prefix_B)_restored$(ext)",
        overwrite_existing = true
    )

    @test_nowarn set!(simulation_B_new, true)
    @test_nowarn run!(simulation_B_new)

    # Compare at iteration 10 (time 1.0) - first window just completed
    wta_A = simulation_A.output_writers[:averaged].outputs[output_key]
    wta_B = simulation_B_new.output_writers[:averaged].outputs[output_key]

    # Both should have completed exactly 1 averaging window
    @test wta_A.schedule.actuations == 1
    @test wta_B.schedule.actuations == 1

    # The accumulated averages should be identical
    @test all(Array(wta_A.result) .≈ Array(wta_B.result))

    rm.(glob("$(prefix_A)*$(ext)"), force=true)
    rm.(glob("$(prefix_B)*$(ext)"), force=true)

    return nothing
end

function test_manual_checkpoint_with_checkpointer(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)
    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "manual_checkpoint_with_checkpointer_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(10),  # Won't trigger during this test
        prefix = prefix
    )

    @test_nowarn run!(simulation)
    @test_nowarn checkpoint(simulation)

    expected_filepath = "$(prefix)_iteration5.jld2"
    @test isfile(expected_filepath)

    # Verify we can restore from it
    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid)
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
        schedule = IterationInterval(10),
        prefix = prefix
    )

    @test_nowarn set!(new_simulation, expected_filepath)
    @test iteration(new_simulation) == 5

    test_model_equality(new_model, model)

    rm(expected_filepath, force=true)

    return nothing
end

function test_manual_checkpoint_without_checkpointer(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)
    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    # No Checkpointer configured
    @test_nowarn run!(simulation)

    # Manually checkpoint - should use default path
    @test_nowarn checkpoint(simulation)

    # Verify file was created with default naming
    expected_filepath = "checkpoint_iteration5.jld2"
    @test isfile(expected_filepath)

    # Verify we can restore from it
    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid)
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)

    @test_nowarn set!(new_simulation, expected_filepath)
    @test iteration(new_simulation) == 5

    test_model_equality(new_model, model)

    rm(expected_filepath, force=true)

    return nothing
end

function test_manual_checkpoint_with_filepath(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(; grid)
    set!(model, u=1, v=0.5)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    # Add a Checkpointer with a different prefix
    prefix = "should_not_use_this_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule = IterationInterval(10),
        prefix = prefix
    )

    @test_nowarn run!(simulation)

    # Manually checkpoint with explicit filepath
    custom_filepath = "custom_checkpoint_$(typeof(arch)).jld2"
    @test_nowarn checkpoint(simulation, filepath=custom_filepath)

    @test isfile(custom_filepath)
    @test !isfile("$(prefix)_iteration5.jld2")

    # Verify we can restore from it
    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(; grid=new_grid)
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)

    @test_nowarn set!(new_simulation, custom_filepath)
    @test iteration(new_simulation) == 5

    test_model_equality(new_model, model)

    rm(custom_filepath, force=true)

    return nothing
end

for arch in archs
    for pickup_method in (:boolean, :iteration, :filepath)
        @testset "Minimal restore [$(typeof(arch)), $(pickup_method)]" begin
            @info "  Testing minimal restore [$(typeof(arch)), $(pickup_method)]..."
            test_minimal_restore_nonhydrostatic(arch, Float64, pickup_method)
        end
    end

    @testset "Checkpointer cleanup [$(typeof(arch))]" begin
        @info "  Testing checkpointer cleanup [$(typeof(arch))]..."
        test_checkpointer_cleanup(arch)
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @testset "Thermal bubble checkpointing [$(typeof(arch)), $(timestepper)]" begin
            @info "  Testing thermal bubble checkpointing [$(typeof(arch)), $(timestepper)]..."
            test_thermal_bubble_checkpointing_nonhydrostatic(arch, timestepper)
        end
    end

    for pickup_method in (:boolean, :iteration, :filepath)
        @testset "Minimal restore hydrostatic [$(typeof(arch)), $(pickup_method)]" begin
            @info "  Testing minimal restore hydrostatic [$(typeof(arch)), $(pickup_method)]..."
            test_minimal_restore_hydrostatic(arch, Float64, pickup_method)
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "Thermal bubble checkpointing hydrostatic [$(typeof(arch)), $(timestepper)]" begin
            @info "  Testing thermal bubble checkpointing hydrostatic [$(typeof(arch)), $(timestepper)]..."
            test_thermal_bubble_checkpointing_hydrostatic(arch, timestepper)
        end
    end

    for pickup_method in (:boolean, :iteration, :filepath)
        @testset "Minimal restore shallow water [$(typeof(arch)), $(pickup_method)]" begin
            @info "  Testing minimal restore shallow water [$(typeof(arch)), $(pickup_method)]..."
            test_minimal_restore_shallow_water(arch, Float64, pickup_method)
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @testset "Height perturbation checkpointing shallow water [$(typeof(arch)), $(timestepper)]" begin
            @info "  Testing height perturbation checkpointing shallow water [$(typeof(arch)), $(timestepper)]..."
            test_height_perturbation_checkpointing_shallow_water(arch, timestepper)
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper]..."
            test_checkpointing_split_explicit_free_surface(arch, timestepper)
        end
    end

    for solver_method in (:PreconditionedConjugateGradient,)
        @testset "ImplicitFreeSurface checkpointing [$(typeof(arch)), $solver_method]" begin
            @info "  Testing ImplicitFreeSurface checkpointing [$(typeof(arch)), $solver_method]..."
            test_checkpointing_implicit_free_surface(arch, solver_method)
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @testset "Lagrangian particles checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing Lagrangian particles checkpointing [$(typeof(arch)), $timestepper]..."
            test_checkpointing_lagrangian_particles(arch, timestepper)
        end
    end

    for boundary_type in (:GridFittedBottom, :PartialCellBottom)
        @testset "ImmersedBoundaryGrid checkpointing [$(typeof(arch)), $boundary_type]" begin
            @info "  Testing ImmersedBoundaryGrid checkpointing [$(typeof(arch)), $boundary_type]..."
            test_checkpointing_immersed_boundary_grid(arch, boundary_type)
        end
    end

    @testset "LatitudeLongitudeGrid checkpointing [$(typeof(arch))]" begin
        @info "  Testing LatitudeLongitudeGrid checkpointing [$(typeof(arch))]..."
        test_checkpointing_latitude_longitude_grid(arch)
    end

    @testset "Float32 checkpointing [$(typeof(arch))]" begin
        @info "  Testing Float32 checkpointing [$(typeof(arch))]..."
        test_checkpointing_float32(arch)
    end

    @testset "Auxiliary fields checkpointing [$(typeof(arch))]" begin
        @info "  Testing auxiliary fields checkpointing [$(typeof(arch))]..."
        test_checkpointing_auxiliary_fields(arch)
    end

    @testset "Closure fields checkpointing [$(typeof(arch))]" begin
        @info "  Testing closure fields checkpointing [$(typeof(arch))]..."
        test_checkpointing_closure_fields(arch)
    end

    @testset "CATKE closure checkpointing [$(typeof(arch))]" begin
        @info "  Testing CATKE closure checkpointing [$(typeof(arch))]..."
        test_checkpointing_catke_closure(arch)
    end

    @testset "DynamicSmagorinsky closure checkpointing [$(typeof(arch))]" begin
        @info "  Testing DynamicSmagorinsky closure checkpointing [$(typeof(arch))]..."
        test_checkpointing_dynamic_smagorinsky_closure(arch)
    end

    @testset "RiBasedVerticalDiffusivity closure checkpointing [$(typeof(arch))]" begin
        @info "  Testing RiBasedVerticalDiffusivity closure checkpointing [$(typeof(arch))]..."
        test_checkpointing_ri_based_closure(arch)
    end

    @testset "TKEDissipationVerticalDiffusivity closure checkpointing [$(typeof(arch))]" begin
        @info "  Testing TKEDissipationVerticalDiffusivity closure checkpointing [$(typeof(arch))]..."
        test_checkpointing_tke_dissipation_closure(arch)
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @testset "Checkpoint continuation [$(typeof(arch)), $timestepper]" begin
            @info "  Testing checkpoint continuation consistency [$(typeof(arch)), $timestepper]..."
            test_checkpoint_continuation_matches_direct(arch, timestepper)
        end
    end

    for schedule_type in (:SpecifiedTimes, :ConsecutiveIterations, :TimeInterval, :WallTimeInterval)
        @testset "Stateful schedule checkpointing [$schedule_type] [$(typeof(arch))]" begin
            @info "  Testing stateful schedule checkpointing [$schedule_type] [$(typeof(arch))]..."
            test_stateful_schedule_checkpointing(arch, schedule_type)
        end
    end

    for WriterType in (JLD2Writer, NetCDFWriter)
        @testset "WindowedTimeAverage checkpointing [$WriterType] [$(typeof(arch))]" begin
            @info "  Testing WindowedTimeAverage checkpointing [$WriterType] [$(typeof(arch))]..."
            test_windowed_time_average_checkpointing(arch, WriterType)
        end
    end

    for WriterType in (JLD2Writer, NetCDFWriter)
        @testset "WindowedTimeAverage continuation correctness [$WriterType] [$(typeof(arch))]" begin
            @info "  Testing WindowedTimeAverage continuation correctness [$WriterType] [$(typeof(arch))]..."
            test_windowed_time_average_continuation_correctness(arch, WriterType)
        end
    end

    @testset "Edge cases [$(typeof(arch))]" begin
        @info "  Testing edge cases [$(typeof(arch))]..."
        test_checkpoint_empty_tracers(arch)
        test_checkpoint_missing_file_warning(arch)
    end

    @testset "Manual checkpointing [$(typeof(arch))]" begin
        @info "  Testing manual checkpointing [$(typeof(arch))]..."
        test_manual_checkpoint_with_checkpointer(arch)
        test_manual_checkpoint_without_checkpointer(arch)
        test_manual_checkpoint_with_filepath(arch)
    end
end
