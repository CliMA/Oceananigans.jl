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

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

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
    end

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
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


    for schedule_type in (:SpecifiedTimes, :ConsecutiveIterations, :TimeInterval)
        @testset "Stateful schedule checkpointing [$schedule_type] [$(typeof(arch))]" begin
            @info "  Testing stateful schedule checkpointing [$schedule_type] [$(typeof(arch))]..."
            test_stateful_schedule_checkpointing(arch, schedule_type)
        end
    end

    @testset "Edge cases [$(typeof(arch))]" begin
        @info "  Testing edge cases [$(typeof(arch))]..."
        test_checkpoint_empty_tracers(arch)
        test_checkpoint_missing_file_warning(arch)
    end
end
