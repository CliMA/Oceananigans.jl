include("dependencies_for_runtests.jl")

using Glob
using NCDatasets

using Oceananigans: restore_prognostic_state!, prognostic_fields
using Oceananigans.TurbulenceClosures.Smagorinskys: Smagorinsky,
    DirectionallyAveragedDynamicSmagorinsky, LagrangianAveragedDynamicSmagorinsky
using Oceananigans.Models.ShallowWaterModels: ShallowWaterScalarDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: ForwardBackwardScheme
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.OutputWriters: load_checkpoint_state

function test_model_equality(test_model, true_model; atol=0)
    # Test prognostic field equality
    test_model_fields = prognostic_fields(test_model)
    true_model_fields = prognostic_fields(true_model)
    field_names = keys(test_model_fields)

    for name in field_names
        @test all(isapprox.(test_model_fields[name].data, true_model_fields[name].data; atol))

        if name ∈ keys(test_model.timestepper.Gⁿ)
            @test all(isapprox.(test_model.timestepper.Gⁿ[name].data, true_model.timestepper.Gⁿ[name].data; atol))

            if hasfield(typeof(test_model.timestepper), :G⁻)
                @test all(isapprox.(test_model.timestepper.G⁻[name].data, true_model.timestepper.G⁻[name].data; atol))
            end

            if hasfield(typeof(test_model.timestepper), :Ψ⁻)
                @test all(isapprox.(test_model.timestepper.Ψ⁻[name].data, true_model.timestepper.Ψ⁻[name].data; atol))
            end
        end
    end

    # Test particle equality
    if hasproperty(test_model, :particles) && !isnothing(test_model.particles)
        for name in propertynames(test_model.particles.properties)
            test_prop = getproperty(test_model.particles.properties, name)
            true_prop = getproperty(true_model.particles.properties, name)
            @test all(isapprox.(Array(test_prop), Array(true_prop); atol))
        end
    end

    # Test free surface equality
    if hasproperty(test_model, :free_surface) && test_model.free_surface isa SplitExplicitFreeSurface
        fs_test = test_model.free_surface
        fs_true = true_model.free_surface
        @test all(isapprox.(interior(fs_test.barotropic_velocities.U), interior(fs_true.barotropic_velocities.U); atol))
        @test all(isapprox.(interior(fs_test.barotropic_velocities.V), interior(fs_true.barotropic_velocities.V); atol))
        @test all(isapprox.(interior(fs_test.filtered_state.η̅),        interior(fs_true.filtered_state.η̅); atol))
        @test all(isapprox.(interior(fs_test.filtered_state.U̅),        interior(fs_true.filtered_state.U̅); atol))
        @test all(isapprox.(interior(fs_test.filtered_state.V̅),        interior(fs_true.filtered_state.V̅); atol))
    end

    # Test auxiliary fields equality
    if hasproperty(test_model, :auxiliary_fields) && length(test_model.auxiliary_fields) > 0
        for name in keys(test_model.auxiliary_fields)
            @test all(isapprox.(interior(test_model.auxiliary_fields[name]), interior(true_model.auxiliary_fields[name]); atol))
        end
    end

    return nothing
end

function test_minimal_restore(arch, FT, pickup_method, model_type)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
                           size = (N, N, N),
                           topology = (Periodic, Bounded, Bounded),
                           extent = (L, L, L))

    if model_type == :nonhydrostatic
        model = NonhydrostaticModel(grid)
    elseif model_type == :hydrostatic
        model = HydrostaticFreeSurfaceModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    end

    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_$(model_type)_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(model;
                                schedule = TimeInterval(1.0),
                                prefix = prefix,
                                cleanup = false,
                                verbose = true)

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
                               extent = (L, L, L))

    if model_type == :nonhydrostatic
        new_model = NonhydrostaticModel(new_grid)
    elseif model_type == :hydrostatic
        new_model = HydrostaticFreeSurfaceModel(new_grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    end

    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(new_model;
                                    schedule = TimeInterval(1.0),
                                    prefix = prefix,
                                    cleanup = false,
                                    verbose = true)

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        @test_nowarn set!(new_simulation; checkpoint=:latest)
    elseif pickup_method == :iteration
        @test_nowarn set!(new_simulation; iteration=3)
    elseif pickup_method == :filepath
        @test_nowarn set!(new_simulation; checkpoint="$(prefix)_iteration3.jld2")
    end

    @test iteration(new_simulation) == 3
    @test time(new_simulation) == 3.0

    @test new_checkpointer.schedule.actuations == 3

    rm.(glob("$(prefix)_iteration*.jld2"))

    return nothing
end

function test_checkpointer_cleanup(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(grid; buoyancy=SeawaterBuoyancy(), tracers=(:T, :S))
    simulation = Simulation(model, Δt=0.2, stop_iteration=10)

    prefix = "checkpointer_cleanup_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                            schedule = IterationInterval(3),
                                                            prefix = prefix,
                                                            cleanup = true)

    @test_nowarn run!(simulation)

    [@test !isfile("$(prefix)_iteration$i.jld2") for i in 1:10 if i != 9]
    @test isfile("$(prefix)_iteration9.jld2")

    rm("$(prefix)_iteration9.jld2", force=true)

    return nothing
end

function test_thermal_bubble_checkpointing(arch, timestepper, model_type::Symbol)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        if model_type == :nonhydrostatic
            return NonhydrostaticModel(grid; timestepper,
                                       closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                       buoyancy = SeawaterBuoyancy(),
                                       tracers = (:T, :S))
        elseif model_type == :hydrostatic
            return HydrostaticFreeSurfaceModel(grid; timestepper,
                                               closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                               buoyancy = SeawaterBuoyancy(),
                                               tracers = (:T, :S))
        end
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=bubble, S=bubble)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint, then another 5 iterations
    model = make_model()
    set!(model, T=bubble, S=bubble)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "thermal_bubble_checkpointing_$(model_type)_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_minimal_restore_shallow_water(arch, FT, pickup_method)
    N = 16
    L = 50

    grid = RectilinearGrid(arch, FT,
                           size = (N, N),
                           topology = (Periodic, Periodic, Flat),
                           extent = (L, L))

    model = ShallowWaterModel(grid; gravitational_acceleration=1)
    set!(model, h=1)
    simulation = Simulation(model; Δt=1.0, stop_time=3.0)

    prefix = "mwe_checkpointer_shallow_water_$(typeof(arch))_$(FT)"

    checkpointer = Checkpointer(model;
                                schedule = TimeInterval(1.0),
                                prefix = prefix,
                                cleanup = false,
                                verbose = true)

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
                               extent = (L, L))

    new_model = ShallowWaterModel(new_grid; gravitational_acceleration=1)
    new_simulation = Simulation(new_model; Δt=1.0, stop_time=3.0)

    new_checkpointer = Checkpointer(new_model;
                                    schedule = TimeInterval(1.0),
                                    prefix = prefix,
                                    cleanup = false,
                                    verbose = true)

    new_simulation.output_writers[:checkpointer] = new_checkpointer

    if pickup_method == :boolean
        @test_nowarn set!(new_simulation; checkpoint=:latest)
    elseif pickup_method == :iteration
        @test_nowarn set!(new_simulation; iteration=3)
    elseif pickup_method == :filepath
        @test_nowarn set!(new_simulation; checkpoint="$(prefix)_iteration3.jld2")
    end

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

    perturbation(x, y) = 1 + 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2) / (Lx^2 + Ly^2))

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny), extent=(Lx, Ly), topology=(Periodic, Periodic, Flat))
        return ShallowWaterModel(grid; timestepper,
                                 gravitational_acceleration = 1,
                                 closure = ShallowWaterScalarDiffusivity(ν=4e-2, ξ=0))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, h=perturbation)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, h=perturbation)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "height_perturbation_checkpointing_shallow_water_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_split_explicit_free_surface(arch, timestepper, free_surface_timestepper)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1000, 1000, 100
    Δt = 0.1

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        free_surface = SplitExplicitFreeSurface(grid; substeps=30, timestepper=free_surface_timestepper)
        return HydrostaticFreeSurfaceModel(grid; timestepper, free_surface,
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=bubble, S=bubble)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=bubble, S=bubble)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    fs_ts_name = nameof(typeof(free_surface_timestepper))
    prefix = "split_explicit_checkpointing_$(typeof(arch))_$(timestepper)_$(fs_ts_name)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_zstar_coordinate(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1000, 1000, 100
    Δt = 0.1

    # Perturbations that drive free surface motion (exercises z-star dynamics)
    T_init(x, y, z) = 20 + 0.01 * z + 0.1 * exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (Lx/4)^2)
    u_init(x, y, z) = 0.1 * sin(2π * x / Lx)

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz),
                               x = (0, Lx), y = (0, Ly),
                               z = MutableVerticalDiscretization((-Lz, 0)))
        free_surface = SplitExplicitFreeSurface(grid; substeps=30)
        return HydrostaticFreeSurfaceModel(grid; timestepper, free_surface,
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=T_init, S=35, u=u_init)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=T_init, S=35, u=u_init)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "zstar_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    # Verify z-star is being exercised (non-zero free surface)
    @test maximum(abs, parent(ref_model.grid.z.ηⁿ)) > 0

    # Check grid's mutable vertical discretization fields
    ref_z = ref_model.grid.z
    new_z = new_model.grid.z
    @test all(parent(ref_z.ηⁿ) .≈ parent(new_z.ηⁿ))
    @test all(parent(ref_z.σᶜᶜⁿ) .≈ parent(new_z.σᶜᶜⁿ))
    @test all(parent(ref_z.σᶜᶜ⁻) .≈ parent(new_z.σᶜᶜ⁻))

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_implicit_free_surface(arch, solver_method)
    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 1000, 1000, 1000
    Δt = 0.1

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz),
                               topology=(Bounded, Bounded, Bounded))
        free_surface = ImplicitFreeSurface(solver_method=solver_method)
        return HydrostaticFreeSurfaceModel(grid; free_surface,
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=bubble, S=bubble)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=bubble, S=bubble)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "implicit_free_surface_checkpointing_$(typeof(arch))_$(solver_method)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_lagrangian_particles(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.01
    P = 10  # number of particles

    function make_model()
        grid = RectilinearGrid(arch,
                               size = (Nx, Ny, Nz),
                               extent = (Lx, Ly, Lz),
                               topology = (Periodic, Periodic, Bounded))
        xs = on_architecture(arch, 0.5 * ones(P))
        ys = on_architecture(arch, 0.5 * ones(P))
        zs = on_architecture(arch, -0.5 * ones(P))
        particles = LagrangianParticles(x=xs, y=ys, z=zs)
        return NonhydrostaticModel(grid; timestepper, particles,
                                   closure = ScalarDiffusivity(ν=1e-4, κ=1e-4))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5, w=0)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, u=1, v=0.5, w=0)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "lagrangian_particles_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_immersed_boundary_grid(arch, boundary_type)
    Nx, Ny, Nz = 16, 16, 8
    Lx, Ly, Lz = 100, 100, 50
    Δt = 0.1

    bottom(x, y) = -40 + 10 * sin(2π * x / Lx)
    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z + Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))

    function make_model()
        underlying_grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        if boundary_type == :GridFittedBottom
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
        elseif boundary_type == :PartialCellBottom
            grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))
        end
        return NonhydrostaticModel(grid;
                                   closure = ScalarDiffusivity(ν=4e-2, κ=4e-2),
                                   buoyancy = SeawaterBuoyancy(),
                                   tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=bubble, S=bubble)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=bubble, S=bubble)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "immersed_boundary_checkpointing_$(typeof(arch))_$(boundary_type)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_latitude_longitude_grid(arch)
    # Use parameters that ensure numerical stability
    Nx, Ny, Nz = 8, 8, 4
    Δt = 300  # 5 minute timestep

    T_init(λ, φ, z) = 20 + 5 * (z + 1000) / 1000

    function make_model()
        grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz),
                                     longitude=(0, 60),
                                     latitude=(-30, 30),
                                     z=(-1000, 0))
        free_surface = SplitExplicitFreeSurface(grid; substeps=30)
        return HydrostaticFreeSurfaceModel(grid; free_surface,
                                           coriolis = HydrostaticSphericalCoriolis(),
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=T_init, S=35)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=T_init, S=35)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "lat_lon_grid_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_float32(arch)
    N = 8
    L = 1
    Δt = 0.1

    function make_model()
        grid = RectilinearGrid(arch, Float32, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid)
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Float32(Δt), stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Float32(Δt), stop_iteration=5)

    prefix = "float32_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Float32(Δt), stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_auxiliary_fields(arch)
    N = 8
    L = 1
    Δt = 0.1

    custom_field_init(x, y, z) = x + y + z

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        auxiliary_fields = (custom_field = CenterField(grid),)
        return NonhydrostaticModel(grid; auxiliary_fields)
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model.auxiliary_fields.custom_field, custom_field_init)
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model.auxiliary_fields.custom_field, custom_field_init)
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "auxiliary_fields_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_closure_fields(arch)
    N = 8
    L = 1
    Δt = 0.01

    u₀(x, y, z) = sin(2π*x)
    v₀(x, y, z) = cos(2π*y)
    T₀(x, y, z) = 20
    S₀(x, y, z) = 35

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid;
                                   closure = SmagorinskyLilly(),
                                   buoyancy = SeawaterBuoyancy(),
                                   tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=u₀, v=v₀, T=T₀, S=S₀)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, u=u₀, v=v₀, T=T₀, S=S₀)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "closure_fields_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_smagorinsky_closure(arch, timestepper, closure, closure_name)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.1

    u_init(x, y, z) = sin(2π * z / Lz)

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        return NonhydrostaticModel(grid; timestepper, closure,
                                   buoyancy = SeawaterBuoyancy(),
                                   tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=u_init)
    set!(ref_model, T=20, S=35)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, u=u_init)
    set!(model, T=20, S=35)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "$(closure_name)_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Verify closure-specific fields
    new_cf = new_model.closure_fields
    ref_cf = ref_model.closure_fields

    if closure isa DirectionallyAveragedDynamicSmagorinsky
        @test all(Array(interior(new_cf.𝒥ᴸᴹ)) .≈ Array(interior(ref_cf.𝒥ᴸᴹ)))
        @test all(Array(interior(new_cf.𝒥ᴹᴹ)) .≈ Array(interior(ref_cf.𝒥ᴹᴹ)))
    elseif closure isa LagrangianAveragedDynamicSmagorinsky
        @test new_cf.previous_compute_time[] ≈ ref_cf.previous_compute_time[]
        @test all(Array(interior(new_cf.𝒥ᴸᴹ)) .≈ Array(interior(ref_cf.𝒥ᴸᴹ)))
        @test all(Array(interior(new_cf.𝒥ᴹᴹ)) .≈ Array(interior(ref_cf.𝒥ᴹᴹ)))
        @test all(Array(interior(new_cf.𝒥ᴸᴹ⁻)) .≈ Array(interior(ref_cf.𝒥ᴸᴹ⁻)))
        @test all(Array(interior(new_cf.𝒥ᴹᴹ⁻)) .≈ Array(interior(ref_cf.𝒥ᴹᴹ⁻)))
    end

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model; atol=1e-20)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_ri_based_closure(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 60

    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.1 * z / Lz

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        return HydrostaticFreeSurfaceModel(grid; timestepper,
                                           closure = RiBasedVerticalDiffusivity(Cᵃᵛ=0.6),
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=T_init, S=35, u=u_init)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=T_init, S=35, u=u_init)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "ri_based_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Verify closure field state matches reference at iteration 10
    @test all(Array(interior(new_model.closure_fields.κc)) .≈ Array(interior(ref_model.closure_fields.κc)))
    @test all(Array(interior(new_model.closure_fields.κu)) .≈ Array(interior(ref_model.closure_fields.κu)))

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_catke_closure(arch, timestepper, closure=CATKEVerticalDiffusivity())
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 100, 100, 100
    Δt = 0.1

    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.01 * sin(2π * x / Lx + 3π * y / Ly)

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        return HydrostaticFreeSurfaceModel(grid; timestepper,
                                           closure,
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=T_init, S=35, u=u_init)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=T_init, S=35, u=u_init)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    closure_prefix = closure isa CATKEVerticalDiffusivity ? "catke" :
                     closure isa NTuple{1} && closure[1] isa CATKEVerticalDiffusivity ? "catke" :
                     (closure isa Tuple && any(x -> x isa CATKEVerticalDiffusivity, closure)) ? "catke_etal" :
                     "some_closure"

    prefix = closure_prefix * "_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    # We need a small atol because while all prognostic fields are exactly equal, the e
    # field can have tiny differences due to floating-point ordering with RK3 I think.
    atol = timestepper == :SplitRungeKutta3 ? 1e-20 : 0
    test_model_equality(new_model, ref_model; atol)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpointing_tke_dissipation_closure(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 100, 100, 100
    Δt = 60

    T_init(x, y, z) = 20 + 0.01 * z
    u_init(x, y, z) = 0.01 * sin(2π * x / Lx + 3π * y / Ly)

    function make_model()
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
        return HydrostaticFreeSurfaceModel(grid; timestepper,
                                           closure = TKEDissipationVerticalDiffusivity(),
                                           buoyancy = SeawaterBuoyancy(),
                                           tracers = (:T, :S))
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, T=T_init, S=35, u=u_init)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)

    if timestepper == :SplitRungeKutta3
        # See: https://github.com/CliMA/Oceananigans.jl/issues/5127
        @test_broken run!(ref_simulation) |> isnothing
        return nothing
    end

    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, T=T_init, S=35, u=u_init)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "tke_dissipation_checkpointing_$(typeof(arch))_$(timestepper)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Verify previous_velocities state matches reference at iteration 10
    ref_pv = ref_model.closure_fields.previous_velocities
    new_pv = new_model.closure_fields.previous_velocities
    @test all(Array(interior(new_pv.u)) .≈ Array(interior(ref_pv.u)))
    @test all(Array(interior(new_pv.v)) .≈ Array(interior(ref_pv.v)))

    # Compare final states at iteration 10
    # We need a small atol because while all prognostic fields are exactly equal, the e and ϵ
    # fields can have tiny differences due to floating-point ordering with RK3 I think.
    atol = timestepper == :SplitRungeKutta3 ? 1e-16 : 0
    test_model_equality(new_model, ref_model; atol)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_continuation_matches_direct(arch, timestepper)
    Nx, Ny, Nz = 8, 8, 8
    Lx, Ly, Lz = 1, 1, 1
    Δt = 0.01

    # Run A: Direct run for 10 iterations
    grid_A = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_A = NonhydrostaticModel(grid_A; timestepper,
                                  closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    bubble(x, y, z) = 0.01 * exp(-100 * ((x - Lx/2)^2 + (y - Ly/2)^2 + (z - Lz/2)^2) / (Lx^2 + Ly^2 + Lz^2))
    set!(model_A, T=bubble, S=bubble, u=0.1)

    simulation_A = Simulation(model_A, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(simulation_A)

    # Run B: Run 5 iterations, checkpoint, restore, run 5 more
    grid_B = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B = NonhydrostaticModel(grid_B; timestepper,
                                  closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                  buoyancy = SeawaterBuoyancy(),
                                  tracers = (:T, :S))

    set!(model_B, T=bubble, S=bubble, u=0.1)

    simulation_B = Simulation(model_B, Δt=Δt, stop_iteration=5)

    prefix = "continuation_test_$(typeof(arch))_$(timestepper)"
    simulation_B.output_writers[:checkpointer] = Checkpointer(model_B,
                                                              schedule = IterationInterval(5),
                                                              prefix = prefix)

    @test_nowarn run!(simulation_B)

    # Create fresh model and restore from checkpoint
    grid_B_new = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B_new = NonhydrostaticModel(grid_B_new; timestepper,
                                      closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
                                      buoyancy = SeawaterBuoyancy(),
                                      tracers = (:T, :S))

    simulation_B_new = Simulation(model_B_new, Δt=Δt, stop_iteration=10)

    simulation_B_new.output_writers[:checkpointer] = Checkpointer(model_B_new,
                                                                  schedule = IterationInterval(5),
                                                                  prefix = prefix)

    @test_nowarn set!(simulation_B_new; checkpoint=:latest)

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

function test_stateful_schedule_checkpointing(arch, schedule_type)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(grid)
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
                                                            prefix = prefix)

    # We will test the schedule via a callback that does nothing.
    simulation.callbacks[:test_schedule] = Callback(_ -> nothing, schedule)

    @test_nowarn run!(simulation)

    original_schedule = simulation.callbacks[:test_schedule].schedule

    new_grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    new_model = NonhydrostaticModel(new_grid)

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=15)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(10),
                                                                prefix = prefix)

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

    @test_nowarn set!(new_simulation; checkpoint=:latest)

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
    model = NonhydrostaticModel(grid)

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
                                                            prefix = prefix)

    simulation.output_writers[:averaged] = WriterType(model, model.velocities,
                                                      schedule = AveragedTimeInterval(1.0, window=0.5),
                                                      filename = "$(prefix)_averaged$(ext)",
                                                      overwrite_existing = true)

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
    new_model = NonhydrostaticModel(new_grid)

    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=15)

    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(8),
                                                                prefix = prefix)

    new_simulation.output_writers[:averaged] = WriterType(new_model, new_model.velocities,
                                                          schedule = AveragedTimeInterval(1.0, window=0.5),
                                                          filename = "$(prefix)_averaged_restored$(ext)",
                                                          overwrite_existing = true)

    # Restore from checkpoint at iteration 8
    @test_nowarn set!(new_simulation; iteration=8)

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
    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

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
    model_A = NonhydrostaticModel(grid_A)
    u_init(x, y, z) = sin(2π * x / Lx) * cos(2π * y / Ly)
    set!(model_A, u=u_init, v=0.1)

    simulation_A = Simulation(model_A, Δt=Δt, stop_iteration=10)

    simulation_A.output_writers[:averaged] = WriterType(model_A, model_A.velocities,
                                                       schedule = AveragedTimeInterval(1.0, window=0.5),
                                                       filename = "$(prefix_A)$(ext)",
                                                       overwrite_existing = true)

    @test_nowarn run!(simulation_A)

    # Run B: From 0 to iteration 7, checkpoint in middle of first window
    grid_B = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B = NonhydrostaticModel(grid_B)
    set!(model_B, u=u_init, v=0.1)

    simulation_B = Simulation(model_B, Δt=Δt, stop_iteration=7)

    simulation_B.output_writers[:checkpointer] = Checkpointer(model_B,
                                                              schedule = IterationInterval(7),
                                                              prefix = prefix_B)
    simulation_B.output_writers[:averaged] = WriterType(model_B, model_B.velocities,
                                                        schedule = AveragedTimeInterval(1.0, window=0.5),
                                                        filename = "$(prefix_B)$(ext)",
                                                        overwrite_existing = true)

    @test_nowarn run!(simulation_B)

    # Verify checkpoint was taken during active collection
    wta_B_at_checkpoint = simulation_B.output_writers[:averaged].outputs[output_key]
    @test wta_B_at_checkpoint.schedule.collecting == true

    # Run B_new: Restore from iteration 7, continue to iteration 10
    grid_B_new = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    model_B_new = NonhydrostaticModel(grid_B_new)

    simulation_B_new = Simulation(model_B_new, Δt=Δt, stop_iteration=10)

    simulation_B_new.output_writers[:checkpointer] = Checkpointer(model_B_new,
                                                                  schedule = IterationInterval(7),
                                                                  prefix = prefix_B)
    simulation_B_new.output_writers[:averaged] = WriterType(model_B_new, model_B_new.velocities,
                                                            schedule = AveragedTimeInterval(1.0, window=0.5),
                                                            filename = "$(prefix_B)_restored$(ext)",
                                                            overwrite_existing = true)

    @test_nowarn set!(simulation_B_new; checkpoint=:latest)
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
    rm.(glob("$(prefix_B)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_empty_tracers(arch)
    N = 8
    L = 1
    Δt = 0.1

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid; tracers=())
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, checkpoint
    model = make_model()
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "empty_tracers_checkpointing_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(5),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=:latest)
    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_missing_file_warning(arch)
    N = 8
    L = 1
    Δt = 0.1

    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(grid)

    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    # Use a unique prefix that doesn't have any checkpoint files
    prefix = "nonexistent_checkpoint_$(typeof(arch))_$(rand(1:100000))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(5),
                                                            prefix = prefix)

    # Should warn but not error when no checkpoint files exist
    @test_logs (:warn,) set!(simulation; checkpoint=:latest)

    # Simulation should still be at iteration 0
    @test iteration(simulation) == 0

    return nothing
end

function test_manual_checkpoint_with_checkpointer(arch)
    N = 8
    L = 1
    Δt = 0.1

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid)
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, manual checkpoint
    model = make_model()
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    prefix = "manual_checkpoint_with_checkpointer_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(10),  # Won't trigger during this test
                                                            prefix = prefix)

    @test_nowarn run!(simulation)
    @test_nowarn checkpoint(simulation)

    expected_filepath = "$(prefix)_iteration5.jld2"
    @test isfile(expected_filepath)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)
    new_simulation.output_writers[:checkpointer] = Checkpointer(new_model,
                                                                schedule = IterationInterval(10),
                                                                prefix = prefix)

    @test_nowarn set!(new_simulation; checkpoint=expected_filepath)
    @test iteration(new_simulation) == 5

    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_manual_checkpoint_without_checkpointer(arch)
    N = 8
    L = 1
    Δt = 0.1

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid)
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, manual checkpoint (no Checkpointer configured)
    model = make_model()
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    @test_nowarn run!(simulation)

    # Manually checkpoint - should use default path
    @test_nowarn checkpoint(simulation)

    # Verify file was created with default naming
    expected_filepath = "checkpoint_iteration5.jld2"
    @test isfile(expected_filepath)

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)

    @test_nowarn set!(new_simulation; checkpoint=expected_filepath)
    @test iteration(new_simulation) == 5

    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm(expected_filepath, force=true)

    return nothing
end

function test_manual_checkpoint_with_filepath(arch)
    N = 8
    L = 1
    Δt = 0.1

    function make_model()
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
        return NonhydrostaticModel(grid)
    end

    # Reference run: 10 iterations continuously
    ref_model = make_model()
    set!(ref_model, u=1, v=0.5)
    ref_simulation = Simulation(ref_model, Δt=Δt, stop_iteration=10)
    @test_nowarn run!(ref_simulation)

    # Checkpointed run: 5 iterations, manual checkpoint with custom filepath
    model = make_model()
    set!(model, u=1, v=0.5)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    # Add a Checkpointer with a different prefix
    prefix = "should_not_use_this_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                            schedule = IterationInterval(10),
                                                            prefix = prefix)

    @test_nowarn run!(simulation)

    # Manually checkpoint with explicit filepath
    custom_filepath = "custom_checkpoint_$(typeof(arch)).jld2"
    @test_nowarn checkpoint(simulation, filepath=custom_filepath)

    @test isfile(custom_filepath)
    @test !isfile("$(prefix)_iteration5.jld2")

    # Restore and continue for 5 more iterations
    new_model = make_model()
    new_simulation = Simulation(new_model, Δt=Δt, stop_iteration=10)

    @test_nowarn set!(new_simulation; checkpoint=custom_filepath)
    @test iteration(new_simulation) == 5

    @test_nowarn run!(new_simulation)

    # Compare final states at iteration 10
    test_model_equality(new_model, ref_model)

    rm(custom_filepath, force=true)
    rm.(glob("$(prefix)_iteration*.jld2"), force=true)

    return nothing
end

function test_checkpoint_at_end(arch)
    N = 8
    L = 1
    Δt = 0.1
    expected_filepath = "checkpoint_iteration5.jld2"

    # Test with checkpoint_at_end=false (default) - should NOT create checkpoint
    grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model = NonhydrostaticModel(grid)
    simulation = Simulation(model, Δt=Δt, stop_iteration=5)

    @test_nowarn run!(simulation)

    @test !isfile(expected_filepath)

    # Test with checkpoint_at_end=true - should create checkpoint
    grid2 = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L))
    model2 = NonhydrostaticModel(grid2)
    set!(model2, u=1, v=0.5)
    simulation2 = Simulation(model2, Δt=Δt, stop_iteration=5)

    @test_nowarn run!(simulation2, checkpoint_at_end=true)  # Should create checkpoint

    @test isfile(expected_filepath)
    rm(expected_filepath, force=true)

    return nothing
end

"""
Test checkpointing for models with OpenBoundaryCondition using the specified scheme.
Verifies that every element of model.boundary_mass_fluxes is correctly saved and restored.
"""
function test_open_boundary_condition_scheme_checkpointing(arch, timestepper, scheme)
    Nx, Ny, Nz = 4, 4, 4
    Δt = 0.5

    function make_model()
        grid = RectilinearGrid(arch, topology=(Bounded, Bounded, Bounded), size=(Nx, Ny, Nz), extent=(10, 10, 10))
        obc = OpenBoundaryCondition(0.1, scheme=scheme)
        u_bcs = FieldBoundaryConditions(west=obc, east=obc)
        return NonhydrostaticModel(grid; timestepper, boundary_conditions=(u=u_bcs,), tracers=:c)
    end

    # Run simulation and checkpoint
    model = make_model()
    set!(model, c=1)
    simulation = Simulation(model, Δt=Δt, stop_iteration=3)

    scheme_name = replace(string(typeof(scheme)), "." => "_")
    prefix = "obc_$(scheme_name)_checkpoint_$(timestepper)_$(typeof(arch))"
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(3), prefix=prefix)
    @test_nowarn run!(simulation)

    # Store original boundary mass fluxes
    original_bmf = model.boundary_mass_fluxes

    # Restore from checkpoint and verify boundary_mass_fluxes match exactly
    restored_model = make_model()
    restored_simulation = Simulation(restored_model, Δt=Δt, stop_iteration=3)
    @test_nowarn set!(restored_simulation; checkpoint=:latest)

    restored_bmf = restored_model.boundary_mass_fluxes

    # Test that structure is identical
    @test propertynames(original_bmf) == propertynames(restored_bmf)

    # Test that every element matches exactly
    for field_name in propertynames(original_bmf)
        original_field = getproperty(original_bmf, field_name)
        restored_field = getproperty(restored_bmf, field_name)

        if original_field isa Field
            @test all(interior(original_field) .== interior(restored_field))
            @test size(original_field) == size(restored_field)
        else
            @test original_field == restored_field
        end
    end

    # Clean up
    rm.(glob(prefix * "*"), force=true)
    return nothing
end

for arch in archs
    for model_type in (:nonhydrostatic, :hydrostatic)
        for pickup_method in (:boolean, :iteration, :filepath)
            @testset "Minimal restore [$model_type, $pickup_method] [$(typeof(arch))]" begin
                @info "  Testing minimal restore [$model_type, $pickup_method] [$(typeof(arch))]..."
                test_minimal_restore(arch, Float64, pickup_method, model_type)
            end
        end
    end

    @testset "Checkpointer cleanup [$(typeof(arch))]" begin
        @info "  Testing checkpointer cleanup [$(typeof(arch))]..."
        test_checkpointer_cleanup(arch)
    end

    for model_type in (:nonhydrostatic, :hydrostatic)
        timesteppers = model_type == :hydrostatic ?
            (:QuasiAdamsBashforth2, :SplitRungeKutta3) :
            (:QuasiAdamsBashforth2, :RungeKutta3)

        for timestepper in timesteppers
            @testset "Thermal bubble checkpointing [$model_type, $timestepper] [$(typeof(arch))]" begin
                @info "  Testing thermal bubble checkpointing [$model_type, $timestepper] [$(typeof(arch))]..."
                test_thermal_bubble_checkpointing(arch, timestepper, model_type)
            end
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
        free_surface_timestepper = ForwardBackwardScheme()
        fs_ts_name = nameof(typeof(free_surface_timestepper))
        @testset "SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper, $fs_ts_name]" begin
            @info "  Testing SplitExplicitFreeSurface checkpointing [$(typeof(arch)), $timestepper, $ForwardBackwardScheme]..."
            test_checkpointing_split_explicit_free_surface(arch, timestepper, free_surface_timestepper)
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "ZStarCoordinate checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing ZStarCoordinate checkpointing [$(typeof(arch)), $timestepper]..."
            test_checkpointing_zstar_coordinate(arch, timestepper)
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

    smagorinsky_closures = [
        (Smagorinsky(coefficient=0.16), "Smagorinsky"),
        (SmagorinskyLilly(), "SmagorinskyLilly"),
        (DynamicSmagorinsky(averaging=(1, 2)), "DirectionallyAveragedDynamicSmagorinsky"),
        (DynamicSmagorinsky(), "LagrangianAveragedDynamicSmagorinsky"),
    ]

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        for (closure, name) in smagorinsky_closures
            @testset "$name closure checkpointing [$(typeof(arch)), $timestepper]" begin
                @info "  Testing $name closure checkpointing [$(typeof(arch)), $timestepper]..."
                test_checkpointing_smagorinsky_closure(arch, timestepper, closure, name)
            end
        end
    end

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "RiBasedVerticalDiffusivity closure checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing RiBasedVerticalDiffusivity closure checkpointing [$(typeof(arch)), $timestepper]..."
            test_checkpointing_ri_based_closure(arch, timestepper)
        end

        if timestepper == :SplitRungeKutta3 # currently, CATKE and TKE-ε tests fail with :QuasiAdamsBashforth2
            @testset "CATKE closure checkpointing [$(typeof(arch)), $timestepper]" begin
                @info "  Testing CATKE closure checkpointing [$(typeof(arch)), $timestepper]..."
                test_checkpointing_catke_closure(arch, timestepper, CATKEVerticalDiffusivity())
                test_checkpointing_catke_closure(arch, timestepper, (CATKEVerticalDiffusivity(),))
                @info "  Testing CATKE+another closure checkpointing [$(typeof(arch)), $timestepper]..."
                test_checkpointing_catke_closure(arch, timestepper, (CATKEVerticalDiffusivity(), VerticalScalarDiffusivity(κ=1e-5)))
            end

            @testset "TKEDissipationVerticalDiffusivity closure checkpointing [$(typeof(arch)), $timestepper]" begin
                @info "  Testing TKEDissipationVerticalDiffusivity closure checkpointing [$(typeof(arch)), $timestepper]..."
                test_checkpointing_tke_dissipation_closure(arch, timestepper)
            end
        end
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

    schemes = [
        PerturbationAdvection(inflow_timescale=2, outflow_timescale=1),
    ]

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3), scheme in schemes
        scheme_name = replace(string(typeof(scheme)), "." => "_")
        @testset "OpenBoundaryCondition with $scheme_name checkpointing [$(typeof(arch)), $timestepper]" begin
            @info "  Testing OpenBoundaryCondition with $scheme_name checkpointing [$(typeof(arch)), $timestepper]..."
            test_open_boundary_condition_scheme_checkpointing(arch, timestepper, scheme)
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
        test_checkpoint_at_end(arch)
    end
end
