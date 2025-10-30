include("dependencies_for_runtests.jl")

using Glob

using Oceananigans: restore_prognostic_state!, prognostic_fields
using Oceananigans.OutputWriters: load_checkpoint_state

function test_model_equality(test_model, true_model)
    @allowscalar begin
        test_model_fields = prognostic_fields(test_model)
        true_model_fields = prognostic_fields(true_model)
        field_names = keys(test_model_fields)

        for name in field_names
            @test all(test_model_fields[name].data .≈ true_model_fields[name].data)

            if name ∈ keys(test_model.timestepper.Gⁿ)
                @test all(test_model.timestepper.Gⁿ[name].data .≈ true_model.timestepper.Gⁿ[name].data)
                @test all(test_model.timestepper.G⁻[name].data .≈ true_model.timestepper.G⁻[name].data)
            end
        end
    end

    return nothing
end

function test_minimal_restore(arch, FT, pickup_method)
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

function test_thermal_bubble_checkpointing(arch, timestepper)
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

for arch in [CPU(), GPU()]
    for pickup_method in (:boolean, :iteration, :filepath)
        @testset "Minimal restore [$(typeof(arch)), $(pickup_method)]" begin
            @info "  Testing minimal restore [$(typeof(arch)), $(pickup_method)]..."
            test_minimal_restore(arch, Float64, pickup_method)
        end
    end

    @testset "Checkpointer cleanup [$(typeof(arch))]" begin
        @info "  Testing checkpointer cleanup [$(typeof(arch))]..."
        test_checkpointer_cleanup(arch)
    end

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        @testset "Thermal bubble checkpointing [$(typeof(arch)), $(timestepper)]" begin
            @info "  Testing thermal bubble checkpointing [$(typeof(arch)), $(timestepper)]..."
            test_thermal_bubble_checkpointing(arch, timestepper)
        end
    end
end
