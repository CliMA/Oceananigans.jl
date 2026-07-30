include("dependencies_for_runtests.jl")

using NCDatasets
using Zarr
using Oceananigans.OutputWriters: TimeDerivative, seed_time_derivative!, update_time_derivative!

#####
##### A tracer relaxed at rate λ obeys ∂ₜc = -λ c exactly, which lets the backward
##### difference be checked against an analytical solution.
#####

function relaxing_tracer_model(arch, λ)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))
    forcing = (; c = Relaxation(rate=λ))
    model = NonhydrostaticModel(grid; tracers=:c, forcing)
    set!(model, c=1)
    return model
end

time_derivative_outputs(model) = (; ∂ₜc = TimeDerivative(model.tracers.c))
named_time_derivative_outputs(model) = Dict("dcdt" => TimeDerivative(model.tracers.c))

function test_time_derivative_of_field(arch)
    λ, Δt = 2, 1e-3
    model = relaxing_tracer_model(arch, λ)
    ∂ₜc = TimeDerivativeCallback(model.tracers.c)

    @test ∂ₜc isa TimeDerivativeCallback

    simulation = Simulation(model; Δt, stop_iteration=4)
    simulation.callbacks[:∂ₜc] = ∂ₜc
    run!(simulation)

    derivative = ∂ₜc.func

    # The backward difference is centered at t - Δt/2, where c is larger by exp(λ Δt / 2)
    c = Array(interior(model.tracers.c))
    expected = @. -λ * c * exp(λ * Δt / 2)

    @test all(isapprox.(Array(interior(derivative)), expected, rtol=1e-4))

    # `result` is a Field, so it can be copied into a Field of the caller's choosing
    copied = CenterField(model.grid)
    set!(copied, derivative.result)
    @test all(Array(interior(copied)) .≈ Array(interior(derivative)))

    return nothing
end

function test_time_derivative_operators(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    c = CenterField(grid)
    u = CenterField(grid)
    set!(u, 7)

    ∂ₜc = TimeDerivative(c)
    parent(∂ₜc.result) .= 5

    # Every operator form must build the same operation as the one written on `result`
    @test (2 * ∂ₜc)[1, 1, 1]      == (2 * ∂ₜc.result)[1, 1, 1]
    @test (∂ₜc * 2)[1, 1, 1]      == (∂ₜc.result * 2)[1, 1, 1]
    @test (∂ₜc / 2)[1, 1, 1]      == (∂ₜc.result / 2)[1, 1, 1]
    @test (2 - ∂ₜc)[1, 1, 1]      == (2 - ∂ₜc.result)[1, 1, 1]
    @test (∂ₜc^2)[1, 1, 1]        == (∂ₜc.result^2)[1, 1, 1]
    @test (∂ₜc > 0)[1, 1, 1]      == (∂ₜc.result > 0)[1, 1, 1]
    @test abs(∂ₜc)[1, 1, 1]       == abs(∂ₜc.result)[1, 1, 1]
    @test (-∂ₜc)[1, 1, 1]         == (-∂ₜc.result)[1, 1, 1]
    @test sqrt(abs(∂ₜc))[1, 1, 1] == sqrt(abs(∂ₜc.result))[1, 1, 1]
    @test (∂ₜc * u)[1, 1, 1]      == (∂ₜc.result * u)[1, 1, 1]
    @test (u * ∂ₜc)[1, 1, 1]      == (u * ∂ₜc.result)[1, 1, 1]
    @test (∂ₜc + ∂ₜc)[1, 1, 1]    == (∂ₜc.result + ∂ₜc.result)[1, 1, 1]
    @test (2 * ∂ₜc * u)[1, 1, 1]  == (2 * ∂ₜc.result * u)[1, 1, 1]

    # And must compute through a Field like any other operation
    product = Field(2 * ∂ₜc * u)
    compute!(product)
    @test Array(interior(product))[1, 1, 1] == 70

    # Reductions read the derivative directly
    @test maximum(abs, ∂ₜc) == 5
    @test size(∂ₜc) == size(∂ₜc.result)

    return nothing
end

function test_time_derivative_seeding(arch)
    λ = 2
    model = relaxing_tracer_model(arch, λ)

    # Constructing with a model seeds the operand and the time immediately
    seeded = TimeDerivative(model.tracers.c, model)
    @test seeded.previous_time == model.clock.time
    @test all(Array(interior(seeded.previous)) .≈ Array(interior(model.tracers.c)))

    # Constructing without one defers seeding
    ∂ₜc = TimeDerivative(model.tracers.c)
    @test all(Array(interior(∂ₜc.previous)) .== 0)

    seed_time_derivative!(∂ₜc, model)
    @test ∂ₜc.previous_time == model.clock.time
    @test all(Array(interior(∂ₜc)) .== 0)

    # A backward difference needs two evaluations, so nothing happens at the seeded time
    update_time_derivative!(∂ₜc, model)
    @test all(Array(interior(∂ₜc)) .== 0)

    return nothing
end

function test_time_derivative_of_reduction(arch)
    λ, Δt = 2, 1e-3
    model = relaxing_tracer_model(arch, λ)
    c = model.tracers.c
    ∂ₜ∫c² = TimeDerivativeCallback(Integral(c^2))

    simulation = Simulation(model; Δt, stop_iteration=4)
    simulation.callbacks[:∂ₜ∫c²] = ∂ₜ∫c²
    run!(simulation)

    ∫c² = Field(Integral(c^2))
    compute!(∫c²)

    # ∫c² decays at twice the rate of c
    expected = -2λ * Array(interior(∫c²))[1, 1, 1] * exp(λ * Δt)

    @test Array(interior(∂ₜ∫c².func))[1, 1, 1] ≈ expected rtol=1e-4

    return nothing
end

function test_time_derivative_schedule(arch)
    λ, Δt = 2, 1e-3
    model = relaxing_tracer_model(arch, λ)
    ∂ₜc = TimeDerivativeCallback(model.tracers.c, schedule=IterationInterval(2))

    simulation = Simulation(model; Δt, stop_iteration=4)
    simulation.callbacks[:∂ₜc] = ∂ₜc
    run!(simulation)

    derivative = ∂ₜc.func
    @test derivative.previous_time == model.clock.time

    # Differencing every other step widens the interval to 2Δt
    c = Array(interior(model.tracers.c))
    expected = @. -λ * c * exp(λ * Δt)

    @test all(isapprox.(Array(interior(derivative)), expected, rtol=1e-4))

    return nothing
end

#####
##### Output writers add the updating callback on their own
#####

function test_time_derivative_dependency_adding(arch, writer_type, filename, outputs, key)
    model = relaxing_tracer_model(arch, 2)

    simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
    simulation.output_writers[:derivative] = writer_type(model, outputs(model);
                                                         filename,
                                                         schedule = IterationInterval(1),
                                                         overwrite_existing = true)
    run!(simulation)

    written = simulation.output_writers[:derivative].outputs[key]

    @test any(cb -> cb.func === written, values(simulation.callbacks))
    @test all(Array(interior(written)) .< 0)

    rm(filename, force=true)

    return nothing
end

function test_zarr_written_time_derivative(arch)
    model = relaxing_tracer_model(arch, 2)
    ∂ₜc = TimeDerivative(model.tracers.c)

    storepath = abspath(joinpath(".", "test_time_derivative.zarr"))
    isdir(storepath) && rm(storepath; recursive=true, force=true)

    simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
    simulation.output_writers[:derivative] = ZarrWriter(model, (; ∂ₜc);
                                                        filename = "test_time_derivative",
                                                        dir = ".",
                                                        schedule = IterationInterval(1),
                                                        with_halos = false,
                                                        overwrite_existing = true)
    run!(simulation)

    store = Zarr.zopen(storepath)
    saved = store["∂ₜc"][:, :, :, :]
    rm(storepath; recursive=true, force=true)

    Nx, Ny, Nz = size(model.grid)
    @test size(saved) == (Nx, Ny, Nz, 4)
    @test all(saved[:, :, :, 1] .== 0)
    @test all(saved[:, :, :, end] .< 0)

    return nothing
end

function test_written_time_derivative(arch)
    model = relaxing_tracer_model(arch, 2)
    ∂ₜc = TimeDerivative(model.tracers.c)

    filename = "test_time_derivative.jld2"
    simulation = Simulation(model, Δt=1e-3, stop_iteration=3)
    simulation.output_writers[:derivative] = JLD2Writer(model, (; ∂ₜc);
                                                        filename,
                                                        schedule = IterationInterval(1),
                                                        overwrite_existing = true)
    run!(simulation)

    written = simulation.output_writers[:derivative].outputs.∂ₜc

    file = jldopen(filename)
    saved_location = file["timeseries/∂ₜc/serialized/location"]
    initial = file["timeseries/∂ₜc/0"]
    final = file["timeseries/∂ₜc/3"]
    close(file)
    rm(filename, force=true)

    @test saved_location == (Center, Center, Center)

    # There is no history to difference against at the first output
    @test all(initial .== 0)
    @test all(final .≈ Array(parent(written)))

    return nothing
end

#####
##### Checkpointing, through both the writer and the callback
#####

function test_time_derivative_checkpointing(arch)
    prefix = "time_derivative_checkpointing_$(typeof(arch))"
    λ, Δt = 2, 1e-3

    model = relaxing_tracer_model(arch, λ)
    ∂ₜc = TimeDerivative(model.tracers.c)

    simulation = Simulation(model; Δt, stop_iteration=4)
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(4),
                                                            prefix = prefix)
    simulation.output_writers[:derivative] = JLD2Writer(model, (; ∂ₜc),
                                                        filename = "$(prefix).jld2",
                                                        schedule = IterationInterval(1),
                                                        overwrite_existing = true)
    run!(simulation)

    written = simulation.output_writers[:derivative].outputs.∂ₜc
    original_result = copy(Array(interior(written)))
    original_previous = copy(Array(interior(written.previous)))
    original_previous_time = written.previous_time

    restored_model = relaxing_tracer_model(arch, λ)
    restored_∂ₜc = TimeDerivative(restored_model.tracers.c)

    restored_simulation = Simulation(restored_model; Δt, stop_iteration=8)
    restored_simulation.output_writers[:checkpointer] = Checkpointer(restored_model,
                                                                     schedule = IterationInterval(4),
                                                                     prefix = prefix)
    restored_simulation.output_writers[:derivative] = JLD2Writer(restored_model, (; ∂ₜc=restored_∂ₜc),
                                                                 filename = "$(prefix)_restored.jld2",
                                                                 schedule = IterationInterval(1),
                                                                 overwrite_existing = true)

    set!(restored_simulation; iteration=4)

    restored_written = restored_simulation.output_writers[:derivative].outputs.∂ₜc

    @test restored_written.previous_time == original_previous_time
    @test all(Array(interior(restored_written)) .≈ original_result)
    @test all(Array(interior(restored_written.previous)) .≈ original_previous)

    rm("$(prefix).jld2", force=true)
    rm("$(prefix)_restored.jld2", force=true)
    rm("$(prefix)_iteration4.jld2", force=true)

    return nothing
end

function test_time_derivative_callback_checkpointing(arch)
    prefix = "time_derivative_callback_checkpointing_$(typeof(arch))"
    λ, Δt = 2, 1e-3

    model = relaxing_tracer_model(arch, λ)
    ∂ₜc = TimeDerivativeCallback(model.tracers.c)

    simulation = Simulation(model; Δt, stop_iteration=4)
    simulation.callbacks[:∂ₜc] = ∂ₜc
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(4),
                                                            prefix = prefix)
    run!(simulation)

    original_result = copy(Array(interior(∂ₜc.func)))
    original_previous_time = ∂ₜc.func.previous_time

    restored_model = relaxing_tracer_model(arch, λ)
    restored_∂ₜc = TimeDerivativeCallback(restored_model.tracers.c)

    restored_simulation = Simulation(restored_model; Δt, stop_iteration=8)
    restored_simulation.callbacks[:∂ₜc] = restored_∂ₜc
    restored_simulation.output_writers[:checkpointer] = Checkpointer(restored_model,
                                                                     schedule = IterationInterval(4),
                                                                     prefix = prefix)

    set!(restored_simulation; iteration=4)

    @test restored_∂ₜc.func.previous_time == original_previous_time
    @test all(Array(interior(restored_∂ₜc.func)) .≈ original_result)

    rm("$(prefix)_iteration4.jld2", force=true)

    return nothing
end

#####
##### Run
#####

@testset "TimeDerivative" begin
    for arch in archs
        @info "  Testing TimeDerivative [$(typeof(arch))]..."

        @testset "TimeDerivative of Fields and Reductions [$(typeof(arch))]" begin
            test_time_derivative_of_field(arch)
            test_time_derivative_operators(arch)
            test_time_derivative_seeding(arch)
            test_time_derivative_of_reduction(arch)
            test_time_derivative_schedule(arch)
        end

        @testset "TimeDerivative output [$(typeof(arch))]" begin
            test_time_derivative_dependency_adding(arch, JLD2Writer, "test_time_derivative.jld2",
                                                   time_derivative_outputs, :∂ₜc)

            test_time_derivative_dependency_adding(arch, NetCDFWriter, "test_time_derivative.nc",
                                                   named_time_derivative_outputs, "dcdt")

            test_written_time_derivative(arch)
            test_zarr_written_time_derivative(arch)
        end

        @testset "TimeDerivative checkpointing [$(typeof(arch))]" begin
            test_time_derivative_checkpointing(arch)
            test_time_derivative_callback_checkpointing(arch)
        end
    end
end
