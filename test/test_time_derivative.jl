include("dependencies_for_runtests.jl")

using NCDatasets
using Zarr
using Oceananigans: initialize!
using Oceananigans.OutputWriters: TimeDerivative, update_time_derivative!

#####
##### A tracer relaxed at rate λ obeys ∂ₜc = -λ c exactly, which lets the backward
##### difference be checked against an analytical solution.
#####

function relaxing_tracer_model(arch, λ=2)
    grid = RectilinearGrid(arch, size=(2, 2, 2), extent=(1, 1, 1))
    forcing = (; c = Relaxation(rate=λ))
    model = NonhydrostaticModel(grid; tracers=:c, forcing)
    set!(model, c=1)
    return model
end

# Three derivatives share one simulation: a field on the default schedule, a reduction,
# and a field differenced every other step
function test_time_derivative_evolution(arch)
    λ, Δt = 2, 1e-3
    model = relaxing_tracer_model(arch, λ)
    c = model.tracers.c

    ∂ₜc = TimeDerivativeCallback(c)
    ∂ₜ∫c² = TimeDerivativeCallback(Integral(c^2))
    coarse_∂ₜc = TimeDerivativeCallback(c, schedule=IterationInterval(2))

    simulation = Simulation(model; Δt, stop_iteration=2)
    simulation.callbacks[:∂ₜc] = ∂ₜc
    simulation.callbacks[:∂ₜ∫c²] = ∂ₜ∫c²
    simulation.callbacks[:coarse_∂ₜc] = coarse_∂ₜc
    run!(simulation)

    cⁿ = Array(interior(c))

    # The backward difference is centered at t - Δt/2, where c is larger by exp(λ Δt / 2)
    @test all(isapprox.(Array(interior(∂ₜc.func)), @.(-λ * cⁿ * exp(λ * Δt / 2)), rtol=1e-4))

    # Differencing every other step widens the interval to 2Δt
    @test all(isapprox.(Array(interior(coarse_∂ₜc.func)), @.(-λ * cⁿ * exp(λ * Δt)), rtol=1e-4))

    # ∫c² decays at twice the rate of c
    ∫c² = Field(Integral(c^2))
    compute!(∫c²)
    expected = -2λ * Array(interior(∫c²))[1, 1, 1] * exp(λ * Δt)
    @test Array(interior(∂ₜ∫c².func))[1, 1, 1] ≈ expected rtol=1e-4

    # `result` is a Field, so it can be copied into a Field of the caller's choosing
    copied = CenterField(model.grid)
    set!(copied, ∂ₜc.func.result)
    @test all(Array(interior(copied)) .≈ Array(interior(∂ₜc.func)))

    return nothing
end

function test_time_derivative_operators(arch)
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    u = CenterField(grid)
    set!(u, 7)

    ∂ₜc = TimeDerivative(CenterField(grid))
    parent(∂ₜc.result) .= 1//2

    # Each generated unary method must build the operation written on `result`
    for op in (sqrt, sin, cos, exp, tanh, abs, log10, log, tan, sinh, cosh, -, +)
        @test op(∂ₜc)[1, 1, 1] == op(∂ₜc.result)[1, 1, 1]
    end

    # The three generated binary methods are identical across operators, so exercise every
    # argument pairing for one of them and the scalar pairing for the rest
    @test (∂ₜc * 2)[1, 1, 1]   == (∂ₜc.result * 2)[1, 1, 1]
    @test (2 * ∂ₜc)[1, 1, 1]   == (2 * ∂ₜc.result)[1, 1, 1]
    @test (∂ₜc * ∂ₜc)[1, 1, 1] == (∂ₜc.result * ∂ₜc.result)[1, 1, 1]
    @test (∂ₜc * u)[1, 1, 1]   == (∂ₜc.result * u)[1, 1, 1]
    @test (u * ∂ₜc)[1, 1, 1]   == (u * ∂ₜc.result)[1, 1, 1]

    for op in (+, -, /, ^, >, <, >=, <=, atan, atand, mod)
        @test op(2, ∂ₜc)[1, 1, 1] == op(2, ∂ₜc.result)[1, 1, 1]
    end

    # Chained calls fold pairwise, and the result computes through a Field
    product = Field(2 * ∂ₜc * u)
    compute!(product)
    @test Array(interior(product))[1, 1, 1] == 7

    # Reductions read the derivative directly
    @test maximum(abs, ∂ₜc) == 1//2

    return nothing
end

function test_time_derivative_initialization(arch)
    model = relaxing_tracer_model(arch)

    # Constructing with a model seeds the operand and the time immediately
    seeded = TimeDerivative(model.tracers.c, model)
    @test seeded.previous_time == model.clock.time
    @test all(Array(interior(seeded.previous)) .≈ Array(interior(model.tracers.c)))

    # Constructing without one defers seeding to `initialize!`
    ∂ₜc = TimeDerivative(model.tracers.c)
    @test all(Array(interior(∂ₜc.previous)) .== 0)

    initialize!(∂ₜc, model)
    @test ∂ₜc.previous_time == model.clock.time

    # A backward difference needs two evaluations, so nothing happens at the seeded time
    update_time_derivative!(∂ₜc, model)
    @test all(Array(interior(∂ₜc)) .== 0)

    # With a calendar clock the time type must come from the model at construction
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    clock = Clock(time=DateTime(2020, 1, 1))
    calendar_model = NonhydrostaticModel(grid; clock, tracers=:c)

    @test TimeDerivative(calendar_model.tracers.c, calendar_model).previous_time == clock.time
    @test_throws ArgumentError initialize!(TimeDerivative(calendar_model.tracers.c), calendar_model)

    return nothing
end

#####
##### Output writing, exercised once per writer backend
#####

# Each backend differs only in how outputs are keyed, how the writer is named, and how the
# saved series is read back
time_derivative_outputs(writer_type, model) = (; dcdt = TimeDerivative(model.tracers.c))
time_derivative_outputs(::Type{NetCDFWriter}, model) = Dict("dcdt" => TimeDerivative(model.tracers.c))

output_writer_kwargs(writer_type, path) = (; filename = path)
output_writer_kwargs(::Type{ZarrWriter}, path) =
    (; filename = first(splitext(basename(path))), dir = dirname(path))

read_saved_output(::Type{JLD2Writer}, path) =
    jldopen(path) do file
        cat((file["timeseries/dcdt/$i"] for i in 0:2)..., dims=4)
    end

function read_saved_output(::Type{NetCDFWriter}, path)
    dataset = NCDataset(path)
    saved = Array(dataset["dcdt"][:, :, :, :])
    close(dataset)
    return saved
end

read_saved_output(::Type{ZarrWriter}, path) = Zarr.zopen(path)["dcdt"][:, :, :, :]

function test_time_derivative_output(arch, writer_type, path)
    ispath(path) && rm(path; recursive=true, force=true)

    model = relaxing_tracer_model(arch)
    Δt = 1e-3

    simulation = Simulation(model; Δt, stop_iteration=2)
    simulation.output_writers[:derivative] =
        writer_type(model, time_derivative_outputs(writer_type, model);
                    schedule = IterationInterval(1),
                    with_halos = false,
                    overwrite_existing = true,
                    output_writer_kwargs(writer_type, path)...)
    run!(simulation)

    written = first(values(simulation.output_writers[:derivative].outputs))

    # The writer adds the updating callback on its own
    @test any(cb -> cb.func === written, values(simulation.callbacks))
    @test location(written) == (Center, Center, Center)
    @test all(Array(interior(written)) .< 0)

    # Sliced for output, so the forwarded `parent` holds exactly the interior
    @test Array(parent(written)) == Array(interior(written))

    saved = read_saved_output(writer_type, path)
    Nx, Ny, Nz = size(model.grid)

    @test size(saved) == (Nx, Ny, Nz, 3)
    @test all(saved[:, :, :, 1] .== 0)   # no history to difference against at the first output
    @test all(saved[:, :, :, end] .≈ Array(interior(written)))

    rm(path; recursive=true, force=true)

    return nothing
end

# The derivative only has to be evaluated at the output and on the iteration before it, which
# is all a difference across one time step needs
function test_time_derivative_output_schedule(arch)
    model = relaxing_tracer_model(arch)
    ∂ₜc = TimeDerivative(model.tracers.c)
    Δt = 1e-2

    simulation = Simulation(model; Δt, stop_iteration=20)
    simulation.output_writers[:derivative] = JLD2Writer(model, (; dcdt=∂ₜc);
                                                        filename = "test_dcdt_schedule.jld2",
                                                        schedule = IterationInterval(10),
                                                        with_halos = false,
                                                        overwrite_existing = true)
    run!(simulation)

    name = only(filter(key -> startswith(string(key), "TimeDerivative"), collect(keys(simulation.callbacks))))
    schedule = simulation.callbacks[name].schedule

    @test schedule isa PrecedingIterations

    actuations = filter(0:20) do iteration
        model.clock.iteration = iteration
        model.clock.last_Δt = Δt
        schedule(model)
    end

    @test actuations == [0, 9, 10, 19, 20]

    rm("test_dcdt_schedule.jld2", force=true)

    return nothing
end

#####
##### Checkpointing, through both the writer and the callback, in one pickup
#####

function test_time_derivative_checkpointing(arch)
    prefix = "time_derivative_checkpointing_$(typeof(arch))"
    Δt = 1e-3

    model = relaxing_tracer_model(arch)
    ∂ₜc = TimeDerivative(model.tracers.c)
    callback_∂ₜc = TimeDerivativeCallback(model.tracers.c)

    simulation = Simulation(model; Δt, stop_iteration=2)
    simulation.callbacks[:callback_∂ₜc] = callback_∂ₜc
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(2),
                                                            prefix = prefix)
    simulation.output_writers[:derivative] = JLD2Writer(model, (; ∂ₜc),
                                                        filename = "$(prefix).jld2",
                                                        schedule = IterationInterval(1),
                                                        overwrite_existing = true)
    run!(simulation)

    written = simulation.output_writers[:derivative].outputs.∂ₜc
    written_result = copy(Array(interior(written)))
    written_time = written.previous_time
    callback_result = copy(Array(interior(callback_∂ₜc.func)))

    restored_model = relaxing_tracer_model(arch)
    restored_∂ₜc = TimeDerivative(restored_model.tracers.c)
    restored_callback_∂ₜc = TimeDerivativeCallback(restored_model.tracers.c)

    restored_simulation = Simulation(restored_model; Δt, stop_iteration=4)
    restored_simulation.callbacks[:callback_∂ₜc] = restored_callback_∂ₜc
    restored_simulation.output_writers[:checkpointer] = Checkpointer(restored_model,
                                                                     schedule = IterationInterval(2),
                                                                     prefix = prefix)
    restored_simulation.output_writers[:derivative] = JLD2Writer(restored_model, (; ∂ₜc=restored_∂ₜc),
                                                                 filename = "$(prefix)_restored.jld2",
                                                                 schedule = IterationInterval(1),
                                                                 overwrite_existing = true)

    set!(restored_simulation; iteration=2)

    # Restored through the writer's outputs
    restored_written = restored_simulation.output_writers[:derivative].outputs.∂ₜc
    @test restored_written.previous_time == written_time
    @test all(Array(interior(restored_written)) .≈ written_result)

    # ... and through the callback in `simulation.callbacks`
    @test all(Array(interior(restored_callback_∂ₜc.func)) .≈ callback_result)

    rm("$(prefix).jld2", force=true)
    rm("$(prefix)_restored.jld2", force=true)
    rm("$(prefix)_iteration2.jld2", force=true)

    return nothing
end

#####
##### Run
#####

@testset "TimeDerivative" begin
    for arch in archs
        @info "  Testing TimeDerivative [$(typeof(arch))]..."

        @testset "TimeDerivative evolution [$(typeof(arch))]" begin
            test_time_derivative_evolution(arch)
            test_time_derivative_operators(arch)
            test_time_derivative_initialization(arch)
        end

        @testset "TimeDerivative output [$(typeof(arch))]" begin
            for (writer_type, path) in ((JLD2Writer,   "test_dcdt.jld2"),
                                        (NetCDFWriter, "test_dcdt.nc"),
                                        (ZarrWriter,   abspath("test_dcdt.zarr")))
                test_time_derivative_output(arch, writer_type, path)
            end

            test_time_derivative_output_schedule(arch)
        end

        @testset "TimeDerivative checkpointing [$(typeof(arch))]" begin
            test_time_derivative_checkpointing(arch)
        end
    end
end
