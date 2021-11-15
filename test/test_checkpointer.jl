using Oceananigans
using CUDA
using Glob
using Test

include("utils_for_runtests.jl")

archs = test_architectures()

#####
##### Checkpointer tests
#####

function test_model_equality(test_model, true_model)
    CUDA.@allowscalar begin
        @test all(test_model.velocities.u.data     .≈ true_model.velocities.u.data)
        @test all(test_model.velocities.v.data     .≈ true_model.velocities.v.data)
        @test all(test_model.velocities.w.data     .≈ true_model.velocities.w.data)
        @test all(test_model.tracers.T.data        .≈ true_model.tracers.T.data)
        @test all(test_model.tracers.S.data        .≈ true_model.tracers.S.data)
        @test all(test_model.timestepper.Gⁿ.u.data .≈ true_model.timestepper.Gⁿ.u.data)
        @test all(test_model.timestepper.Gⁿ.v.data .≈ true_model.timestepper.Gⁿ.v.data)
        @test all(test_model.timestepper.Gⁿ.w.data .≈ true_model.timestepper.Gⁿ.w.data)
        @test all(test_model.timestepper.Gⁿ.T.data .≈ true_model.timestepper.Gⁿ.T.data)
        @test all(test_model.timestepper.Gⁿ.S.data .≈ true_model.timestepper.Gⁿ.S.data)
        @test all(test_model.timestepper.G⁻.u.data .≈ true_model.timestepper.G⁻.u.data)
        @test all(test_model.timestepper.G⁻.v.data .≈ true_model.timestepper.G⁻.v.data)
        @test all(test_model.timestepper.G⁻.w.data .≈ true_model.timestepper.G⁻.w.data)
        @test all(test_model.timestepper.G⁻.T.data .≈ true_model.timestepper.G⁻.T.data)
        @test all(test_model.timestepper.G⁻.S.data .≈ true_model.timestepper.G⁻.S.data)
    end
    return nothing
end

"""
Run two coarse rising thermal bubble simulations and make sure

1. When restarting from a checkpoint, the restarted model matches the non-restarted
   model to machine precision.

2. When using set!(test_model) to a checkpoint, the new model matches the non-restarted
   simulation to machine precision.

3. run!(test_model, pickup) works as expected
"""
function test_thermal_bubble_checkpointer_output(arch)

    #####
    ##### Create and run "true model"
    #####

    Nx, Ny, Nz = 16, 16, 16
    Lx, Ly, Lz = 100, 100, 100
    Δt = 6

    grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))
    closure = IsotropicDiffusivity(ν=4e-2, κ=4e-2)
    true_model = NonhydrostaticModel(architecture=arch, grid=grid, closure=closure,
                                     buoyancy=SeawaterBuoyancy(), tracers=(:T, :S),
                                     )
    test_model = deepcopy(true_model)

    # Add a cube-shaped warm temperature anomaly that takes up the middle 50%
    # of the domain volume.
    i1, i2 = round(Int, Nx/4), round(Int, 3Nx/4)
    j1, j2 = round(Int, Ny/4), round(Int, 3Ny/4)
    k1, k2 = round(Int, Nz/4), round(Int, 3Nz/4)
    CUDA.@allowscalar true_model.tracers.T.data[i1:i2, j1:j2, k1:k2] .+= 0.01

    true_simulation = Simulation(true_model, Δt=Δt, stop_iteration=5)

    checkpointer = Checkpointer(true_model, schedule=IterationInterval(5), force=true)
    push!(true_simulation.output_writers, checkpointer)

    run!(true_simulation) # for 5 iterations

    checkpointed_model = deepcopy(true_simulation.model)

    true_simulation.stop_iteration = 9
    run!(true_simulation) # for 4 more iterations

    #####
    ##### Test `set!(model, checkpoint_file)`
    #####

    set!(test_model, "checkpoint_iteration5.jld2")

    @test test_model.clock.iteration == checkpointed_model.clock.iteration
    @test test_model.clock.time == checkpointed_model.clock.time
    test_model_equality(test_model, checkpointed_model)

    # This only applies to QuasiAdamsBashforthTimeStepper:
    @test test_model.timestepper.previous_Δt == checkpointed_model.timestepper.previous_Δt

    #####
    ##### Test pickup from explicit checkpoint path
    #####

    test_simulation = Simulation(test_model, Δt=Δt, stop_iteration=9)

    # Pickup from explicit checkpoint path
    run!(test_simulation, pickup="checkpoint_iteration0.jld2")

    @info "Testing model equality when running with pickup=checkpoint_iteration0.jld2."
    @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
    @test test_simulation.model.clock.time == true_simulation.model.clock.time
    test_model_equality(test_model, true_model)

    run!(test_simulation, pickup="checkpoint_iteration5.jld2")
    @info "Testing model equality when running with pickup=checkpoint_iteration5.jld2."

    @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
    @test test_simulation.model.clock.time == true_simulation.model.clock.time
    test_model_equality(test_model, true_model)

    #####
    ##### Test `run!(sim, pickup=true)
    #####

    # Pickup using existing checkpointer
    test_simulation.output_writers[:checkpointer] =
        Checkpointer(test_model, schedule=IterationInterval(5), force=true)

    run!(test_simulation, pickup=true)
    @info "    Testing model equality when running with pickup=true."

    @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
    @test test_simulation.model.clock.time == true_simulation.model.clock.time
    test_model_equality(test_model, true_model)

    run!(test_simulation, pickup=0)
    @info "    Testing model equality when running with pickup=0."

    @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
    @test test_simulation.model.clock.time == true_simulation.model.clock.time
    test_model_equality(test_model, true_model)

    run!(test_simulation, pickup=5)
    @info "    Testing model equality when running with pickup=5."

    @test test_simulation.model.clock.iteration == true_simulation.model.clock.iteration
    @test test_simulation.model.clock.time == true_simulation.model.clock.time
    test_model_equality(test_model, true_model)

    rm("checkpoint_iteration0.jld2", force=true)
    rm("checkpoint_iteration5.jld2", force=true)

    return nothing
end

function run_checkpointer_cleanup_tests(arch)
    grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))
    model = NonhydrostaticModel(architecture=arch, grid=grid,
                                buoyancy=SeawaterBuoyancy(), tracers=(:T, :S),
                                )

    simulation = Simulation(model, Δt=0.2, stop_iteration=10)

    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(3), cleanup=true)
    run!(simulation)

    [@test !isfile("checkpoint_iteration$i.jld2") for i in 1:10 if i != 9]
    @test isfile("checkpoint_iteration9.jld2")

    rm("checkpoint_iteration9.jld2", force=true)

    return nothing
end

for arch in archs
    @testset "Checkpointer [$(typeof(arch))]" begin
        @info "  Testing Checkpointer [$(typeof(arch))]..."
        test_thermal_bubble_checkpointer_output(arch)
        run_checkpointer_cleanup_tests(arch)
    end
end
