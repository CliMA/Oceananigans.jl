include("dependencies_for_runtests.jl")
using Reactant

Reactant.set_default_backend("cpu")

@testset "Reactant FFT-based model construction" begin
    reactnt_arch = ReactantState()

    @testset "NonhydrostaticModel 3D (Periodic, Periodic, Periodic)" begin
        grid = RectilinearGrid(reactnt_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))

        model = NonhydrostaticModel(grid)
        @test model isa NonhydrostaticModel
        @test model.grid.architecture isa ReactantState
    end

    @testset "NonhydrostaticModel 2D (Periodic, Periodic, Flat)" begin
        grid = RectilinearGrid(reactnt_arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))

        model = NonhydrostaticModel(grid)
        @test model isa NonhydrostaticModel
    end
end

#####
##### Time-stepping sanity check: invoke pressure correction via time_step!
#####
# NonhydrostaticModel uses FFT-based pressure solver (for Periodic topology).
# This is direct execution only - NO Reactant compilation.

@testset "Reactant FFT-based model time-stepping (direct execution)" begin
    reactnt_arch = ReactantState()
    @testset "NonhydrostaticModel 3D time_step! (invokes FFT pressure solver)" begin
        grid = RectilinearGrid(reactnt_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        
        model = NonhydrostaticModel(grid)
        @test model isa NonhydrostaticModel
        
        # Take a time step - this invokes the FFT-based pressure solver
        Δt = 0.001
        time_step!(model, Δt)
        
        @test model.clock.iteration == 1
        @test model.clock.time ≈ Δt
    end

    @testset "NonhydrostaticModel 2D time_step!" begin
        grid = RectilinearGrid(reactnt_arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        
        model = NonhydrostaticModel(grid)
        
        Δt = 0.001
        time_step!(model, Δt)
        
        @test model.clock.iteration == 1
    end
end
