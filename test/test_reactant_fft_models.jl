using Test
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA

Reactant.set_default_backend("cpu")

@testset "Reactant FFT-based model construction" begin
    arch = ReactantState()

    @testset "NonhydrostaticModel 3D (Periodic, Periodic, Periodic)" begin
        grid = RectilinearGrid(arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        
        model = NonhydrostaticModel(grid)
        @test model isa NonhydrostaticModel
        @test model.grid.architecture isa ReactantState
    end

    @testset "NonhydrostaticModel 2D (Periodic, Periodic, Flat)" begin
        grid = RectilinearGrid(arch; size=(4, 4), extent=(1, 1),
                               topology=(Periodic, Periodic, Flat))
        
        model = NonhydrostaticModel(grid)
        @test model isa NonhydrostaticModel
    end
end
