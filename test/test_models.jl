using Oceananigans, JULES

@testset "Models" begin
    @info "Testing models..."

    @testset "Modified potential temperature" begin
        @info "  Testing model construction with Θᵐ prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        pt   = ModifiedPotentialTemperature()

        model = CompressibleModel(grid=grid, prognostic_temperature=pt, tracers=(:Θᵐ,))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, prognostic_temperature=pt, tracers=(:T,)))
    end

    @testset "Entropy" begin
        @info "  Testing model construction with S prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        pt   = Entropy()

        model = CompressibleModel(grid=grid, prognostic_temperature=pt, tracers=(:S,))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, prognostic_temperature=pt, tracers=(:T,)))
    end
end
