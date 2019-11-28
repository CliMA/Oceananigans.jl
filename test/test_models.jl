using Oceananigans, JULES

@testset "Models" begin
    @testset "Modified potential temperature" begin
        @info "Testing model construction with Θᵐ prognostic temperature..."
        model = CompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)))
        @test isa(model, CompressibleModel)
    end

    @testset "Entropy" begin
        @info "Testing model construction with S prognostic temperature..."
        model = CompressibleModel(grid=RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1)),
                                  prognostic_temperature=Entropy(), tracers=(:S,))
        @test isa(model, CompressibleModel)
    end
end

