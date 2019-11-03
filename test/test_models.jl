using Oceananigans, JULES

@testset "Models" begin
    model = CompressibleModel(grid=RegularCartesianGrid(size=(10, 10, 10), length=(1, 1, 1)))
    @test isa(model, CompressibleModel)
end

