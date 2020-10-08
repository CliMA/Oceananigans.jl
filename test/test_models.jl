@testset "Models" begin
    @info "Testing models..."

    @testset "Energy thermodynamic variable" begin
        @info "  Testing model construction with energy thermodynamic variable..."

        grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
        model = CompressibleModel(grid = grid, gases = DryEarth(),
                                  thermodynamic_variable = Energy())
        @test model isa CompressibleModel
    end

    @testset "Entropy thermodynamic variable" begin
        @info "  Testing model construction with entropy thermodynamic variable..."

        grid = RegularCartesianGrid(size=(16, 16, 16), extent=(1, 1, 1))
        model = CompressibleModel(grid = grid, gases = DryEarth(),
                                  thermodynamic_variable = Entropy())
        @test model isa CompressibleModel
    end
end
