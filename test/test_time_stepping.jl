@testset "Time stepping" begin
    @info "Testing time stepping..."

    @testset "Energy thermodynamic variable" begin
        @info "  Testing time stepping with energy thermodynamic variable..."

        grid = RegularCartesianGrid(size=(16, 16, 16), halo=(2, 2, 2), extent=(1, 1, 1))
        model = CompressibleModel(grid = grid, gases = DryEarth(),
                                  thermodynamic_variable = Energy())
        time_step!(model, 1)
        @test model isa CompressibleModel
    end

    @testset "Entropy thermodynamic variable" begin
        @info "  Testing time stepping with entropy thermodynamic variable..."

        grid = RegularCartesianGrid(size=(16, 16, 16), halo=(2, 2, 2), extent=(1, 1, 1))
        model = CompressibleModel(grid = grid, gases = DryEarth(),
                                  thermodynamic_variable = Entropy())
        time_step!(model, 1)
        @test model isa CompressibleModel
    end
end
