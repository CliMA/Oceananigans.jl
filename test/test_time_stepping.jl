using Oceananigans, JULES

@testset "Time stepping" begin
    @info "Testing time stepping..."

    @testset "Modified potential temperature" begin
        @info "  Testing time stepping with Θᵐ prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), halo=(2, 2, 2))
        model = CompressibleModel(grid=grid)
        time_step!(model; Δt=1)

        @test model isa CompressibleModel
    end

    @testset "Entropy" begin
        @info "  Testing model construction with S prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), halo=(2, 2, 2))
        model = CompressibleModel(grid=grid, prognostic_temperature=Entropy(), tracers=(:S,))
        time_step!(model; Δt=1)

        @test model isa CompressibleModel
    end
end
