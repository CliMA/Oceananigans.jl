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
        @info "  Testing time stepping with S prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), halo=(2, 2, 2))
        model = CompressibleModel(grid=grid, prognostic_temperature=Entropy(), tracers=(:S,))
        time_step!(model; Δt=1)
        @test model isa CompressibleModel
    end

    @testset "VaporPlaceholder" begin
        @info "  Testing time stepping with VaporPlaceholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), halo=(2, 2, 2))
        model = CompressibleModel(grid=grid, microphysics=VaporPlaceholder(), tracers=(:Θᵐ, :Qv))
        time_step!(model, Δt=1)
        @test model isa CompressibleModel
    end

    @testset "VaporLiquidIcePlaceholder" begin
        @info "  Testing time stepping with VaporLiquidIcePlaceholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1), halo=(2, 2, 2))
        model = CompressibleModel(grid=grid, microphysics=VaporLiquidIcePlaceholder(),
                                  tracers=(:Θᵐ, :Qv, :Ql, :Qi))
        time_step!(model, Δt=1)
        @test model isa CompressibleModel
    end
end
