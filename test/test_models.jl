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

    @testset "VaporPlaceholder" begin
        @info "  Testing model construction with vapor placeholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        pt   = ModifiedPotentialTemperature()
        mp   = VaporPlaceholder()

        model = CompressibleModel(grid=grid, prognostic_temperature=pt,
                                  microphysics=mp, tracers=(:Θᵐ, :Qv))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, prognostic_temperature=pt,
                              microphysics=mp, tracers=(:Θᵐ, :qv)))
    end

    @testset "VaporLiquidIcePlaceholder" begin
        @info "  Testing model construction with vapor+liquid+ice placeholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        pt   = ModifiedPotentialTemperature()
        mp   = VaporLiquidIcePlaceholder()

        model = CompressibleModel(grid=grid, prognostic_temperature=pt,
                                  microphysics=mp, tracers=(:Θᵐ, :Qv, :Ql, :Qi))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, prognostic_temperature=pt,
                              microphysics=mp, tracers=(:Θᵐ, :Qv, :Ql)))
    end
end
