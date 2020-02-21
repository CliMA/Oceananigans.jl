@testset "Models" begin
    @info "Testing models..."

    @testset "Modified potential temperature" begin
        @info "  Testing model construction with Θᵐ prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        tvar = ModifiedPotentialTemperature()

        model = CompressibleModel(grid=grid, thermodynamic_variable=tvar, tracers=(:Θᵐ,))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, thermodynamic_variable=tvar, tracers=(:T,)))
    end

    @testset "Entropy" begin
        @info "  Testing model construction with S prognostic temperature..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        tvar = Entropy()

        model = CompressibleModel(grid=grid, thermodynamic_variable=tvar, tracers=(:S,))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, thermodynamic_variable=tvar, tracers=(:T,)))
    end

    @testset "VaporPlaceholder" begin
        @info "  Testing model construction with vapor placeholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        tvar = ModifiedPotentialTemperature()
        mp   = VaporPlaceholder()

        model = CompressibleModel(grid=grid, thermodynamic_variable=tvar,
                                  microphysics=mp, tracers=(:Θᵐ, :Qv))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, thermodynamic_variable=tvar,
                              microphysics=mp, tracers=(:Θᵐ, :qv)))
    end

    @testset "VaporLiquidIcePlaceholder" begin
        @info "  Testing model construction with vapor+liquid+ice placeholder microphysics..."

        grid = RegularCartesianGrid(size=(16, 16, 16), length=(1, 1, 1))
        tvar = ModifiedPotentialTemperature()
        mp   = VaporLiquidIcePlaceholder()

        model = CompressibleModel(grid=grid, thermodynamic_variable=tvar,
                                  microphysics=mp, tracers=(:Θᵐ, :Qv, :Ql, :Qi))
        @test model isa CompressibleModel

        @test_throws(ArgumentError,
            CompressibleModel(grid=grid, thermodynamic_variable=tvar,
                              microphysics=mp, tracers=(:Θᵐ, :Qv, :Ql)))
    end
end
