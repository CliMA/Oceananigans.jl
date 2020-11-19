using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded

@testset "Shallow Water Models" begin
    @info "Testing shallow water models..."

    @testset "Model constructor errors" begin
        grid = RegularCartesianGrid(size=(1, 1, 1), extent=(1, 1, 1))
        @test_throws TypeError ShallowWaterModel(architecture=CPU, grid=grid)
        @test_throws TypeError ShallowWaterModel(architecture=GPU, grid=grid)
    end

    topos = (
             (Periodic, Periodic,  Bounded),
             (Periodic,  Bounded,  Bounded),
             (Bounded,   Bounded,  Bounded),
            )

    for topo in topos
        @testset "$topo model construction" begin
            @info "  Testing $topo model construction..."
            for arch in archs, FT in float_types
		        arch isa GPU && topo == (Bounded, Bounded, Bounded) && continue

                grid = RegularCartesianGrid(FT, topology=topo, size=(1, 1, 1), extent=(1, 2, 3))
                model = ShallowWaterModel(grid=grid, architecture=arch, float_type=FT)

                # Just testing that the model was constructed with no errors/crashes.
                @test model isa ShallowWaterModel

                # Test that the grid didn't get mangled
                @test grid === model.grid

                too_big_grid = RegularCartesianGrid(FT, topology=topo, size=(1, 1, 2), extent=(1, 2, 3))

                @test_throws ArgumentError ShallowWaterModel(grid=too_big_grid, architecture=arch, float_type=FT)
            end
        end
    end

    @testset "Setting ShallowWaterModel fields" begin
        @info "  Testing setting shallow water model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 1)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            model = ShallowWaterModel(grid=grid, architecture=arch, float_type=FT)

            x, y, z = nodes((Face, Cell, Cell), grid, reshape=true)

            uh₀(x, y, z) = x * y^2
            uh_answer = @. x * y^2

            h₀ = rand(size(grid)...)
            h_answer = deepcopy(h₀)

            set!(model, uh=uh₀, h=h₀)

            uh, vh, h = model.solution

            @test all(interior(uh) .≈ uh_answer)
            @test all(interior(h) .≈ h_answer)
        end
    end
end
