using Oceananigans.Models: ShallowWaterModel

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

                grid = RegularCartesianGrid(FT, topology=topo, size=(16, 16, 2), extent=(1, 2, 3))
                model = ShallowWaterModel(grid=grid, architecture=arch, float_type=FT)

                # Just testing that the model was constructed with no errors/crashes.
                @test model isa ShallowWaterModel

                # Test that the grid didn't get mangled
                @test grid === model.grid
            end
        end
    end

    #=
    @testset "Setting model fields" begin
        @info "  Testing setting model fields..."
        for arch in archs, FT in float_types
            N = (4, 4, 4)
            L = (2π, 3π, 5π)

            grid = RegularCartesianGrid(FT, size=N, extent=L)
            x, y, z = nodes((Face, Cell, Cell), grid, reshape=true)

            u₀(x, y, z) = x * y^2 * z^3
            u_answer = @. x * y^2 * z^3

            T₀ = rand(size(grid)...)
            T_answer = deepcopy(T₀)

            @test set_velocity_tracer_fields(arch, grid, :u, u₀, u_answer)
            @test set_velocity_tracer_fields(arch, grid, :T, T₀, T_answer)
            @test initial_conditions_correctly_set(arch, FT)
        end
    end
    =#
end
