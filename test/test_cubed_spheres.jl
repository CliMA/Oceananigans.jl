using Oceananigans.CubedSpheres

@testset "Cubed spheres" begin
    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        # Test show function
       grid = ConformalCubedSphereGrid(face_size=(10, 10, 1), z=(-1, 0))
       show(grid); println();
       @test grid isa ConformalCubedSphereGrid
    end
end
