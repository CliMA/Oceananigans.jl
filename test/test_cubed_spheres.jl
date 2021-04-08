using Oceananigans.CubedSpheres
using Oceananigans.Models.HydrostaticFreeSurfaceModels

@testset "Cubed spheres" begin
    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        # Test show function
       grid = ConformalCubedSphereGrid(face_size=(10, 10, 1), z=(-1, 0))
       show(grid); println();
       @test grid isa ConformalCubedSphereGrid
    end

    @testset "Constructing a grid from file" begin
        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
        grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

        @test grid isa ConformalCubedSphereGrid
    end

    @testset "Constructing a HydrostaticFreeSurfaceModel" begin
        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
        grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

        model = HydrostaticFreeSurfaceModel(
                  architecture = CPU(),
                          grid = grid,
            momentum_advection = VectorInvariant(),
                  free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                      coriolis = nothing,
                       closure = nothing,
                       tracers = nothing,
                       buoyancy = nothing
        )

        @test model isa HydrostaticFreeSurfaceModel
    end

    @testset "Time stepping a HydrostaticFreeSurfaceModel" begin
        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
        grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

        model = HydrostaticFreeSurfaceModel(
                  architecture = CPU(),
                          grid = grid,
            momentum_advection = VectorInvariant(),
                  free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                      coriolis = nothing,
                       closure = nothing,
                       tracers = nothing,
                       buoyancy = nothing
        )

        time_step!(model, 1)

        @test model isa HydrostaticFreeSurfaceModel
    end
end
