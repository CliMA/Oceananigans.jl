using Oceananigans.CubedSpheres
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

@testset "Cubed spheres" begin

    arch = CPU()

    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        grid = ConformalCubedSphereGrid(face_size=(10, 10, 1), z=(-1, 0))
        @test try show(grid); println(); true; catch; false; end
    end

    # Prototype grid and model for subsequent tests
    cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
    grid = ConformalCubedSphereGrid(cs32_filepath, Nz=1, z=(-1, 0))

    model = HydrostaticFreeSurfaceModel(
              architecture = arch,
                      grid = grid,
        momentum_advection = VectorInvariant(),
              free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                  coriolis = nothing,
                   closure = nothing,
                   tracers = :c,
                  buoyancy = nothing
    )

    @testset "Constructing a grid from file" begin
        @test grid isa ConformalCubedSphereGrid
    end

    @testset "CubedSphereData and CubedSphereFields" begin
        c = model.tracers

        @test try fill_halo_regions!(c, arch); true; catch; false; end
        @test try set!(c, 0); true; catch; false; end
        @test maximum(abs, c) == 0
        @test minimum(abs, c) == 0
        @test mean(c) == 0
    end

    @testset "Constructing a HydrostaticFreeSurfaceModel" begin
        @test model isa HydrostaticFreeSurfaceModel
    end

    @testset "Time stepping a HydrostaticFreeSurfaceModel" begin
        time_step!(model, 1)
        @test try time_step!(model, 1); true; catch; false; end
    end

    @testset "KernelComputedField on the CubedSphere" begin
        ζ = VerticalVorticityField(model)

        @test ζ isa KernelComputedField    

        set!(model, u = (x, y, z) -> rand())

        @test try compute!(ζ); true; catch; false; end
        @test maximum(abs, ζ) > 0 # fingers crossed
    end
end
