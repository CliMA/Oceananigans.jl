using Test

using Statistics: mean
using CUDA

using Oceananigans
using Oceananigans.CubedSpheres
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

include("data_dependencies.jl")

@testset "Cubed spheres" begin

    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        grid = ConformalCubedSphereGrid(face_size=(10, 10, 1), z=(-1, 0))
        @test try show(grid); println(); true; catch; false; end
    end

    for arch in archs

        @info "Constructing a ConformalCubedSphereGrid from file [$(typeof(arch))]..."

        # Prototype grid and model for subsequent tests
        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
        grid = ConformalCubedSphereGrid(cs32_filepath, architecture=arch, Nz=1, z=(-1, 0))

        @info "Constructing a HydrostaticFreeSurfaceModel on a ConformalCubedSphereGrid [$(typeof(arch))]..."

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

        @testset "Constructing a grid from file [$(typeof(arch))]" begin
            @test grid isa ConformalCubedSphereGrid
        end

        @testset "CubedSphereData and CubedSphereFields [$(typeof(arch))]" begin
            @info "Testing CubedSphereData and CubedSphereFields [$(typeof(arch))]..."
            c = model.tracers.c
            set!(c, 0)

            CUDA.allowscalar(true)
            @test all(all(face_c .== 0) for face_c in faces(c))
            CUDA.allowscalar(false)

            @test maximum(abs, c) == 0
            @test minimum(abs, c) == 0
            @test mean(c) == 0
        end

        @testset "Constructing a HydrostaticFreeSurfaceModel [$(typeof(arch))]" begin
            @test model isa HydrostaticFreeSurfaceModel
        end

        @testset "Time stepping a HydrostaticFreeSurfaceModel [$(typeof(arch))]" begin
            @info "Time-stepping HydrostaticFreeSurfaceModel on a ConformalCubedSphereGrid [$(typeof(arch))]..."
            time_step!(model, 1)
            @test try time_step!(model, 1); true; catch; false; end
        end

        @testset "KernelComputedField on ConformalCubedSphereGrid [$(typeof(arch))]" begin
            @info "Testing KernelComputedField on a ConformalCubedSphereGrid [$(typeof(arch))]..."
            ζ = VerticalVorticityField(model)

            @test ζ isa KernelComputedField    

            set!(model, u = (x, y, z) -> rand())

            @test try compute!(ζ); true; catch; false; end
            @test maximum(abs, ζ) > 0 # fingers crossed
        end
    end
end
