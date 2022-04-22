include("dependencies_for_runtests.jl")

using Statistics: mean
using Oceananigans.CubedSpheres
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

@testset "Cubed spheres" begin

    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        grid = ConformalCubedSphereGrid(face_size=(10, 10, 1), z=(-1, 0))
        @test try show(grid); println(); true; catch; false; end
    end

    for arch in archs

        @info "  Constructing a ConformalCubedSphereGrid from file [$(typeof(arch))]..."

        # Prototype grid and model for subsequent tests
        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
        grid = ConformalCubedSphereGrid(cs32_filepath, arch, Nz=1, z=(-1, 0))

        @info "  Constructing a HydrostaticFreeSurfaceModel on a ConformalCubedSphereGrid [$(typeof(arch))]..."

        free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)
        model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                            momentum_advection = VectorInvariant(),
                                            coriolis = nothing,
                                            closure = nothing,
                                            tracers = :c,
                                            buoyancy = nothing)

        @testset "Constructing a grid from file [$(typeof(arch))]" begin
            @test grid isa ConformalCubedSphereGrid
        end

        @testset "CubedSphereData and CubedSphereFields [$(typeof(arch))]" begin
            @info "  Testing CubedSphereData and CubedSphereFields [$(typeof(arch))]..."
            c = model.tracers.c
            η = model.free_surface.η

            set!(c, 0)
            set!(η, 0)

            CUDA.allowscalar(true)
            @test all(all(face_c .== 0) for face_c in faces(c))
            @test all(all(face_η .== 0) for face_η in faces(η))
            CUDA.allowscalar(false)

            @test maximum(abs, c) == 0
            @test minimum(abs, c) == 0
            @test mean(c) == 0

            @test maximum(abs, η) == 0
            @test minimum(abs, η) == 0
            @test mean(η) == 0
        end

        @testset "Constructing a HydrostaticFreeSurfaceModel [$(typeof(arch))]" begin
            @test model isa HydrostaticFreeSurfaceModel
        end

        @testset "Time stepping a HydrostaticFreeSurfaceModel [$(typeof(arch))]" begin
            @info "  Time-stepping HydrostaticFreeSurfaceModel on a ConformalCubedSphereGrid [$(typeof(arch))]..."
            time_step!(model, 1)
            @test try time_step!(model, 1); true; catch; false; end
        end

        @testset "VerticalVorticityField on ConformalCubedSphereGrid [$(typeof(arch))]" begin
            @info "  Testing VerticalVorticityField on a ConformalCubedSphereGrid [$(typeof(arch))]..."
            ζ = VerticalVorticityField(model)

            @test ζ isa Field

            set!(model, u = (x, y, z) -> rand())

            @test try
                compute!(ζ)
                true
            catch err
                println(sprint(showerror, err))
                false
            end
            @test maximum(abs, ζ) > 0 # fingers crossed
        end
    end
end
