include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Statistics: mean
using Oceananigans.Operators
using Oceananigans.CubedSpheres
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VerticalVorticityField

# To be used in the test below as `KernelFunctionOperation`s
@inline intrinsic_vector_x_component(i, j, k, grid, uₑ, vₑ) = 
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[1]
    
@inline intrinsic_vector_y_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[2]

@inline extrinsic_vector_x_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[1]
    
@inline extrinsic_vector_y_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[2]

function kinetic_energy(u, v)
    ke = Field(0.5 * (u * u + v * v))
    return compute!(ke)
end

function test_vector_rotation(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)
    
    # Purely longitudinal flow in the extrinsic coordinate system
    set!(u, 1)
    set!(v, 0)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Face, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Face, Center}(intrinsic_vector_y_component, grid, u, v)
    
    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, as well as their mean value should
    # be equivalent on an "intrinsic" frame
    @test maximum(uᵢ) ≈ maximum(vᵢ)
    @test minimum(uᵢ) ≈ minimum(vᵢ)
    @test mean(uᵢ) ≈ mean(vᵢ)
    @test mean(uᵢ) > 0 # The mean value should be positive

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @test all(on_architecture(CPU(), interior(KE)) .≈ 0.5)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Face, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Face, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)
    
    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a 
    # purely zonal flow in the extrensic frame (v ≈ 0)
    @test all(on_architecture(CPU(), interior(vₑ)) .≈ 0)
    @test all(on_architecture(CPU(), interior(uₑ)) .≈ 1)

    # Purely meridional flow in the extrinsic coordinate system
    set!(u, 0)
    set!(v, 1)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Face, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Face, Center}(intrinsic_vector_y_component, grid, u, v)
    
    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, as well as their mean value should
    # be equivalent on an "intrinsic" frame
    @test maximum(uᵢ) ≈ maximum(vᵢ)
    @test minimum(uᵢ) ≈ minimum(vᵢ)
    @test mean(uᵢ) ≈ mean(vᵢ)
    @test mean(vᵢ) > 0 # The mean value should be positive

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @test all(on_architecture(CPU(), interior(KE)) .≈ 0.5)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Face, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Face, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)
    
    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a 
    # purely zonal flow in the extrensic frame (v ≈ 0)
    @test all(on_architecture(CPU(), interior(vₑ)) .≈ 1)
    @test all(on_architecture(CPU(), interior(uₑ)) .≈ 0)

    # Mixed zonal and meridional flow.
    set!(u, 0.5)
    set!(v, 0.5)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Face, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Face, Center}(intrinsic_vector_y_component, grid, u, v)
    
    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, as well as their mean value should
    # be equivalent on an "intrinsic" frame
    @test maximum(uᵢ) ≈ maximum(vᵢ)
    @test minimum(uᵢ) ≈ minimum(vᵢ)
    @test mean(uᵢ) ≈ mean(vᵢ)
    @test mean(vᵢ) > 0 # The mean value should be positive

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @test all(on_architecture(CPU(), interior(KE)) .≈ 0.25)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Face, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Face, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)
    
    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a 
    # purely zonal flow in the extrensic frame (v ≈ 0)
    @test all(on_architecture(CPU(), interior(vₑ)) .≈ 0.5)
    @test all(on_architecture(CPU(), interior(uₑ)) .≈ 0.5)
end
    

@testset "Cubed spheres" begin

    @testset "Conformal cubed sphere grid" begin
        @info "  Testing conformal cubed sphere grid..."

        grid = ConformalCubedSphereGrid(panel_size=(10, 10, 1), z=(-1, 0))
        @test try show(grid); println(); true; catch; false; end
    end

    for arch in archs

        @info "  Constructing a ConformalCubedSphereGrid from file [$(typeof(arch))]..."

        # These tests cause an undefined `Bound Access Error` on GPU's CI with the new CUDA version.
        # The error is not reproducible neither on Tartarus nor on Sverdrup.
        # These are excised for the moment (PR #2253) as Cubed sphere will be reworked
        if !(arch isa GPU)
            # Prototype grid and model for subsequent tests
            cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
            grid = OldConformalCubedSphereGrid(cs32_filepath, arch, Nz=1, z=(-1, 0))

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

            @testset "Conversion from Intrinsic to Extrinsic reference frame [$(typeof(arch))]" begin
                @info "  Testing the conversion of a vector between the Intrinsic and Extrinsic reference frame"
                test_vector_rotation(grid)
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
end
