include("dependencies_for_runtests.jl")

using Statistics: mean
using Oceananigans.Operators
    
# To be used in the test below as `KernelFunctionOperation`s
@inline intrinsic_vector_x_component(i, j, k, grid, uₑ, vₑ) = 
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[1]
    
@inline intrinsic_vector_y_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[2]

@inline extrinsic_vector_x_component(i, j, k, grid, uᵢ, vᵢ) =
    @inbounds extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)[1]
    
@inline extrinsic_vector_y_component(i, j, k, grid, uᵢ, vᵢ) =
    @inbounds extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)[2]

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
    
@testset "Vector rotation" begin
    for arch in archs
        @testset "Conversion from Intrinsic to Extrinsic reference frame [$(typeof(arch))]" begin
            @info "  Testing the conversion of a vector between the Intrinsic and Extrinsic reference frame"
            grid = ConformalCubedSphereGrid(arch; panel_size=(10, 10, 1), z=(-1, 0))
            test_vector_rotation(grid)
        end
    end
end
