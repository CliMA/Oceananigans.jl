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

@inline function kinetic_energyᶜᶜᶜ(i, j, k, grid, uᶜᶜᶜ, vᶜᶜᶜ)
    @inbounds u² = uᶜᶜᶜ[i, j, k]^2
    @inbounds v² = vᶜᶜᶜ[i, j, k]^2
    return (u² + v²) / 2
end

function kinetic_energy(u, v)
    grid  = u.grid
    ke_op = KernelFunctionOperation{Center, Center, Center}(kinetic_energyᶜᶜᶜ, grid, u, v)
    ke    = Field(ke_op)
    return compute!(ke)
end

function pointwise_approximate_equal(field, val)
    CPU_field = on_architecture(CPU(), field)
    @test all(interior(CPU_field) .≈ val)
end

# A purely zonal flow with an west-east velocity > 0
# on a cubed sphere in an intrinsic coordinate system
# has the following properties:
function test_purely_zonal_flow(uᵢ, vᵢ, grid)
    c1 = maximum(uᵢ) ≈ - minimum(vᵢ)
    c2 = minimum(uᵢ) ≈ - maximum(vᵢ)
    c3 = mean(uᵢ) ≈ - mean(vᵢ)
    c4 = mean(uᵢ) > 0 # The mean value should be positive)

    return c1 & c2 & c3 & c4
end

# A purely meridional flow with a south-north velocity > 0
# on a cubed sphere in an intrinsic coordinate system
# has the following properties:
function test_purely_meridional_flow(uᵢ, vᵢ, grid)
    c1 = maximum(uᵢ) ≈ maximum(vᵢ)
    c2 = minimum(uᵢ) ≈ minimum(vᵢ)
    c3 = mean(uᵢ) ≈ mean(vᵢ)
    c4 = mean(vᵢ) > 0 # The mean value should be positive

    return c1 & c2 & c3 & c4
end

function test_vector_rotation(grid)
    u = CenterField(grid)
    v = CenterField(grid)

    # Purely longitudinal flow in the extrinsic coordinate system
    fill!(u, 1)
    fill!(v, 0)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_y_component, grid, u, v)

    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, as well as their mean value should
    # be equivalent on an "intrinsic" frame
    @test test_purely_zonal_flow(uᵢ, vᵢ, grid)

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @apply_regionally pointwise_approximate_equal(KE, 0.5)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)

    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a
    # purely zonal flow in the extrensic frame (v ≈ 0)
    if architecture(grid) isa CPU
        # Note that on the GPU, there are (apparently?) larger numerical errors
        # which lead to -1e-17 < vₑ < 1e-17 for which this test fails.
        @apply_regionally pointwise_approximate_equal(vₑ, 0)
    end

    @apply_regionally pointwise_approximate_equal(uₑ, 1)

    # Purely meridional flow in the extrinsic coordinate system
    fill!(u, 0)
    fill!(v, 1)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_y_component, grid, u, v)

    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, as well as their mean value should
    # be equivalent on an "intrinsic" frame
    @test test_purely_meridional_flow(uᵢ, vᵢ, grid)

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @apply_regionally pointwise_approximate_equal(KE, 0.5)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)

    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a
    # purely zonal flow in the extrensic frame (v ≈ 0)
    @apply_regionally pointwise_approximate_equal(vₑ, 1)

    if architecture(grid) isa CPU
        # Note that on the GPU, there are (apparently?) larger numerical errors
        # which lead to - 4e-17 < uₑ < 4e-17 for which this test fails.
        @apply_regionally pointwise_approximate_equal(uₑ, 0)
    end

    # Mixed zonal and meridional flow.
    fill!(u, 0.5)
    fill!(v, 0.5)

    # Convert it to an "Instrinsic" reference frame
    uᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_x_component, grid, u, v)
    vᵢ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_y_component, grid, u, v)

    uᵢ = compute!(Field(uᵢ))
    vᵢ = compute!(Field(vᵢ))

    # The extrema of u and v, should be equivalent on an "intrinsic" frame
    # when u == v on an extrinsic frame
    @test maximum(uᵢ) ≈ maximum(vᵢ)
    @test minimum(uᵢ) ≈ minimum(vᵢ)

    # Kinetic energy should remain the same
    KE = kinetic_energy(uᵢ, vᵢ)
    @apply_regionally pointwise_approximate_equal(KE, 0.25)

    # Convert it back to a purely zonal velocity (vₑ == 0)
    uₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_x_component, grid, uᵢ, vᵢ)
    vₑ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_y_component, grid, uᵢ, vᵢ)

    uₑ = compute!(Field(uₑ))
    vₑ = compute!(Field(vₑ))

    # Make sure that the flow was converted back to a
    # purely zonal flow in the extrensic frame (v ≈ 0)
    @apply_regionally pointwise_approximate_equal(vₑ, 0.5)
    @apply_regionally pointwise_approximate_equal(uₑ, 0.5)
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
