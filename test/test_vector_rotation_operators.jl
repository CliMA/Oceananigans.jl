include("dependencies_for_runtests.jl")

using Statistics: mean
using Oceananigans.Operators
using Oceananigans.Operators: rotation_angle
using Distances: haversine

# To be used in the test below as `KernelFunctionOperation`s
@inline intrinsic_vector_x_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[1]

@inline intrinsic_vector_y_component(i, j, k, grid, uₑ, vₑ) =
    @inbounds intrinsic_vector(i, j, k, grid, uₑ, vₑ)[2]

@inline extrinsic_vector_x_component(i, j, k, grid, uᵢ, vᵢ) =
    @inbounds extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)[1]

@inline extrinsic_vector_y_component(i, j, k, grid, uᵢ, vᵢ) =
    @inbounds extrinsic_vector(i, j, k, grid, uᵢ, vᵢ)[2]

function pointwise_approximate_equal(field1, field2)
    CPU_field1 = on_architecture(CPU(), field1)
    CPU_field2 = on_architecture(CPU(), field2)
    @test all(interior(CPU_field1) .≈ interior(CPU_field2))
end

# Test vector invariants i.e.
# -> dot product of two vectors
# -> cross product of two vectors
function test_vector_rotation(grid)
    x₁ = CenterField(grid)
    y₁ = CenterField(grid)
    x₂ = CenterField(grid)
    y₂ = CenterField(grid)

    set!(x₁, (x, y, z) -> rand())
    set!(y₁, (x, y, z) -> rand())
    set!(x₂, (x, y, z) -> rand())
    set!(y₂, (x, y, z) -> rand())

    fill_halo_regions!((x₁, y₁, x₂, y₂))

    d = compute!(Field(x₁ * x₂ + y₁ * y₂))
    c = compute!(Field(x₁ * y₂ - y₁ * x₂))

    @info "  Testing the conversion of a vector between to the Intrinsice reference frame on $(summary(grid))"
    xᵢ₁ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_x_component, grid, x₁, y₁)
    yᵢ₁ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_y_component, grid, x₁, y₁)
    xᵢ₂ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_x_component, grid, x₂, y₂)
    yᵢ₂ = KernelFunctionOperation{Center, Center, Center}(intrinsic_vector_y_component, grid, x₂, y₂)

    xᵢ₁ = compute!(Field(xᵢ₁))
    yᵢ₁ = compute!(Field(yᵢ₁))
    xᵢ₂ = compute!(Field(xᵢ₂))
    yᵢ₂ = compute!(Field(yᵢ₂))

    dᵢ = compute!(Field(xᵢ₁ * xᵢ₂ + yᵢ₁ * yᵢ₂))
    cᵢ = compute!(Field(xᵢ₁ * yᵢ₂ - yᵢ₁ * xᵢ₂))

    @apply_regionally pointwise_approximate_equal(dᵢ, d)
    @apply_regionally pointwise_approximate_equal(cᵢ, c)

    xₑ₁ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_x_component, grid, xᵢ₁, yᵢ₁)
    yₑ₁ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_y_component, grid, xᵢ₁, yᵢ₁)
    xₑ₂ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_x_component, grid, xᵢ₂, yᵢ₂)
    yₑ₂ = KernelFunctionOperation{Center, Center, Center}(extrinsic_vector_y_component, grid, xᵢ₂, yᵢ₂)

    xₑ₁ = compute!(Field(xₑ₁))
    yₑ₁ = compute!(Field(yₑ₁))
    xₑ₂ = compute!(Field(xₑ₂))
    yₑ₂ = compute!(Field(yₑ₂))

    @info "  Testing the conversion of a vector between to the Extrinsic reference frame on $(summary(grid))"
    @apply_regionally pointwise_approximate_equal(xₑ₁, x₁)
    @apply_regionally pointwise_approximate_equal(xₑ₁, x₁)
    @apply_regionally pointwise_approximate_equal(yₑ₂, y₂)
    @apply_regionally pointwise_approximate_equal(yₑ₂, y₂)
end

@testset "Vector rotation" begin
    for arch in archs
        @testset "Rotation between Intrinsic to Extrinsic reference frame [$(typeof(arch))]" begin

            if arch isa CPU
                @info "  Testing the calculation of the rotation angle between the Intrinsic and Extrinsic reference frame"

                # Build a custom grid that is rotated by θᵢ degrees clockwise from the vertical
                # and test that the rotation_angle is computed correctly.

                # Since we want to test the rotation angle, we build a grid with coordinates
                # that are uniformly spaced Δ << 1 apart. Small coordinate angle spacing ensures
                # that geometric factors related with spherical geometry don't come into play.
                Δ = 1e-3

                angles = [-22.5, 30, 45, 60]

                for θᵢ in angles
                    radius = 1
                    Nx, Ny = 4, 4

                    # allocate an empty grid
                    grid = OrthogonalSphericalShellGrid(; size=(Nx, Ny, 1), z=(0, 1), radius, conformal_mapping=nothing)
                    λᶠᶠᵃ = grid.λᶠᶠᵃ
                    φᶠᶠᵃ = grid.φᶠᶠᵃ

                    # fill in coordinates
                    sinθ = sind(θᵢ)
                    cosθ = cosd(θᵢ)
                    for i in 1:Nx+1, j in 1:Ny+1
                        λᶠᶠᵃ[i, j, 1] =   (i-1) * Δ * cosθ + (j-1) * Δ * sinθ
                        φᶠᶠᵃ[i, j, 1] = - (i-1) * Δ * sinθ + (j-1) * Δ * cosθ
                    end

                    # fill in metrics
                    for i in 1:Nx+1, j in 1:Ny+1
                        grid.Δxᶜᶠᵃ[i, j] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), radius)
                        grid.Δyᶠᶜᵃ[i, j] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]), (λᶠᶠᵃ[i, j], φᶠᶠᵃ[i, j]), radius)
                    end

                    # ensure that rotation_angle returns θᵢ
                    for i in 1:Nx, j in 1:Ny
                        θ = rotation_angle(i, j, grid)
                        @test θ ≈ θᵢ
                    end
                end
            end
        end

        @testset "Conversion from Intrinsic to Extrinsic reference frame [$(typeof(arch))]" begin
            @info "  Testing the conversion of a vector between the Intrinsic and Extrinsic reference frame"
            cubed_sphere_grid = ConformalCubedSphereGrid(arch; panel_size=(10, 10, 1), z=(-1, 0))
            tripolar_grid = TripolarGrid(arch; size = (40, 40, 1), z=(-1, 0))
            test_vector_rotation(cubed_sphere_grid)
            test_vector_rotation(tripolar_grid)
        end
    end
end
