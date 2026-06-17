include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, AbstractMutableGrid, MultiEnvelopeGrid,
    static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ
using Oceananigans.Operators: σⁿ, σ⁻, ∂t_σ

@testset "MultiEnvelope grid construction and dispatch" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(-1000:100:0))

        for topology in [(Periodic, Periodic, Bounded), (Bounded, Bounded, Bounded)]
            grid = RectilinearGrid(arch; size=(4, 4, 10), x=(0, 1), y=(0, 1), z, topology)

            @test grid isa AbstractMutableGrid          # dispatches the z-star step path
            @test grid isa MultiEnvelopeGrid
            @test size(grid, 3) == 10

            # Base.show must not error
            @test (show(IOBuffer(), grid.z); true)
        end
    end
end

@testset "MultiEnvelope resting depth and baseline metric (σᵉ = 1 ⇒ z-star)" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(-1000:100:0))
        grid = RectilinearGrid(arch; size=(4, 4, 10), x=(0, 1), y=(0, 1), z,
                               topology=(Periodic, Periodic, Bounded))

        # With σᵉ = 1 the physical resting depth equals the reference extent Lz at every stagger.
        @test static_column_depthᶜᶜᵃ(2, 2, grid) ≈ grid.Lz
        @test static_column_depthᶠᶜᵃ(2, 2, grid) ≈ grid.Lz
        @test static_column_depthᶜᶠᵃ(2, 2, grid) ≈ grid.Lz
        @test static_column_depthᶠᶠᵃ(2, 2, grid) ≈ grid.Lz

        # σ = σᵉ · σ_fs reduces to the z-star scaling (σ_fs = 1 at rest), independent of k.
        for k in 1:size(grid, 3)
            @test σⁿ(2, 2, k, grid, Center(), Center(), Center()) ≈ 1
            @test σ⁻(2, 2, k, grid, Center(), Center(), Center()) ≈ 1
            @test ∂t_σ(2, 2, k, grid) ≈ 0
        end
    end
end
