include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, static_column_depthᶜᶜᵃ
using Oceananigans.Models: ZStarCoordinate
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom

# ME on an ImmersedBoundaryGrid uses the immersed boundary only for lateral land masking; the bottom is
# the deepest envelope. The envelope-aware static_column_depth must report the envelope depth on ocean
# columns (not the reference extent) so the z-star closure stays consistent with σᵉ.
@testset "MultiEnvelope + immersed land masking" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=11));
                                                formulation=MultiEnvelope(level_counts=(4, 3, 3)))
        ug = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                             topology=(Bounded, Bounded, Bounded))
        materialize_envelopes!(ug, ((x, y) -> 250.0, (x, y) -> 600.0, (x, y) -> 850.0))

        # land in the western quarter (bottom_height ≥ surface), full-depth ocean elsewhere
        ibg = ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> x < 2.5e4 ? 0.0 : -1000.0))

        @test static_column_depthᶜᶜᵃ(6, 3, ibg) ≈ 850   # ocean column → envelope depth, not 1000
        @test static_column_depthᶜᶜᵃ(1, 3, ibg) ≈ 0     # land column → masked

        model = HydrostaticFreeSurfaceModel(ibg;
                                            free_surface = SplitExplicitFreeSurface(ibg; substeps=10),
                                            tracers = (:c, :constant), buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        set!(model, c=(x, y, z) -> 1 + 0.1 * sin(2π * x / 1e5), constant=1,
             η=(x, y, z) -> 0.05 * exp(-((x - 6e4)^2) / 1e8))

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        for _ in 1:30
            time_step!(model, 15.0)
        end
        compute!(∫c)

        @test isapprox(Array(interior(∫c, 1, 1, 1))[1], ∫c₀; rtol=1e-9)
        # constant tracer preserved in ocean cells (exclude masked western land columns)
        @test maximum(abs, Array(interior(model.tracers.constant))[3:8, :, :] .- 1) < 1e-9
    end
end
