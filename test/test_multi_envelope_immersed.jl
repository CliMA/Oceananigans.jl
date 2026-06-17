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

# Partial topography (an immersed slope that cuts the column mid-depth, not just full land columns). The
# immersed mask is in the reference coordinate while σᵉ maps reference → physical, so the column depth must
# be the physical thickness summed over the *wet* cells; otherwise the z-star closure is inconsistent and
# blows up. This test guards that ME + a partial immersed bottom stays stable and conserves.
@testset "MultiEnvelope + immersed partial topography (slope)" begin
    for arch in test_architectures()
        slope(x) = 200 + 600 * clamp((x - 2e4) / 8e4, 0, 1)
        ug = RectilinearGrid(arch; size=(48, 4, 20), x=(0, 1.5e5), y=(0, 1.2e4), halo=(4, 4, 4),
                             topology=(Bounded, Periodic, Bounded),
                             z=MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, 21));
                                                                   formulation=MultiEnvelope(level_counts=(10, 10))))
        materialize_envelopes!(ug, ((x, y) -> 150.0, (x, y) -> 1000.0))   # flat envelopes
        ibg = ImmersedBoundaryGrid(ug, GridFittedBottom((x, y) -> -slope(x)))

        model = HydrostaticFreeSurfaceModel(ibg;
                                            free_surface = SplitExplicitFreeSurface(ibg; substeps=30),
                                            tracers = (:b, :constant), buoyancy = BuoyancyTracer(),
                                            coriolis = nothing, momentum_advection = WENO(),
                                            tracer_advection = WENO(), timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        set!(model, b=(x, y, z) -> 1e-5 * z, constant=1, η=(x, y, z) -> 0.05 * exp(-((x - 6e4)^2) / 1e8))

        ∫c = Field(Integral(model.tracers.constant)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        for _ in 1:80
            time_step!(model, 60.0)
        end
        compute!(∫c)

        @test all(isfinite, Array(interior(model.velocities.u)))            # stable (no blow-up)
        @test isapprox(Array(interior(∫c, 1, 1, 1))[1], ∫c₀; rtol=1e-9)     # conserved
    end
end
