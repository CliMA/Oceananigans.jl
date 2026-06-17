include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, znode
using Oceananigans.Models: ZStarCoordinate

# Cubic-spline (C¹) transition zones (Bruciaferri et al. 2018, Eq. 5b/12): a monotone cubic Hermite makes
# σᵉ depth-smooth (no jump at zone interfaces) while still passing exactly through the envelope depths, and
# stays positive (monotone ⇒ σᵉ > 0). Conservation is unaffected.
@testset "MultiEnvelope cubic-spline transitions" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=11));
                                                formulation=MultiEnvelope(level_counts=(4, 3, 3)))
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))
        materialize_envelopes!(grid, ((x, y) -> 250.0, (x, y) -> 600.0, (x, y) -> 850.0);
                               smooth_transitions=true)

        σᵉ = on_architecture(CPU(), grid.z.σᶜᶜᵉ)

        # interfaces still land exactly on the envelopes
        @test znode(3, 3, 1,  grid, Center(), Center(), Face()) ≈ -850 rtol=1e-10
        @test znode(3, 3, 4,  grid, Center(), Center(), Face()) ≈ -600 rtol=1e-10
        @test znode(3, 3, 7,  grid, Center(), Center(), Face()) ≈ -250 rtol=1e-10
        @test znode(3, 3, 11, grid, Center(), Center(), Face()) ≈ 0    atol=1e-9

        @test all(σᵉ[3, 3, k] > 0 for k in 1:10)                          # monotone ⇒ σᵉ > 0

        # smoother than the piecewise-constant case: the largest cell-to-cell σᵉ jump is well below the
        # piecewise-linear interface jump (zone2−zone1 = 1.167 − 0.625 ≈ 0.54)
        jumps = [abs(σᵉ[3, 3, k+1] - σᵉ[3, 3, k]) for k in 1:9]
        @test maximum(jumps) < 0.4
    end
end

@testset "MultiEnvelope spline conservation" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=11));
                                                formulation=MultiEnvelope(level_counts=(4, 3, 3)))
        grid = RectilinearGrid(arch; size=(8, 8, 10), x=(0, 1e5), y=(0, 1e5), z,
                               topology=(Bounded, Bounded, Bounded))
        materialize_envelopes!(grid, ((x, y) -> 250.0, (x, y) -> 600.0, (x, y) -> 850 + 50 * cos(2π * x / 1e5));
                               smooth_transitions=true)
        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                            tracers = (:c, :constant), buoyancy = nothing,
                                            timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        set!(model, c=(x, y, z) -> 1 + 0.1 * sin(2π * x / 1e5), constant=1,
             η=(x, y, z) -> 0.1 * exp(-((x - 5e4)^2) / 1e8))
        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        for _ in 1:40
            time_step!(model, 20.0)
        end
        compute!(∫c)
        @test isapprox(Array(interior(∫c, 1, 1, 1))[1], ∫c₀; rtol=1e-10)
        @test maximum(abs, Array(interior(model.tracers.constant)) .- 1) < 1e-10
    end
end
