include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, znode
using Oceananigans.Models: ZStarCoordinate

# CILF (Bruciaferri et al. 2018, §3.2.3): cold-intermediate-layer / pycnocline transport. With a
# pycnocline-following upper envelope, a passive tracer initialised on the pycnocline advects along it with
# little spurious vertical spreading (the computational levels track the pycnocline). Qualitative validation:
# the experiment runs stably on the ME coordinate, conserves the tracer, and the tracer stays concentrated
# near the pycnocline (centre of mass near −100 m, vertical spread bounded). Full quantitative comparison of
# tracer-variance decay against the high-resolution reference needs the paper's multi-day integration.
@testset "CILF: tracer transport along a pycnocline-following envelope" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=17));
                                                formulation=MultiEnvelope(level_counts=(8, 8)))
        grid = RectilinearGrid(arch; size=(32, 4, 16), x=(0, 2e5), y=(0, 2.5e4), z,
                               topology=(Bounded, Periodic, Bounded))
        materialize_envelopes!(grid, ((x, y) -> 100.0, (x, y) -> 1000.0))   # upper envelope = pycnocline

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=20),
                                            tracers = (:b, :c), buoyancy = BuoyancyTracer(),
                                            coriolis = FPlane(f=1e-4), timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        # two-layer stratification (pycnocline at −100 m); passive tracer band on the pycnocline
        set!(model, b=(x, y, z) -> z > -100 ? 0.0 : -2e-3, c=(x, y, z) -> abs(z + 100) < 40 ? 1.0 : 0.0)

        ∫c = Field(Integral(model.tracers.c)); compute!(∫c)
        ∫c₀ = Array(interior(∫c, 1, 1, 1))[1]
        zc = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:32, k in 1:16]
        cw(m) = dropdims(sum(Array(interior(m.tracers.c)), dims=2), dims=2)
        com₀ = sum(cw(model) .* zc) / sum(cw(model))
        spread₀ = sqrt(sum(cw(model) .* (zc .- com₀).^2) / sum(cw(model)))

        for _ in 1:100
            time_step!(model, 120.0)
        end
        compute!(∫c)
        comₙ = sum(cw(model) .* zc) / sum(cw(model))
        spreadₙ = sqrt(sum(cw(model) .* (zc .- comₙ).^2) / sum(cw(model)))

        @test all(isfinite, Array(interior(model.velocities.u)))            # stable
        @test isapprox(Array(interior(∫c, 1, 1, 1))[1], ∫c₀; rtol=1e-8)     # tracer conserved
        @test -130 < comₙ < -60                                             # stays near the pycnocline
        @test spreadₙ < spread₀ + 10                                        # little spurious vertical spreading
    end
end
