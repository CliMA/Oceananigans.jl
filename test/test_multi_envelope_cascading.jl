include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization, znode
using Oceananigans.Models: ZStarCoordinate

# CASC (Bruciaferri et al. 2018, §3.2.2): dense water cascading down a slope. The terrain-following bottom
# envelope (LinearEnvelope) lets the computational levels follow the slope, so a dense plume descends along
# the levels rather than over z-steps. This is a *qualitative* validation — it confirms the experiment runs
# stably on the ME coordinate, conserves the dense-water tracer, and the dense water descends. The full
# quantitative comparison to the Shapiro–Hill (1997) downslope speed needs the paper's multi-day integration.
@testset "CASC: dense cascading on a terrain-following slope" begin
    for arch in test_architectures()
        z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=17)); formulation=LinearEnvelope())
        grid = RectilinearGrid(arch; size=(48, 4, 16), x=(0, 1.5e5), y=(0, 1.2e4), z,
                               topology=(Bounded, Periodic, Bounded))
        materialize_envelopes!(grid, (x, y) -> 200 + 600 * clamp((x - 2e4) / 8e4, 0, 1))   # 200 m shelf → 800 m

        model = HydrostaticFreeSurfaceModel(grid;
                                            free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                            tracers = (:b, :dense), buoyancy = BuoyancyTracer(),
                                            coriolis = FPlane(f=1e-4), timestepper = :SplitRungeKutta3,
                                            vertical_coordinate = ZStarCoordinate())
        # ambient stratification + a dense (negatively buoyant) patch on the shelf, tagged by `dense`
        set!(model, b=(x, y, z) -> 1e-5 * z - ((x < 2e4 && z > -180) ? 5e-3 : 0.0),
                    dense=(x, y, z) -> (x < 2e4 && z > -180) ? 1.0 : 0.0)

        ∫d = Field(Integral(model.tracers.dense)); compute!(∫d)
        ∫d₀ = Array(interior(∫d, 1, 1, 1))[1]
        zc = [znode(i, 1, k, grid, Center(), Center(), Center()) for i in 1:48, k in 1:16]
        dense_weight(m) = dropdims(sum(Array(interior(m.tracers.dense)), dims=2), dims=2)
        com_z₀ = sum(dense_weight(model) .* zc) / sum(dense_weight(model))

        for _ in 1:150
            time_step!(model, 120.0)
        end
        compute!(∫d)

        @test all(isfinite, Array(interior(model.velocities.u)))                       # stable
        @test isapprox(Array(interior(∫d, 1, 1, 1))[1], ∫d₀; rtol=1e-8)                 # dense tracer conserved
        @test sum(dense_weight(model) .* zc) / sum(dense_weight(model)) < com_z₀ - 1    # dense water descends
    end
end
