include("dependencies_for_runtests.jl")

using Oceananigans.Grids: MultiEnvelopeVerticalDiscretization
using Oceananigans.Models: ZStarCoordinate

# Horizontal pressure-gradient error (HPGE) on multi-envelope grids.
#
# The chain-rule slope correction ∂ϕ/∂x|_z = ∂ϕ/∂x|_r − (∂z/∂x)·∂ϕ/∂z is inherited from the mutable-grid
# operators and uses the (fixed) ME znode, so on FLAT computational levels a stratified ocean at rest must
# stay exactly at rest. On sloped levels Oceananigans' subtraction-form pressure gradient leaves a residual
# HPGE that scales with the slope (it is larger than the paper's NEMO pressure-Jacobian scheme); ME relies
# on smooth, gentle envelopes — or a future pressure-Jacobian PGF — to keep it small. We assert the exact
# property (flat ⇒ no spurious flow) and that the error shrinks monotonically as the slope is reduced.
function spurious_u(e3func; steps=100, Δt=30.0)
    z = MultiEnvelopeVerticalDiscretization(collect(range(-1000, 0, length=21));
                                            formulation=MultiEnvelope(level_counts=(8, 6, 6)))
    grid = RectilinearGrid(size=(16, 4, 20), x=(0, 1e5), y=(0, 2.5e4), z,
                           topology=(Bounded, Periodic, Bounded))
    materialize_envelopes!(grid, ((x, y) -> 200.0, (x, y) -> 450.0, e3func))
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=10),
                                        tracers = (:b,), buoyancy = BuoyancyTracer(), coriolis = nothing,
                                        timestepper = :SplitRungeKutta3, vertical_coordinate = ZStarCoordinate())
    set!(model, b=(x, y, z) -> 1e-4 * z)   # horizontally uniform stratification ⇒ true PGF = 0
    umax = 0.0
    for _ in 1:steps
        time_step!(model, Δt)
        umax = max(umax, maximum(abs, Array(interior(model.velocities.u))))
    end
    return umax
end

@testset "Multi-envelope HPGE: exact on flat levels, controllable by slope" begin
    @test spurious_u((x, y) -> 750.0) < 1e-10                       # flat ⇒ no spurious current
    u_steep  = spurious_u((x, y) -> 600 + 300 * (x / 1e5))
    u_gentle = spurious_u((x, y) -> 744 + 12  * (x / 1e5))
    @test u_gentle < u_steep                                        # HPGE shrinks with gentler slope
    @test isfinite(u_steep)                                         # bounded (no blow-up)
end
