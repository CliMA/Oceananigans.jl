include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: vertical_advection_U, vertical_advection_V

# For a uniform u, continuity reduces the vertical advection of u to the cross term of the divergence flux
# minus the same term from the vertical flux. These cancel exactly when both are interpolated with the
# same scheme, so that an inconsistent interpolation appears here as a nonzero tendency.
@testset "WENOVectorInvariant vertical flux cancels the divergence cross term" begin
    grid = RectilinearGrid(CPU(), size=(16, 16, 8), x=(0, 1e6), y=(0, 1e6), z=(-1e4, 0),
                           halo=(6, 6, 6), topology=(Periodic, Periodic, Bounded))

    prng = Random.Xoshiro(44)

    for momentum_advection in (WENOVectorInvariant(), WENOVectorInvariant(order=5))
        model = HydrostaticFreeSurfaceModel(grid; momentum_advection, buoyancy=nothing, tracers=())

        ũ = 10
        ṽ = rand(prng, size(grid)...)
        set!(model, u=ũ, v=ṽ)

        scheme = model.advection.momentum
        U = model.velocities
        Gu = [vertical_advection_U(i, j, k, grid, scheme, U) for i in 1:16, j in 1:16, k in 2:7]

        # The horizontal divergence of the random v is of order 1 / Δy
        @test maximum(abs, Gu) < 1e-10 * ũ * 16 / 1e6

        ṽ = 10
        ũ = rand(prng, size(grid)...)
        set!(model, u=ũ, v=ṽ)
        Gv = [vertical_advection_V(i, j, k, grid, scheme, U) for i in 1:16, j in 1:16, k in 2:7]
        @test maximum(abs, Gv) < 1e-10 * ṽ * 16 / 1e6
    end
end
