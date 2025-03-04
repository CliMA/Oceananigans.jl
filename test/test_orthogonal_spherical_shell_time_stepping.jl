include("dependencies_for_runtests.jl")

using Random
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

@testset "OrthogonalSphericalShellGrid time stepping" begin
    @info "Testing OrthogonalSphericalShellGrid time stepping..."

    size = (64, 64, 2)
    latitude = (-60, 60)
    longitude = (-60, 60)
    z = (-1000, 0)
    topology = (Bounded, Bounded, Bounded)

    η₀ = 1
    Δ = 10
    ηᵢ(λ, φ, z) = η₀ * exp(-(λ^2 + φ^2) / 2Δ^2)

    g1 = LatitudeLongitudeGrid(; size, latitude, longitude, z, topology)
    g2 = RotatedLatitudeLongitudeGrid(; size, latitude, longitude, z, topology, north_pole=(0, 0))

    @test g1 isa LatitudeLongitudeGrid
    @test !(g2 isa LatitudeLongitudeGrid)
    @test g2 isa OrthogonalSphericalShellGrid

    momentum_advection = VectorInvariant()
    closure = ScalarDiffusivity(ν=2e-4, κ=2e-4)
    m1 = HydrostaticFreeSurfaceModel(grid=g1; closure, momentum_advection)
    m2 = HydrostaticFreeSurfaceModel(grid=g2; closure, momentum_advection)

    Random.seed!(123)
    ϵᵢ(λ, φ, z) = 1e-6 * randn()
    set!(m1, η=ηᵢ, u=ϵᵢ, v=ϵᵢ)

    set!(m2, η = interior(m1.free_surface.η),
             u = interior(m1.velocities.u),
             v = interior(m1.velocities.v))

    @test interior(m1.free_surface.η) == interior(m2.free_surface.η)
    @test interior(m1.velocities.u)   == interior(m2.velocities.u)
    @test interior(m1.velocities.v)   == interior(m2.velocities.v)

    for model in (m1, m2)
        simulation = Simulation(model, Δt=3minutes, stop_iteration=100)
        run!(simulation)
    end

    @test interior(m1.free_surface.η) == interior(m2.free_surface.η)
    @test interior(m1.velocities.u)   == interior(m2.velocities.u)
    @test interior(m1.velocities.v)   == interior(m2.velocities.v)
end

