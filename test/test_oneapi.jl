include("dependencies_for_runtests.jl")

using oneAPI

@testset "oneAPI extension" begin
    FT = Float32 # our tests need to use Float32 right now
    oneapi = oneAPI.oneAPIBackend()
    Oceananigans.defaults.FloatType = FT

    arch = GPU(oneapi)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa oneArray
    @test parent(grid.xᶜᵃᵃ) isa oneArray
    @test eltype(grid) == FT
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa oneArray
    @test parent(model.velocities.v) isa oneArray
    @test parent(model.velocities.w) isa oneArray
    @test parent(model.tracers.b) isa oneArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes

    Oceananigans.defaults.FloatType = Float64 # just in case
end
