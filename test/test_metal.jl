include("dependencies_for_runtests.jl")

using Metal

Oceananigans.defaults.FloatType = Float32

# Note that these tests are run on a virtualization framework
# via github actions runners and may break in the future.
# More about that:
# * https://github.com/CliMA/Oceananigans.jl/pull/4124#discussion_r1976449272
# * https://github.com/CliMA/Oceananigans.jl/pull/4152

@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa MtlArray
    @test parent(grid.xᶜᵃᵃ) isa MtlArray
    @test eltype(grid) == Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa MtlArray
    @test parent(model.velocities.v) isa MtlArray
    @test parent(model.velocities.w) isa MtlArray
    @test parent(model.tracers.b) isa MtlArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end

