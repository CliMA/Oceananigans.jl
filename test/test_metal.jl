include("dependencies_for_runtests.jl")

using Metal

Oceananigans.defaults.FloatType = Float32

@testset "MetalGPU extension" begin
    metal = Metal.MetalBackend()
    arch = GPU(metal)
    grid = RectilinearGrid(arch, size=(16, 8, 4), extent=(1, 1, 1))

    @test eltype(grid) = Float32
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end

