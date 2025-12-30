include("dependencies_for_runtests.jl")

using CUDA

@testset "CUDA extension" begin
    cuda = CUDA.CUDABackend()
    arch = GPU(cuda)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa CuArray
    @test parent(grid.xᶜᵃᵃ) isa CuArray
    @test eltype(grid) == Float64
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa CuArray
    @test parent(model.velocities.v) isa CuArray
    @test parent(model.velocities.w) isa CuArray
    @test parent(model.tracers.b) isa CuArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end
