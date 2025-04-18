include("dependencies_for_runtests.jl")

using AMDGPU

@testset "AMDGPU extension" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)
    grid = RectilinearGrid(arch, size=(4, 8, 16), x=[0, 1, 2, 3, 4], y=(0, 1), z=(0, 16))

    @test parent(grid.xᶠᵃᵃ) isa ROCArray
    @test parent(grid.xᶜᵃᵃ) isa ROCArray
    @test eltype(grid) == Float64
    @test architecture(grid) isa GPU

    model = HydrostaticFreeSurfaceModel(; grid,
                                        coriolis = FPlane(latitude=45),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = WENO(order=5),
                                        tracer_advection = WENO(order=5),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=60))

    @test parent(model.velocities.u) isa ROCArray
    @test parent(model.velocities.v) isa ROCArray
    @test parent(model.velocities.w) isa ROCArray
    @test parent(model.tracers.b) isa ROCArray

    simulation = Simulation(model, Δt=1minute, stop_iteration=3)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) == 3minutes
end
