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

    model = HydrostaticFreeSurfaceModel(grid;
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


@testset "CUDA newton_div" begin
    # Test that error is small for random denominators from a single binade
    Random.seed!(44)
    test_input = CuArray(rand(1024)) .+ 1.0

    ref = similar(test_input)
    output_via_f32 = similar(test_input)
    output_via_f64 = similar(test_input)

    ref .= π ./ test_input
    output_via_f32 .= Oceananigans.Utils.newton_div.(Float32, π, test_input)
    output_via_f64 .= Oceananigans.Utils.newton_div.(Float64, π, test_input)

    # Both Float32 and Float64 should call the same function
    @test output_via_f32 == output_via_f64

    @test isapprox(ref, output_via_f64)
end
