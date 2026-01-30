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


# Generate some random points in a single binade [1;2) interval
function test_data_in_single_binade(::Type{FT}, size) where {FT}
    prng = Random.Xoshiro(44)
    return rand(prng, FT, size) .+ 1.0
end


@testset "CUDA f64 newton_div" begin
    test_input = CuArray(test_data_in_single_binade(Float64, 1024))

    ref = similar(test_input)
    output_via_f32 = similar(test_input)
    output_via_f64 = similar(test_input)

    ref .= Float64(π) ./ test_input
    output_via_f32 .= Oceananigans.Utils.newton_div.(Float32, Float64(π), test_input)
    output_via_f64 .= Oceananigans.Utils.newton_div.(Float64, Float64(π), test_input)

    # Both Float32 and Float64 should call the same function
    @test output_via_f32 == output_via_f64

    @test isapprox(ref, output_via_f64)
end


@testset "CUDA f32 newton_div" begin
    test_input = CuArray(test_data_in_single_binade(Float32, 1024))

    ref = similar(test_input)
    output_via_f32 = similar(test_input)

    ref .= Float32(π) ./ test_input
    output_via_f32 .= Oceananigans.Utils.newton_div.(Float32, Float32(π), test_input)

    # Just test that it gives reasonable approximation (i.e. within √ϵ)
    @test isapprox(ref, output_via_f32)
end
