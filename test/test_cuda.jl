include("dependencies_for_runtests.jl")

using CUDA

@testset "CUDA extension" for newton_div_type in (Oceananigans.Utils.BackendOptimizedDivision,
                                                  Oceananigans.Utils.ConvertingDivision{Float32})
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
                                        momentum_advection = WENO(order=5; newton_div=newton_div_type),
                                        tracer_advection = WENO(order=5; newton_div=newton_div_type),
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


@testset "CUDA newton_div" for FT in (Float32, Float64)
    test_input = CuArray(test_data_in_single_binade(FT, 1024))

    WCT = Oceananigans.Utils.BackendOptimizedDivision

    ref = similar(test_input)
    output = similar(test_input)

    ref .= FT(π) ./ test_input
    output .= Oceananigans.Utils.newton_div.(WCT, FT(π), test_input)

    @test isapprox(ref, output)
end
