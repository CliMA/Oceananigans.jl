using Oceananigans
using Test
using CUDA:has_cuda


# Linear background stratification
N = 1e-3
@inline constant_stratification(z, t, p) = p.N² * z
B̄_field = BackgroundField(constant_stratification, parameters=(; N² = N^2))

"""
run_with_background_fields(arch; with_background=true)
Run a model with or without background fields and return the mean buoyancy value.
"""
function run_with_background_fields(arch; with_background=true)
    grid = RectilinearGrid(arch, size=4, z=(0, 1), topology=(Flat, Flat, Bounded))
    buoyancy = Buoyancy(model = BuoyancyTracer())
    # Setup model with or without background fields
    if with_background
        background_fields = Oceananigans.BackgroundFields(; 
                             background_closure_fluxes=true, b=B̄_field)
        model = NonhydrostaticModel(; grid, background_fields, tracers = :b, buoyancy)
        b = model.tracers.b
        B̄ = model.background_fields.tracers.b
        B = B̄ + b # total buoyancy field

    else
        model = NonhydrostaticModel(; grid, tracers = :b, buoyancy)
        b = model.tracers.b
        B = b # total buoyancy field

    end
    
    # Run for a few timesteps
    time_step!(model, 1)
    
    # Return average total buoyancy value for comparison
    return mean(abs, B)
end

@testset "Background Fields Tests" begin
    arch = CPU()
    
    # Test model runs with background fields
    @test run_with_background_fields(arch, with_background=true) !== nothing
    
    # Test model runs without background fields
    @test run_with_background_fields(arch, with_background=false) !== nothing
    
    # Test that background fields affect the solution
    b_with = run_with_background_fields(arch, with_background=true)
    b_without = run_with_background_fields(arch, with_background=false)
    
    # The total buoyancy should be different when using background fields
    @test b_with !== b_without
    
    # Test with GPU
    if CUDA.has_cuda()
        arch_gpu = GPU()
        @test run_with_background_fields(arch_gpu, with_background=true) !== nothing
    end
end