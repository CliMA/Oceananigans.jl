include("depedencies_for_runtests.jl")

"""
run_with_background_fields(arch; with_background=true)
Run a model with or without background fields and compare the two.
"""
function run_with_background_fields(arch; with_background=true)
    grid = RectilinearGrid(arch, size=10, z=(0, 1), topology=(Flat, Flat, Bounded))
    # Setup model with or without background fields
    if with_background
        background_fields = Oceananigans.BackgroundFields(; 
                             background_closure_fluxes=true, b=B̄_field)
        # we want no flux bottom boundary (∂B∂z = 0) and infinite ocean at the top boundary
        B_bcs = FieldBoundaryConditions(
            bottom = GradientBoundaryCondition(-N^2), # ∂B∂z = 0 → ∂b∂z = -∂B∂z = -N²
            top = GradientBoundaryCondition(0.) # ∂B∂z = ∂B̄∂z+∂b∂z = N² → ∂b∂z = 0 
        );                                
        model = NonhydrostaticModel(; grid, background_fields, tracers = :b, buoyancy=BuoyancyTracer(),
                                boundary_conditions=(; b = B_bcs))
        b = model.tracers.b
        B̄ = model.background_fields.tracers.b
        B = B̄ + b # total buoyancy field
        filepath = "with_background.nc"
    else
        # again we want no flux bottom boundary (∂B∂z = 0) and infinite ocean at the top boundary
        B_bcs = FieldBoundaryConditions(
            bottom = GradientBoundaryCondition(0), # ∂B∂z = 0
            top = GradientBoundaryCondition(N^2) # ∂B∂z =  N²
        );
        model = NonhydrostaticModel(; grid, tracers = :b, buoyancy=BuoyancyTracer(),
                                boundary_conditions=(b = B_bcs,))
        Bᵢ(z) = constant_stratification(z, 0, (; N² = N^2))
        set!(model, b=Bᵢ)  # add background buoyancy as an initial condition
        b = model.tracers.b
        B = b # total buoyancy field = perturbation buoyancy because there is no background buoyancy
        filepath = "without_background.nc"
    end
    # remove output file if existed
    !isfile(filepath) ? nothing : rm(filepath) 

    # Run for a few iterations
    simulation = Simulation(model, Δt=0.1, stop_iteration=5)
    simulation.output_writers[:profile] = NetCDFOutputWriter(model, (; B),
                                                schedule = IterationInterval(1),
                                                verbose=true,
                                                filename = filepath,
                                                overwrite_existing = true)
    run!(simulation)
    ds = NCDataset(filepath)

    
    return ds["B"][1,1,:,:]
end

# Linear background stratification
N = 1e-3
@inline constant_stratification(z, t, p) = p.N² * z
B̄_field = BackgroundField(constant_stratification, parameters=(; N² = N^2))

@testset "Background Fields Tests" begin
    for arch in test_archs
    
        # Test model runs with background fields
        @test run_with_background_fields(arch, with_background=true) !== nothing
    
        # Test model runs without background fields
        @test run_with_background_fields(arch, with_background=false) !== nothing
    
        # Test that background fields affect the solution
        b_with = run_with_background_fields(arch, with_background=true)
        b_without = run_with_background_fields(arch, with_background=false)

        # Compare the computed values
        @test all(isapprox.(b_with, b_without, rtol=1e-10))
    end
end