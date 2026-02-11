using Oceananigans
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky

# Test 1: NonhydrostaticModel with DynamicSmagorinsky
println("Testing NonhydrostaticModel with DynamicSmagorinsky...")
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
closure = DynamicSmagorinsky()
model = NonhydrostaticModel(grid; closure)

# Run one time step - this will test initialize_closure_fields! and step_closure_prognostics!
simulation = Simulation(model, Δt=0.01, stop_iteration=2)
run!(simulation)

println("✓ NonhydrostaticModel test passed")

# Test 2: HydrostaticFreeSurfaceModel with DynamicSmagorinsky
println("Testing HydrostaticFreeSurfaceModel with DynamicSmagorinsky...")
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1), topology=(Periodic, Periodic, Bounded))
closure = DynamicSmagorinsky()
model = HydrostaticFreeSurfaceModel(grid; closure)

simulation = Simulation(model, Δt=0.01, stop_iteration=2)
run!(simulation)

println("✓ HydrostaticFreeSurfaceModel test passed")

# Test 3: Check that previous_compute_time is properly initialized
println("Testing previous_compute_time initialization...")
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
closure = DynamicSmagorinsky()
model = NonhydrostaticModel(grid; closure)

# Should not be NaN after model construction
@assert !isnan(model.closure_fields.previous_compute_time[]) "previous_compute_time should not be NaN"

simulation = Simulation(model, Δt=0.01, stop_iteration=1)
run!(simulation)

# Should still not be NaN after running
@assert !isnan(model.closure_fields.previous_compute_time[]) "previous_compute_time should not be NaN after running"

println("✓ Initialization test passed")

println("\n✓✓✓ All tests passed! ✓✓✓")
