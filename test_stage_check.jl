using Oceananigans
using Oceananigans.TurbulenceClosures: DynamicSmagorinsky

println("Testing stage check in step_closure_prognostics!")
println("=" ^ 60)

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
closure = DynamicSmagorinsky()
model = NonhydrostaticModel(grid; closure, timestepper=:RungeKutta3)

# Add a counter to track updates
counter = Ref(0)
original_time = Ref(0.0)

# Simple test: run 2 iterations and see how many times previous_compute_time changes
simulation = Simulation(model, Δt=0.01, stop_iteration=2)

println("\nBefore run:")
println("  previous_compute_time = $(model.closure_fields.previous_compute_time[])")

run!(simulation)

println("\nAfter 2 iterations with RK3:")
println("  previous_compute_time = $(model.closure_fields.previous_compute_time[])")
println("  Expected: 0.02 (updated 2 times, once per iteration)")
println("  If it were 0.06, that would mean it updated 6 times (every stage)")

if abs(model.closure_fields.previous_compute_time[] - 0.02) < 1e-10
    println("\n✓ SUCCESS: Coefficient stepped once per iteration (not per stage)")
else
    println("\n✗ FAILURE: Coefficient may have stepped at every stage")
end
