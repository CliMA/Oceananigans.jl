#!/usr/bin/env julia

# Test script to verify that the CG solver uses previous pressure as initial guess
using Pkg
Pkg.activate(".")

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Solvers
using Oceananigans.Fields
using LinearAlgebra: norm

println("Testing that ConjugateGradientPoissonSolver uses previous pressure as initial guess...")

# Create a simple grid
grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))

# Create a CG solver
solver = ConjugateGradientPoissonSolver(grid, maxiter=5)

# Create a pressure field and right-hand side
pressure = CenterField(grid)
rhs = solver.right_hand_side

# Test 1: Set initial pressure to a non-zero pattern
println("\nTest 1: Setting initial pressure to a known pattern...")
fill!(pressure, 0.0)

# Set a simple pattern in the pressure field
for k in 1:8, j in 1:8, i in 1:8
    pressure[i, j, k] = sin(π * i / 8) * cos(π * j / 8) * sin(π * k / 8)
end

# Create a simple RHS  
fill!(rhs, 1.0)

# Store the initial pressure pattern
initial_pressure = Array(interior(pressure))
println("Initial pressure norm: ", norm(initial_pressure))
println("Initial pressure[1,1,1]: ", pressure[1,1,1])

# Solve with CG - this should use the existing pressure as initial guess
solve!(pressure, solver.conjugate_gradient_solver, rhs)

final_pressure = Array(interior(pressure))
println("Final pressure norm: ", norm(final_pressure))
println("Final pressure[1,1,1]: ", pressure[1,1,1])
println("Number of CG iterations: ", iteration(solver.conjugate_gradient_solver))

# Test 2: Verify that if we start with zero, we get a different result
println("\nTest 2: Comparing with zero initial guess...")
pressure_zero = CenterField(grid)
fill!(pressure_zero, 0.0)

# Create a fresh solver to reset iteration count
solver2 = ConjugateGradientPoissonSolver(grid, maxiter=5)

solve!(pressure_zero, solver2.conjugate_gradient_solver, rhs)
final_pressure_zero = Array(interior(pressure_zero))

println("Final pressure (zero init) norm: ", norm(final_pressure_zero))
println("Final pressure (zero init)[1,1,1]: ", pressure_zero[1,1,1])
println("Number of CG iterations (zero init): ", iteration(solver2.conjugate_gradient_solver))

# Check if the solutions differ significantly
difference = norm(final_pressure - final_pressure_zero)
println("\nDifference between solutions: ", difference)

# The key test: the solver should have started from our initial pattern, not zero
# If the initial guess is being used properly, the two solutions should be different
# (unless by coincidence the pattern was already close to the solution)
if difference > 1e-10
    println("✓ SUCCESS: The solver is using the initial pressure field as starting guess!")
    println("  Solutions differ significantly, indicating initial guess was used.")
else
    println("? UNCLEAR: Solutions are very similar.")
    println("  This could mean either:")
    println("  1. The initial guess was already very close to the solution, or")
    println("  2. The solver is not using the initial guess properly.")
end

println("\nTest completed.") 