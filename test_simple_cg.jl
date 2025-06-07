#!/usr/bin/env julia

# Simple test to verify CG solver functionality
using Pkg
Pkg.activate(".")

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Solvers
using Oceananigans.Fields
using LinearAlgebra: norm

println("Testing ConjugateGradientPoissonSolver basic functionality...")

# Create a simple grid
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))

# Create a CG solver with tighter tolerance for better testing
solver = ConjugateGradientPoissonSolver(grid, maxiter=100, reltol=1e-10, abstol=1e-12)

# Create pressure field and RHS
pressure = CenterField(grid)
rhs = solver.right_hand_side

# Set up a simple test problem
fill!(pressure, 0.0)
fill!(rhs, 0.0)

# Set a simple source in the center
rhs[2, 2, 2] = 1.0

println("\nTest 1: Solving from zero initial guess...")
pressure_zero = deepcopy(pressure)
solve!(pressure_zero, solver.conjugate_gradient_solver, rhs)
println("Solution from zero: pressure[2,2,2] = ", pressure_zero[2,2,2])
println("Iterations: ", iteration(solver.conjugate_gradient_solver))
println("Solution norm: ", norm(Array(interior(pressure_zero))))

# Reset solver for next test
solver2 = ConjugateGradientPoissonSolver(grid, maxiter=100, reltol=1e-10, abstol=1e-12)

println("\nTest 2: Solving with non-zero initial guess...")
# Set initial guess close to expected solution
fill!(pressure, 0.0)
pressure[2, 2, 2] = 0.1  # Some reasonable initial guess

pressure_nonzero = deepcopy(pressure)
println("Initial guess: pressure[2,2,2] = ", pressure_nonzero[2,2,2])

solve!(pressure_nonzero, solver2.conjugate_gradient_solver, rhs)
println("Solution from nonzero: pressure[2,2,2] = ", pressure_nonzero[2,2,2])
println("Iterations: ", iteration(solver2.conjugate_gradient_solver))
println("Solution norm: ", norm(Array(interior(pressure_nonzero))))

# Compare convergence
println("\nComparison:")
println("Iterations with zero initial guess: ", iteration(solver.conjugate_gradient_solver))
println("Iterations with nonzero initial guess: ", iteration(solver2.conjugate_gradient_solver))

# If the solver is using the initial guess properly, the nonzero initial guess 
# might converge faster (if it's closer to the solution) or at least behave differently
if iteration(solver2.conjugate_gradient_solver) <= iteration(solver.conjugate_gradient_solver)
    println("✓ Good sign: Non-zero initial guess converged in same or fewer iterations")
else
    println("? Non-zero initial guess took more iterations")
end

# Check if solutions are the same (they should be, within tolerance)
solution_diff = norm(Array(interior(pressure_zero)) - Array(interior(pressure_nonzero)))
println("Difference between final solutions: ", solution_diff)

if solution_diff < 1e-8
    println("✓ Solutions are essentially identical (as expected)")
else
    println("⚠ Solutions differ significantly (unexpected)")
end

println("\nTest completed.") 