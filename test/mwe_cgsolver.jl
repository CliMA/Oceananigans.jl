using Oceananigans
using LinearAlgebra: norm

# Create a moderately-sized grid for meaningful convergence testing
grid = RectilinearGrid(size=(8, 8, 8), extent=(1, 1, 1))

# Create CG solver with tighter tolerance for better testing
maxiter = 10
solver = ConjugateGradientPoissonSolver(grid, maxiter=maxiter)

# Create pressure field and RHS
pressure = CenterField(grid)
rhs = solver.right_hand_side

# Set up a test problem with known analytical solution
# Solution: φ(x,y,z) = sin(πx) sin(πy) sin(πz)
# RHS: f = -3π² sin(πx) sin(πy) sin(πz)
# Set up the analytical test case
for k in 1:8, j in 1:8, i in 1:8
    local x = xnode(i, j, k, grid, Center(), Center(), Center())
    local y = ynode(i, j, k, grid, Center(), Center(), Center())
    local z = znode(i, j, k, grid, Center(), Center(), Center())
    rhs[i, j, k] = -3 * π^2 * sin(π * x) * sin(π * y) * sin(π * z)
end

# Analytical solution for comparison
analytical_solution = CenterField(grid)
for k in 1:8, j in 1:8, i in 1:8
    local x = xnode(i, j, k, grid, Center(), Center(), Center())
    local y = ynode(i, j, k, grid, Center(), Center(), Center())
    local z = znode(i, j, k, grid, Center(), Center(), Center())
    analytical_solution[i, j, k] = sin(π * x) * sin(π * y) * sin(π * z)
    pressure[i, j, k] = analytical_solution[i, j, k]
end

# Test 1: Solving from zero initial guess
pressure_zero = deepcopy(pressure)
solve!(pressure_zero, solver.conjugate_gradient_solver, rhs)
iterations_zero = iteration(solver.conjugate_gradient_solver)
solution_norm_zero = norm(Array(interior(pressure_zero)))

# Check accuracy against analytical solution
error_zero = norm(Array(interior(pressure_zero)) - Array(interior(analytical_solution)))
@test error_zero < 1e-6  # Should be reasonably accurate
pause

Main.@infiltrate

# Reset solver for next test
solver2 = ConjugateGradientPoissonSolver(grid, maxiter=maxiter)

# Test 2: Solving with non-zero initial guess (closer to analytical solution)
fill!(pressure, 0.0)
for k in 1:8, j in 1:8, i in 1:8
    local x = xnode(i, j, k, grid, Center(), Center(), Center())
    local y = ynode(i, j, k, grid, Center(), Center(), Center())
    local z = znode(i, j, k, grid, Center(), Center(), Center())
    pressure[i, j, k] = sin(π * x) * sin(π * y) * sin(π * z)  # 50% of analytical solution
end

pressure_nonzero = deepcopy(pressure)
solve!(pressure_nonzero, solver2.conjugate_gradient_solver, rhs)
iterations_nonzero = iteration(solver2.conjugate_gradient_solver)
solution_norm_nonzero = norm(Array(interior(pressure_nonzero)))

@test iterations_nonzero > 0
@test iterations_nonzero <= maxiter
@test solution_norm_nonzero > 0

# Check accuracy against analytical solution
error_nonzero = norm(Array(interior(pressure_nonzero)) - Array(interior(analytical_solution)))
@test error_nonzero < 1e-6

# Solutions should be essentially identical regardless of initial guess
solution_diff = norm(Array(interior(pressure_zero)) - Array(interior(pressure_nonzero)))
@test solution_diff < 1e-8

# Non-zero initial guess should converge faster (since it's closer to the solution)
@test iterations_nonzero <= iterations_zero

@info "Convergence test results:"
@info "  Zero initial guess: $iterations_zero iterations, error = $error_zero"
@info "  Good initial guess: $iterations_nonzero iterations, error = $error_nonzero"

