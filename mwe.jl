using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
using Oceananigans.Models.NonhydrostaticModels: calculate_pressure_correction!

N = 2
x = y = (0, 1)
z = [0, 0.2, 1]
grid = RectilinearGrid(size=(N, N, N); x, y, z, halo=(2, 2, 2), topology=(Bounded, Periodic, Bounded))
fft_solver = fft_poisson_solver(grid)
pressure_solver = ConjugateGradientPoissonSolver(grid, preconditioner=fft_poisson_solver(grid))
model = NonhydrostaticModel(; grid, pressure_solver)
set!(model, u=1)

