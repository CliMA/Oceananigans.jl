using Oceananigans.Solvers: arch_sparse_matrix

Nx = 64; Ny = 3;
x  = (0, 10)
z  = (-4, 0)
Δt = 10

# Calculate matrix

N = Nx * Ny

grid = RectilinearGrid(size = (Nx, Ny, 1), x = x, y = (0, 1), z = z, topology = (Periodic, Periodic, Bounded))


fp = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient)
fm = ImplicitFreeSurface(solver_method = :MatrixIterativeSolver) 

mp = HydrostaticFreeSurfaceModel(grid = grid, free_surface = fp)
fp = mp.free_surface

mm = HydrostaticFreeSurfaceModel(grid = grid, free_surface = fm)
fm = mm.free_surface

constr = fm.implicit_step_solver.matrix_iterative_solver.matrix_constructors

for i = 1:1000
    time_step!(mm, Δt)
end

η = similar(fp.η)
β = similar(fp.η)

parent(η) .= 0
η[2,2]     = 1 

func = fp.implicit_step_solver.preconditioned_conjugate_gradient_solver.linear_operation!
Ax   = fp.implicit_step_solver.vertically_integrated_lateral_areas.xᶠᶜᶜ
Ay   = fp.implicit_step_solver.vertically_integrated_lateral_areas.yᶜᶠᶜ
g    = fp.gravitational_acceleration

matr = fm.implicit_step_solver.matrix_iterative_solver.matrix

mcorr = zeros(N, N)

for i in 1:Nx, j in 1:Ny
    parent(η) .= 0
    η[i, j] = 1
    func(β, η, Ax, Ay, g, Δt)
    mcorr[:,i + (j-1) * Nx] = interior(β)[:]
end


