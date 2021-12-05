using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators: Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δyᶠᶜᵃ, Δxᶜᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using IterativeSolvers
using LinearAlgebra, SparseArrays, IncompleteLU
using Oceananigans.Solvers: constructors, MatrixIterativeSolver, update_diag!, arch_sparse_matrix


function explicit_rhs(T, So, Si, Ax, Ay)
    ΔT = similar(T)
    for i=2:Nx-1, j=2:Ny-1
        ΔT[i, j] = Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) + 
                   Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] + Si[i, j]
    end
    for i=2:Nx-1
        ΔT[i, 1] = Ax[i+1, 1] * (T[i+1, 1] - T[i, 1]) + Ax[i, 1] * (T[i, 1] - T[i-1, 1]) + 
                   Ax[i+1, 1] * (T[i+1, 1] - T[i, 1]) + Ax[i, 1] * (T[i, 1] - T[i-1, 1]) +
                   Ay[i,   2] * (T[i,   2] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) + 
                   Ay[i,   2] * (T[i,   2] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) +
                   So[i, 1] + Si[i, 1]
        ΔT[i, Ny] = Ax[i+1, Ny] * (T[i+1, Ny] - T[i, Ny]) + Ax[i, Ny] * (T[i, Ny] - T[i-1, Ny]) +  
                    Ax[i+1, Ny] * (T[i+1, Ny] - T[i, Ny]) + Ax[i, Ny] * (T[i, Ny] - T[i-1, Ny]) + 
                    Ay[i, 1]   * (T[i, 1] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) + 
                    Ay[i, 1]   * (T[i, 1] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) +
                    So[i, Ny] + Si[i, Ny]
    end
    for j=2:Ny-1
        i = 1
        ΔT[i, j] = Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[Nx, j]) + 
                   Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[Nx, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] + Si[i, j]
        i = Nx
        ΔT[i, j] = Ax[1, j] * (T[1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) + 
                   Ax[1, j] * (T[1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] + Si[i, j]
    end

    return ΔT
end

function run_implicit_solver(solver, Δt, source, sink, tolerance)
    T = zeros(Nx * Ny)
    steady = 1.0
    while steady > tolerance
        T_old = deepcopy(T)

        rhs = - T[:] ./ Δt - source[:] - sink[:]
        T = solver.iterative_solver(solver.matrix, rhs; reltol=solver.tolerance, maxiter=solver.maximum_iterations, Pl=solver.preconditioner)

        steady = norm(T_old .- T) / norm(T_old)

        @show steady
    end
    return reshape(T, Nx, Ny)    
end    

function run_explicit_solver(Ax, Ay, Δt, source, sink, tolerance)
    T = zeros(Nx, Ny)
    steady = 1.0
    while steady > tolerance
        T_old = deepcopy(T)
        rhs = explicit_rhs(T, source, sink, Ax, Ay)
    
        T = T .+ Δt .* rhs
        steady = norm(T_old .- T) / norm(T_old)

        @show steady
    end
    return T    
end    

Lh = 10

Nx, Ny = (20, 20)

grid = RectilinearGrid(size = (Nx, Ny),
                          x = (0, Lh), y = (0, Lh),
                   topology = (Bounded, Bounded, Flat))

x = grid.xᶜᵃᵃ[1:grid.Nx]
y = grid.yᵃᶜᵃ[1:grid.Ny]

Diff =  40.0

Ax   =  zeros(Nx, Ny)
Ay   =  zeros(Nx, Ny)
C    = - ones(Nx, Ny)

for i =1:grid.Nx, j = 1:grid.Ny
    Ax[i, j] = Diff * Δyᶠᶜᵃ(i, j, 1, grid) / Δxᶠᶜᵃ(i, j, 1, grid)
    Ay[i, j] = Diff * Δxᶜᶠᵃ(i, j, 1, grid) / Δyᶜᶠᵃ(i, j, 1, grid)
end

Δt = 1

solver = MatrixIterativeSolver((Ax, Ay, C), grid = grid)
constr = deepcopy(solver.matrix_constructors)
update_diag!(constr, solver.architecture, solver.grid, solver.diagonal, Δt)
solver.matrix    = arch_sparse_matrix(solver.architecture, constr) 
solver.preconditioner = ilu(solver.matrix, τ = 1.0)

T_source   = zeros(Nx, Ny) 
T_sink     = zeros(Nx, Ny) 

T_source[8:12, 8:12]   .=  1.0
T_sink[2:4, 2:4]       .= -1.0
T_sink[16:18, 16:18]   .= -1.0


T_implicit = run_implicit_solver(solver, Δt, T_source, T_sink, 1e-4)
T_explicit = run_explicit_solver(Ax, Ay, Δt, T_source, T_sink, 1e-4)

T_implicit = reshape(T_implicit, Nx, Ny)