using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: interior_copy
using Oceananigans.Operators: Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δyᶠᶜᵃ, Δxᶜᶠᵃ,∇²ᶜᶜᶜ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ImplicitFreeSurface
using Statistics
using Oceananigans.ImmersedBoundaries: GridFittedBoundary, ImmersedBoundaryGrid, is_immersed
using IterativeSolvers
using LinearAlgebra, SparseArrays, IncompleteLU
using Oceananigans.Solvers: constructors, MatrixIterativeSolver, update_diag!, arch_sparse_matrix

function compute_∇²(ϕ, arch, grid)
    ∇²ϕ = similar(ϕ)
    fill_halo_regions!(ϕ, arch)
    for i = 1:grid.Nx, j = 1:grid.Ny
        ∇²ϕ[i, j, 1] = ∇²ᶜᶜᶜ(i, j, 1, grid, ϕ)
    end

    return interior_copy(ϕ)[:,:,1]
end

function explicit_rhs(T, So, Ax, Ay)
    ΔT = similar(T)
    for i=2:Nx-1, j=2:Ny-1
        ΔT[i, j] = Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) + 
                   Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] 
    end
    for i=2:Nx-1
        ΔT[i, 1] = Ax[i+1, 1] * (T[i+1, 1] - T[i, 1]) + Ax[i, 1] * (T[i, 1] - T[i-1, 1]) + 
                   Ax[i+1, 1] * (T[i+1, 1] - T[i, 1]) + Ax[i, 1] * (T[i, 1] - T[i-1, 1]) +
                   Ay[i,   2] * (T[i,   2] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) + 
                   Ay[i,   2] * (T[i,   2] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) +
                   So[i, 1] 
        ΔT[i, Ny] = Ax[i+1, Ny] * (T[i+1, Ny] - T[i, Ny]) + Ax[i, Ny] * (T[i, Ny] - T[i-1, Ny]) +  
                    Ax[i+1, Ny] * (T[i+1, Ny] - T[i, Ny]) + Ax[i, Ny] * (T[i, Ny] - T[i-1, Ny]) + 
                    Ay[i, 1]   * (T[i, 1] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) + 
                    Ay[i, 1]   * (T[i, 1] - T[i, 1]) + Ay[i, 1] * (T[i, 1] - T[i, Ny]) +
                    So[i, Ny]
    end
    for j=2:Ny-1
        i = 1
        ΔT[i, j] = Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[Nx, j]) + 
                   Ax[i+1, j] * (T[i+1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[Nx, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] 
        i = Nx
        ΔT[i, j] = Ax[1, j] * (T[1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) + 
                   Ax[1, j] * (T[1, j] - T[i, j]) + Ax[i, j] * (T[i, j] - T[i-1, j]) +
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) + 
                   Ay[i, j+1] * (T[i, j+1] - T[i, j]) + Ay[i, j] * (T[i, j] - T[i, j-1]) +
                   So[i, j] 
    end
    
    return ΔT
end

function time_step_implicit_solver!(T, solver, Δt, source)
    rhs = - T ./ Δt - source[:] 
    cg!(T, solver.matrix, rhs; reltol=solver.tolerance, maxiter=solver.maximum_iterations, Pl=solver.preconditioner)
end    

function time_step_explicit_solver!(T, Ax, Ay, Δt, source)
    rhs = explicit_rhs(T, source, Ax, Ay)
    T = T .+ Δt .* rhs
    return T
end    

Lh = 10

Nx, Ny = (100, 100)

grid = RectilinearGrid(size = (Nx, Ny),
                          x = (0, Lh), y = (0, Lh),
                   topology = (Bounded, Bounded, Flat))

mask(x, y, z) = (x < 5.5 && x > 4.5) && (y < 5.5 && y > 4.4) ? 0 : 1

immersed_grid = GridFittedBoundary(mask)

x = grid.xᶜᵃᵃ[1:grid.Nx]
y = grid.yᵃᶜᵃ[1:grid.Ny]

Diff =  1.0

Ax   =  zeros(Nx, Ny)
Ay   =  zeros(Nx, Ny)
C    =  zeros(Nx, Ny)
# C    = - ones(Nx, Ny)

for i =1:grid.Nx, j = 1:grid.Ny
    # if is_immersed(i, j, grid.Nz, grid, immersed_grid) == true
    #    if is_immersed(i+1, j, grid.Nz, grid, immersed_grid) == true
        Ax[i, j] = Diff * Δyᶠᶜᵃ(i, j, 1, grid) / Δxᶠᶜᵃ(i, j, 1, grid)
    #    end
    #    if is_immersed(i, j+1, grid.Nz, grid, immersed_grid) == true
        Ay[i, j] = Diff * Δxᶜᶠᵃ(i, j, 1, grid) / Δyᶜᶠᵃ(i, j, 1, grid)
    #    end
    # end
end

Δt = 1.0

solver = MatrixIterativeSolver((Ax, Ay, C), grid = grid, precondition=false)
constr = deepcopy(solver.matrix_constructors)
update_diag!(constr, solver.architecture, solver.grid, solver.diagonal, Δt)
solver.matrix    = arch_sparse_matrix(solver.architecture, constr) 
# solver.preconditioner = Identity() # ilu(solver.matrix, τ = 1.0)

T_source   = zeros(Nx, Ny) 
T_implicit = zeros(Nx* Ny) 
T_explicit = zeros(Nx, Ny) 
T_source[10:11, 10:11]  .=  2.0
T_source[2:4, 2:4]      .= -1.0
T_source[17:19, 17:19]  .= -1.0


time_step_implicit_solver!(T_implicit, solver, Δt, T_source)
time_step_explicit_solver!(T_explicit, Ax, Ay, Δt, T_source)

T_implicit = reshape(T_implicit, Nx, Ny)
R = CenterField(grid)
set!(R, T_implicit)

∇²T = compute_∇²(R, CPU(), grid)


