using Oceananigans.Solvers: arch_sparse_matrix
using Oceananigans.Units 
using Oceananigans.Fields: interior_copy
using LinearAlgebra, SparseArrays

@inline gaussian(x, L) = @. exp( - (x / L )^2 )

Nx = 64; Ny = 3;
x  = (0, 1000kilometers)
z  = (-400meters, 0)

# Calculate matrix

N = Nx * Ny

grid = RectilinearGrid(size = (Nx, Ny, 1), x = x, y = (0, 1), z = z, topology = (Bounded, Periodic, Bounded))

fp = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient)
fm = ImplicitFreeSurface(solver_method = :MatrixIterativeSolver) 

g  = fp.gravitational_acceleration
Δt = 2 * grid.Δxᶜᵃᵃ / sqrt(g * grid.Lz)

coriolis = FPlane(f=1e-4)

mp = HydrostaticFreeSurfaceModel(grid = grid, coriolis = coriolis, free_surface = fp)
fp = mp.free_surface

mm = HydrostaticFreeSurfaceModel(grid = grid, coriolis = coriolis, free_surface = fm)
fm = mm.free_surface

constr = fm.implicit_step_solver.matrix_iterative_solver.matrix_constructors

U = 0.1 
L = grid.Lx / 40 
x₀ = grid.Lx / 2 
vᵍ(x, y, z) = - U * (x - x₀) / L * gaussian(x - x₀, L)

η₀ = coriolis.f * U * L / g 
ηᵍ(x) = η₀ * gaussian(x - x₀, L)
ηⁱ(x, y) = 2 * ηᵍ(x)
set!(mm, v=vᵍ, η=ηⁱ)
set!(mp, v=vᵍ, η=ηⁱ)

for i = 1:10
    time_step!(mm, Δt)
    time_step!(mp, Δt)
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

## reconstructing the matrix

# (all Ax, Ay and Δ are the same)

ax = Ax[1,1] / grid.Δxᶠᵃᵃ
ay = Ay[1,1] / grid.Δyᵃᶠᵃ
c  = - grid.Δyᵃᶜᵃ * grid.Δxᶜᵃᵃ / (g * Δt^2)

N = Nx * Ny

# now we start with our pentadiagonal

mcorr2 = diagm(0=>(c - 2ax - 2ay).*ones(N), 1=>ax.*ones(N-1), -1=>ax.*ones(N-1), Nx=>ay.*ones(N-Nx), -Nx=>ay.*ones(N-Nx))

# now it is zero flux in x so we cancel all x terms near a boundary and remove the associated diagonal term

for j in 1:Ny
    tₘ = 1  + (j-1) * Nx
    tₚ = Nx + (j-1) * Nx
    
    mcorr2[tₘ, tₘ] += ax
    mcorr2[tₚ, tₚ] += ax


    if tₘ > 1
         mcorr2[tₘ, tₘ-1] = 0
    end
    if tₚ < Ny*Nx
        mcorr2[tₚ, tₚ+1] = 0
    end
    @show tₘ, tₚ
end

# it is periodic in j so the diagonal should stay the same and we add terms on the off - diagonals

for i in 1:Nx
    tₘ = i + (1 -1) * Nx
    tₚ = i + (Ny-1) * Nx

    mcorr[tₚ, tₚ] -= ay
    mcorr2[tₘ, tₘ + Ny * Nx - Nx] = ay
    @show tₘ, tₚ
end

# lets see how it goes!

mm.free_surface.implicit_step_solver.matrix_iterative_solver.matrix = sparse(mcorr)

set!(mp, v=vᵍ, η=ηⁱ)
set!(mm, v=vᵍ, η=ηⁱ)

for i = 1:100
    time_step!(mm, Δt)
    time_step!(mp, Δt)
end

