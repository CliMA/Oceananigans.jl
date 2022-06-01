using Oceananigans
using Oceananigans.Architectures: child_architecture, device
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver, constructors, arch_sparse_matrix, matrix_from_coefficients, PreconditionedConjugateGradientSolver
using Oceananigans.Architectures: architecture, arch_array
using Statistics: mean
using IterativeSolvers
using Statistics: mean
using AlgebraicMultigrid
using GLMakie
using AlgebraicMultigrid: _solve!

import Oceananigans.Solvers: precondition!

N = 24

grid = RectilinearGrid(size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)
Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 #exp(-x^2 - y^2)
set!(r, r₀)
r .-= mean(r)
fill_halo_regions!(r, grid.architecture)

# Solve ∇²φ = r with `FFTBasedPoissonSolver`
φ_fft = CenterField(grid)

fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(r)

@info "Solving the Poisson equation with an FFT-based solver..."
@time solve!(φ_fft, fft_solver, fft_solver.storage)

fill_halo_regions!(φ_fft)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver`

@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end

function compute_∇²!(∇²φ, φ, arch, grid)
    fill_halo_regions!(φ)
    child_arch = child_architecture(arch)
    event = launch!(child_arch, grid, :xyz, ∇²!, ∇²φ, grid, φ, dependencies=Event(device(child_arch)))
    wait(device(child_arch), event)
    fill_halo_regions!(∇²φ)

    return nothing
end

φ_cg = CenterField(grid)
cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)))

@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver..."
@time solve!(φ_cg, cg_solver, r, arch, grid)

fill_halo_regions!(φ_cg)


# Solve ∇²φ = r with `HeptadiagonalIterativeSolver`

Nx, Ny, Nz = size(grid)
C = zeros(Nx, Ny, Nz)
D = zeros(Nx, Ny, Nz)

Ax = [1 / Δxᶠᶜᵃ(i, j, k, grid)^2 for i=1:Nx, j=1:Ny, k=1:Nz]
Ay = [1 / Δyᶜᶠᵃ(i, j, k, grid)^2 for i=1:Nx, j=1:Ny, k=1:Nz]
Az = [1 / Δzᵃᵃᶠ(i, j, k, grid)^2 for i=1:Nx, j=1:Ny, k=1:Nz]

hd_solver = HeptadiagonalIterativeSolver((Ax, Ay, Az, C, D); grid)

arch = architecture(grid)

solution = arch_array(arch, zeros(Nx * Ny * Nz))
solution .= interior(r)[:]
r_hd = arch_array(arch, interior(r)[:])

@info "Solving the Poisson equation with a heptadiagonal iterative solver..."
@time solve!(solution, hd_solver, r_hd, 1.0)

φ_hd = CenterField(grid)
interior(φ_hd) .= reshape(solution, Nx, Ny, Nz)
fill_halo_regions!(φ_hd)

# Create matrix
matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, grid, (Ax, Ay, Az, C, D), (false, false, false))  


# Solve ∇²φ = r with `AlgebraicMultigrid` solver
A = arch_sparse_matrix(arch, matrix_constructors)
r_array = collect(reshape(interior(r), Nx * Ny * Nz))

@info "Solving the Poisson equation with the Algebraic Multigrid iterative solver..."
ml = ruge_stuben(A, maxiter=100)
n = length(ml) == 1 ? size(ml.final_A, 1) : size(ml.levels[1].A, 1)
V = promote_type(eltype(ml.workspace), eltype(r_array))
φ_mg_array = zeros(V, size(r_array))
@show @allocated _solve!(φ_mg_array, ml, r_array)
# @info "Solving the Poisson equation with the Algebraic Multigrid iterative solver (not in-place)..."
# @time φ_mg_array2 = solve(A, r_array, RugeStubenAMG(), maxiter=100)

φ_mg = CenterField(grid)
interior(φ_mg) .= reshape(φ_mg_array, Nx, Ny, Nz)
# interior(φ_mg) .-= mean(interior(φ_mg))
fill_halo_regions!(φ_mg)


# Solve ∇²φ = r with `PreconditionedConjugateGradientSolver` solver using the AlgebraicMultigrid as preconditioner

struct MultigridPreconditioner{M, A}
    matrix_operator :: M
    maxiter :: Int
    amg_algorithm :: A
end

mgp = MultigridPreconditioner(A, 10, RugeStubenAMG())

"""
    precondition!(z, mgp::MultigridPreconditioner, r, args...)

Return `z` (Field)
"""
function precondition!(z, mgp::MultigridPreconditioner, r, args...)
    Nx, Ny, Nz = r.grid.Nx, r.grid.Ny, r.grid.Nz
    
    r_array = collect(reshape(interior(r), Nx * Ny * Nz))

    z_array = solve(mgp.matrix_operator, r_array, mgp.amg_algorithm, maxiter=mgp.maxiter)

    interior(z) .= reshape(z_array, Nx, Ny, Nz)
    fill_halo_regions!(z)

    return z
end



φ_cgmg = CenterField(grid)
cgmg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=r, reltol=eps(eltype(grid)), preconditioner = mgp)

@info "Solving the Poisson equation with a conjugate gradient preconditioned iterative solver WITH algebraic multigrid as preconditioner..."
@time solve!(φ_cgmg, cgmg_solver, r, arch, grid)

fill_halo_regions!(φ_cgmg)

∇²φ_fft = CenterField(grid)
∇²φ_cg = CenterField(grid)
∇²φ_mg = CenterField(grid)
∇²φ_cgmg = CenterField(grid)

compute_∇²!(∇²φ_fft, φ_fft, arch, grid)
compute_∇²!(∇²φ_cg, φ_cg, arch, grid)
compute_∇²!(∇²φ_mg, φ_mg, arch, grid)
compute_∇²!(∇²φ_cgmg, φ_cgmg, arch, grid)

# Plot results
fig = Figure(resolution=(1600, 1200))

ax_r = Axis(fig[1, 3], aspect=1, title="RHS")

ax_φ_fft = Axis(fig[2, 1], aspect=1, title="FFT-based solution")
ax_φ_cg = Axis(fig[2, 3], aspect=1, title="PreconditionedCG solution")
ax_φ_mg = Axis(fig[2, 5], aspect=1, title="Multigrid solution")
ax_φ_cgmg = Axis(fig[2, 7], aspect=1, title="PreconditionedCG MG solution")

ax_∇²φ_fft = Axis(fig[3, 1], aspect=1, title="FFT-based ∇²φ")
ax_∇²φ_cg = Axis(fig[3, 3], aspect=1, title="PreconditionedCG ∇²φ")
ax_∇²φ_mg = Axis(fig[3, 5], aspect=1, title="Multigrid ∇²φ")
ax_∇²φ_cgmg = Axis(fig[3, 7], aspect=1, title="PreconditionedCG MG ∇²φ")

hm_r = heatmap!(ax_r, interior(r, :, :, 1))
Colorbar(fig[1, 4], hm_r)

hm_fft = heatmap!(ax_φ_fft, interior(φ_fft, :, :, 1))
Colorbar(fig[2, 2], hm_fft)
hm_cg = heatmap!(ax_φ_cg, interior(φ_cg, :, :, 1))
Colorbar(fig[2, 4], hm_cg)
hm_mg = heatmap!(ax_φ_mg, interior(φ_mg, :, :, 1))
Colorbar(fig[2, 6], hm_mg)
hm_cgmg = heatmap!(ax_φ_cgmg, interior(φ_cgmg, :, :, 1))
Colorbar(fig[2, 8], hm_cgmg)

hm_∇²fft = heatmap!(ax_∇²φ_fft, interior(∇²φ_fft, :, :, 1))
Colorbar(fig[3, 2], hm_∇²fft)
hm_∇²cg = heatmap!(ax_∇²φ_cg, interior(∇²φ_cg, :, :, 1))
Colorbar(fig[3, 4], hm_∇²cg)
hm_∇²mg = heatmap!(ax_∇²φ_mg, reshape(∇²φ_mg, (Nx, Ny)))
Colorbar(fig[3, 6], hm_∇²mg)
hm_∇²cgmg = heatmap!(ax_∇²φ_cgmg, reshape(∇²φ_cgmg, (Nx, Ny)))
Colorbar(fig[3, 8], hm_∇²cgmg)

display(fig)
