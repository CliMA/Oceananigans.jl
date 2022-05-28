using Oceananigans
using Oceananigans.Architectures: child_architecture
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using KernelAbstractions: @kernel, @index, Event
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver, constructors, arch_sparse_matrix, matrix_from_coefficients, PreconditionedConjugateGradientSolver
using Oceananigans.Architectures: architecture, arch_array
using IterativeSolvers
using Statistics: mean
using AlgebraicMultigrid
using GLMakie

N = 16
# 6 = 5.0625
# 8 = 16
# 10 = 39
# 12 = 81
# Out by a factor of (N/4)^4
# 7 = 37.515625 = 7^4/4^3
# 16 = 85.3333 = 16^2/3
# 20 = 156.25 = 20^4/4^5
# 21 = 144.703125  = (21/4)^3
# 24 = 162  = (24/4)^3

grid = RectilinearGrid(size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

arch = architecture(grid)
Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

r = CenterField(grid)
r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 #exp(-x^2 - y^2)
set!(r, r₀)
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

fill_halo_regions!(ϕ_cg)


# Solve ∇²φ = r with `HeptadiagonalIterativeSolver`

Nx, Ny, Nz = size(grid)
C = zeros(Nx, Ny, Nz)
D = zeros(Nx, Ny, Nz)
Ax = [Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Ay = [Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Az = [Δxᶜᶜᵃ(i, j, k, grid) * Δyᶜᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]

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
r_array = collect(reshape(interior(r), (Nx * Ny, )))

@info "Solving the Poisson equation with the Algebraic Multigrid iterative solver..."
@time φ_mg_array = solve(A, r_array, RugeStubenAMG(), maxiter=100, verbose=true)

φ_mg = CenterField(grid)
interior(φ_mg) .= reshape(φ_mg_array, Nx, Ny, Nz)
interior(φ_mg) .-= mean(interior(φ_mg))
fill_halo_regions!(φ_mg)

∇²φ_fft = CenterField(grid)
∇²φ_cg = CenterField(grid)
∇²φ_mg = CenterField(grid)

compute_∇²!(∇²φ_fft, φ_fft, arch, grid)
compute_∇²!(∇²φ_cg, φ_cg, arch, grid)
compute_∇²!(∇²φ_mg, φ_mg, arch, grid)

# Plot results
fig = Figure(resolution=(1200, 1200))

ax_r = Axis(fig[1, 3], aspect=1, title="RHS")

ax_φ_fft = Axis(fig[2, 1], aspect=1, title="FFT-based solution")
ax_φ_cg = Axis(fig[2, 3], aspect=1, title="PreconditionedCG solution")
ax_φ_mg = Axis(fig[2, 5], aspect=1, title="Multigrid solution")

ax_∇²φ_fft = Axis(fig[3, 1], aspect=1, title="FFT-based ∇²φ")
ax_∇²φ_cg = Axis(fig[3, 3], aspect=1, title="PreconditionedCG ∇²φ")
ax_∇²φ_mg = Axis(fig[3, 5], aspect=1, title="Multigrid ∇²φ")

hm_r = heatmap!(ax_r, interior(r, :, :, 1))
Colorbar(fig[1, 4], hm_r)

hm_fft = heatmap!(ax_φ_fft, interior(φ_fft, :, :, 1))
Colorbar(fig[2, 2], hm_fft)

cg_lims = extrema(interior(φ_cg))
if abs(cg_lims[1]-cg_lims[2]) < 1e-13
    cg_lims = (cg_lims[1]-1e-13, cg_ligs[2]+1e-13)
end

if abs(cg_lims[1]-cg_lims[2]) > 1e13
    cg_lims = (-1e12, 1e12)
end

hm_cg = heatmap!(ax_φ_cg, interior(φ_cg, :, :, 1), colorrange = cg_lims)
Colorbar(fig[2, 4], hm_cg)
hm_mg = heatmap!(ax_φ_mg, interior(φ_mg, :, :, 1))
Colorbar(fig[2, 6], hm_mg)

hm_∇²fft = heatmap!(ax_∇²φ_fft, interior(∇²φ_fft, :, :, 1))
Colorbar(fig[3, 2], hm_∇²fft)
hm_∇²cg = heatmap!(ax_∇²φ_cg, interior(∇²φ_cg, :, :, 1))
Colorbar(fig[3, 4], hm_∇²cg)
hm_∇²mg = heatmap!(ax_∇²φ_mg, reshape(∇²φ_mg, (Nx, Ny)))
Colorbar(fig[3, 6], hm_∇²mg)

display(fig)

# fft_int = reshape(interior(φ_fft, :, :, 1), Nx*Ny)
# @show b[1]/(A*fft_int)[1]
