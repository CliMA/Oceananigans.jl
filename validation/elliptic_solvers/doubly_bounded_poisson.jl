using Oceananigans
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver, constructors, arch_sparse_matrix, matrix_from_coefficients
using Oceananigans.Architectures: architecture, arch_array
using IterativeSolvers
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

Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

r = CenterField(grid)
ϕ_fft = CenterField(grid)

r₀(x, y, z) = (x^2 + y^2) < 1 ? 1 : 0 #exp(-x^2 - y^2)
set!(r, r₀)
fill_halo_regions!(r, grid.architecture)

# Solve ∇²ϕ = r with `FFTBasedPoissonSolver`
fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(r)
@info "Solving the Poisson equation with an FFT-based solver..."
@time solve!(ϕ_fft, fft_solver, fft_solver.storage)

# Solve ∇²ϕ = r with `HeptadiagonalIterativeSolver`
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

ϕ_hd = CenterField(grid)
interior(ϕ_hd) .= reshape(solution, Nx, Ny, Nz)

# Create matrix
matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, grid, (Ax, Ay, Az, C, D), (false, false, false))  

# Solve ∇²ϕ = r with `AlgebraicMultigrid`
A = arch_sparse_matrix(arch, matrix_constructors)
b = collect(reshape(interior(r), (Nx*Ny, )))

@info "Solving the Poisson equation with the Algebraic Multigrid iterative solver..."
@time ϕ_mg = solve(A, b, RugeStubenAMG(), maxiter=100, verbose=true)

fig = Figure(resolution=(1600, 1000))

ax_r = Axis(fig[1, 3], aspect=1, title="RHS")
ax_ϕ_fft = Axis(fig[2, 1], aspect=1, title="FFT-based solution")
ax_ϕ_hd = Axis(fig[2, 3], aspect=1, title="Heptadiagonal Iterative solution")
ax_ϕ_mg = Axis(fig[2, 5], aspect=1, title="Multigrid solution")

hm_r = heatmap!(ax_r, interior(r, :, :, 1))
Colorbar(fig[1, 4], hm_r)
hm_fft = heatmap!(ax_ϕ_fft, interior(ϕ_fft, :, :, 1))
Colorbar(fig[2, 2], hm_fft)
hm_hd = heatmap!(ax_ϕ_hd, interior(ϕ_hd, :, :, 1))
Colorbar(fig[2, 4], hm_hd)
hm_mg = heatmap!(ax_ϕ_mg, reshape(ϕ_mg, (Nx, Ny)))
Colorbar(fig[2, 6], hm_mg)

display(fig)

fft_int = reshape(interior(ϕ_fft, :, :, 1), Nx*Ny)
@show b[1]/(A*fft_int)[1]
