using Oceananigans
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver, constructors, arch_sparse_matrix, matrix_from_coefficients
using Oceananigans.Architectures: architecture
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

function setup_2D(n=N, T::Type=Float64)
    L = zeros(T, n+2, n+2, 2); L[3:n+1, 2:n+1, 1] .= 1; L[2:n+1, 3:n+1, 2] .= 1; 
    x = T[i-1 for i ∈ 1:n+2, j ∈ 1:n+2]
    Poisson(L), FieldVector(x)
end


grid = RectilinearGrid(size=(N, N), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))

r = CenterField(grid)
ϕ_fft = CenterField(grid)
ϕ_hd = CenterField(grid)

r₀(x, y, z) = (x^2 + y^2) > 1 ? 1 : 0 #exp(-x^2 - y^2)
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

hd_solver = HeptadiagonalIterativeSolver((Ax, Ay, Az, C, D); grid, preconditioner_method = nothing)
@info "Solving the Poisson equation with a heptadiagonal iterative solver..."
@time solve!(ϕ_hd, hd_solver, r, 1.0)

# Create matrix
arch = architecture(grid)
matrix_constructors, diagonal, problem_size = matrix_from_coefficients(arch, grid, (Ax, Ay, Az, C, D), (false, false, false))  

A = arch_sparse_matrix(arch, matrix_constructors)
b = collect(reshape(interior(r), (Nx*Ny, )))

x = solve(A, b, RugeStubenAMG())

fig = Figure(resolution=(1200, 600))
ax_r = Axis(fig[1, 1], title="RHS")
ax_ϕ_fft = Axis(fig[1, 2], title="FFT-based solution")
ax_ϕ_hd = Axis(fig[1, 3], title="Heptadiagonal Iterative solution")
ax_ϕ_mg = Axis(fig[1, 4], title="Multigrid solution")
heatmap!(ax_r, interior(r, :, :, 1))
heatmap!(ax_ϕ_fft, interior(ϕ_fft, :, :, 1))
heatmap!(ax_ϕ_hd, interior(ϕ_hd, :, :, 1))
heatmap!(ax_ϕ_mg, reshape(x, (Nx,Ny)))

display(fig)

fft_int = reshape(interior(ϕ_fft, :, :, 1), Nx*Ny)
@show b[1]/(A*fft_int)[1]