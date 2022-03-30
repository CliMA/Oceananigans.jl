using Oceananigans
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!, HeptadiagonalIterativeSolver
using GLMakie

grid = RectilinearGrid(size=(128, 128), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))
# periodic_grid = RectilinearGrid(size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))

r = CenterField(grid)
ϕ = CenterField(grid)

r₀(x, y, z) = exp(-x^2 - y^2)
set!(r, r₀)
fill_halo_regions!(r, grid.architecture)

# Solve ∇²ϕ = r with `FFTBasedPoissonSolver`
fft_solver = FFTBasedPoissonSolver(grid)
fft_solver.storage .= interior(r)
@time solve!(ϕ, fft_solver, fft_solver.storage)

# Solve ∇²ϕ = r with `HeptadiagonalIterativeSolver`
Nx, Ny, Nz = size(grid)
C = zeros(Nx, Ny, Nz)
D = zeros(Nx, Ny, Nz)
Ax = [Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Ay = [Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Az = [Δxᶜᶜᵃ(i, j, k, grid) * Δyᶜᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]

solver = HeptadiagonalIterativeSolver((Ax, Ay, Az, C, D); grid, preconditioner_method = nothing)


#=
fig = Figure(resolution=(1200, 600))
ax_r = Axis(fig[1, 1], title="RHS")
ax_ϕ = Axis(fig[1, 2], title="Solution")
heatmap!(ax_r, interior(r, :, :, 1))
heatmap!(ax_ϕ, interior(ϕ, :, :, 1))

display(fig)
=#
