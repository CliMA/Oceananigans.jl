using Oceananigans
using Oceananigans.Architectures: unified_array
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: solve!, HeptadiagonalIterativeSolver
using Oceananigans.MultiRegion
using Oceananigans.MultiRegion: UnifiedDiagonalIterativeSolver
using GLMakie

grid = RectilinearGrid(size=(1400, 1400), x=(-4, 4), y=(-4, 4), topology=(Bounded, Bounded, Flat))
mrg  = MultiRegionGrid(grid, partition = XPartition(2), devices=(0, 1))

rs    = CenterField(grid)
ϕ_ser = CenterField(grid)
ϕ_par = CenterField(mrg)

vol = volume(1, 1, 1, grid, Center(), Center(), Center())

r₀(x, y, z) = (x^2 + y^2) > 1 ? vol : 0 
set!(rs, r₀)

fill_halo_regions!(rs, grid.architecture)

# Solve ∇²ϕ_fft = r with `HeptadiagonalIterativeSolver`
Nx, Ny, Nz = size(grid)
C = zeros(Nx, Ny, Nz)
D = zeros(Nx, Ny, Nz)
Ax = [Δzᵃᵃᶜ(i, j, k, grid) * Δyᶠᶜᵃ(i, j, k, grid) / Δxᶠᶜᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Ay = [Δzᵃᵃᶜ(i, j, k, grid) * Δxᶜᶠᵃ(i, j, k, grid) / Δyᶜᶠᵃ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
Az = [Δxᶜᶜᵃ(i, j, k, grid) * Δyᶜᶜᵃ(i, j, k, grid) / Δzᵃᵃᶠ(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]

ss =   HeptadiagonalIterativeSolver((Ax, Ay, Az, C, D); grid, preconditioner_method = nothing, maximum_iterations = 100)
sp = UnifiedDiagonalIterativeSolver((Ax, Ay, Az, C, D); grid, mrg, maximum_iterations = 100)

rp = unified_array(GPU(), reshape(Array(interior(rs)), :))
x  = similar(rp)
@time solve!(ϕ_ser, ss, rs, 1.0)
@time solve!(x, sp, rp, 1.0)

fig = Figure(resolution=(1200, 600))
ax_r = Axis(fig[1, 1], title="RHS")
ax_ϕ_fft = Axis(fig[1, 2], title="FFT-based solution")
ax_ϕ_hd = Axis(fig[1, 3], title="Heptadiagonal Iterative solution")
heatmap!(ax_r, interior(r, :, :, 1))
heatmap!(ax_ϕ_fft, interior(ϕ_fft, :, :, 1))
heatmap!(ax_ϕ_hd, interior(ϕ_hd, :, :, 1))

display(fig)