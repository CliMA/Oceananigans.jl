using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, PartialCellBottom
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Printf
using GLMakie

arch = CPU()

underlying_grid = RectilinearGrid(arch,
                                  size=(128, 64), halo=(3, 3), 
                                  y = (-1, 1),
                                  z = (-1, 0),
                                  topology=(Flat, Periodic, Bounded))

# A bump
h₀ = 0.5 # bump height
L = 0.25 # bump width
h(y) = h₀ * exp(- y^2 / L^2)
seamount(x, y) = - 1 + h(y)

seamount_field = Field{Center, Center, Nothing}(underlying_grid)
set!(seamount_field, seamount)
fill_halo_regions!(seamount_field)

ib = PartialCellBottom(seamount_field.data; minimum_fractional_Δz=0.2)
grid = ImmersedBoundaryGrid(underlying_grid, ib)

Nx, Ny, Nz = size(grid)
fractional_Δzᶜ = zeros(Ny, Nz)
fractional_Δzᶠ = zeros(Ny, Nz)
Δzʳ = underlying_grid.Δzᵃᵃᶜ
for j = 1:Ny, k=1:Nz
    Δzᶜ[j, k] = Δzᶜᶜᶜ(1, j, k, grid) / Δzʳ
    Δzᶠ[j, k] = Δzᶜᶜᶠ(1, j, k, grid) / Δzʳ
end

@show minimum(fractional_Δzᶜ)
@show minimum(fractional_Δzᶠ)

fig = Figure()
ax_c = Axis(fig[1, 1], title="Fractional Δzᶜᶜᶜ")
ax_f = Axis(fig[2, 1], title="Fractional Δzᶜᶜᶠ")
hmc = heatmap!(ax_c, fractional_Δzᶜ)
Colorbar(fig[1, 2], hmc)
hmf = heatmap!(ax_f, fractional_Δzᶠ)
Colorbar(fig[2, 2], hmf)
display(fig)

