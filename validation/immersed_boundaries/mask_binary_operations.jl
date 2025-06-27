using Oceananigans, CairoMakie
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

underlying_grid = RectilinearGrid(architecture; size = (32, 32, 32), x = (-20, 20), y = (-20, 20), z = (-20, 0))

bathy(x, y) = 20 * exp(-(x^2 + y^2) / (2*10^2)) - 20

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathy))

c1 = CenterField(grid)
c2 = CenterField(grid)

Σc = c1 + c2
Πc = c1 * c2
ratio_c = c1 / c2
pow_c = c1 ^ c2

# Σc = 5, Πc = 6, ratio_c = 2/3, pow_c = 8

fig = Figure()

axs = [Axis(fig[1, 1], title = "+"),
       Axis(fig[1, 2], title = "*"),
       Axis(fig[2, 1], title = "/"),
       Axis(fig[2, 2], title = "^")]

xc, yc, zc = nodes(grid, Center(), Center(), Center())

cos = []

for (n, f) in enumerate([Σc, Πc, ratio_c, pow_c])
    c1 isa Number || set!(c1, 2)
    c2 isa Number || set!(c2, 3)

    mask_immersed_field!(f)

    @info f[1, 1, 16], f[16, 16, 16]

    heatmap!(axs[n], xc, yc, [f[i, j, 16] for i in 1:grid.Nx, j in 1:grid.Ny], colorrange = (0, 8))
end

fig

