using Oceananigans
using CairoMakie

grid = RectilinearGrid(size=(4, 4, 1), extent=(1, 1, 1))
c = CenterField(grid)

#heatmap(c)

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, c)
