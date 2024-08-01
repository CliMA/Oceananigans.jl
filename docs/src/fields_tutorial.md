# Fields and operations

`Field`s and its relatives are one of Oceananigans core data structures.

## "Staggered" grids and field locations

```@setup fields
using Oceananigans
using CairoMakie
```

```jldoctest fields
using Oceananigans
using CairoMakie
CairoMakie.activate!(type = "svg") # hide

grid = RectilinearGrid(size=3, x=(0, 3), topology=(Bounded, Flat, Flat))

u = Field{Face, Center, Center}(grid)
c = Field{Center, Center, Center}(grid)

xu = xnodes(u)
xc = xnodes(c)

fig = Figure(size=(800, 800))
ax = Axis(fig[1, 1])

scatter!(ax, xc, 0 * xc, color=:blue)
scatter!(ax, xu, 1 .+ 0 * xu, color=:red)

save("plot_staggered_nodes.svg", fig); nothing # hide
```

![]("plot_staggered_nodes.svg")

