# Fields and operations

`Field`s and its relatives are one of Oceananigans core data structures.

## "Staggered" grids and field locations

```jldoctest fields
using Oceananigans
using CairoMakie
CairoMakie.activate!(type = "svg") # hide

grid = RectilinearGrid(size=3, x=(0, 3), topology=(Bounded, Flat, Flat))

u = Field{Face, Center, Center}(grid)
c = Field{Center, Center, Center}(grid)

xu = xnodes(u)
xc = xnodes(c)

fig = Figure(size=(400, 120))
ax = Axis(fig[1, 1], xlabel="x")

# Visualize the domain
lines!(ax, [0, 3], [0, 0], color=:gray)

scatter!(ax, xc, 0 * xc, marker=:circle, markersize=10, label="Cell centers")
scatter!(ax, xu, 0 * xu, marker=:vline, markersize=20, label="Cell interfaces")

ylims!(ax, -1, 1)
xlims!(ax, -0.1, 3.1)
hideydecorations!(ax)
hidexdecorations!(ax, ticklabels=false, label=false)
hidespines!(ax)

Legend(fig[0, 1], ax, nbanks=2, framevisible=false)

display(fig)
save("plot_staggered_nodes.svg", fig); nothing # hide
```

![]("plot_staggered_nodes.svg")

