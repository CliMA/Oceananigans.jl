# Fields basics

`Field`s and its relatives are core Oceananigans data structures.
`Field`s are more or less arrays of `data` located on a `grid`, whose entries
correspond to the average value of some quantity over some finite-sized volume.
`Field`s also may contain `boundary_conditions`, may be computed from an `operand` 
or expression involving other fields, and may cover only a portion of the total
`indices` spanned by the grid.

## Staggered grids and field locations

Oceananigans ocean-flavored fluids simulations rely fundamentally on
"staggered grid" numerical methods.

[Recall](@ref) that grids represent a physical domain divided into finite volumes.
For example, let's consider a horizontally-periodic, vertically-bounded grid of cells
that divide up a ``1 \times 1 \times 1`` cube:

```jldoctest fields
using Oceananigans

grid = RectilinearGrid(topology = (Periodic, Periodic, Bounded),
                       size = (4, 4, 4),
                       halo = (1, 1, 1),
                       x = (0, 1),
                       y = (0, 1),
                       z = [0, 0.1, 0.3, 0.6, 1])

# output
4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── Periodic x ∈ [0.0, 1.0) regularly spaced with Δx=0.25
├── Periodic y ∈ [0.0, 1.0) regularly spaced with Δy=0.25
└── Bounded  z ∈ [0.0, 1.0] variably spaced with min(Δz)=0.1, max(Δz)=0.4
```

The cubic domain is divided into a "primary mesh" of ``4 \times 4 \times 4 \times = 64`` cells,
which are evenly distributed in ``x, y`` but variably-spaced ``z``.
Now, in addition to the primary mesh, we also define a set of "staggered" grids whose cells are
shifted by half a cell width relative to the primary mesh.
In other words, the staggered grid cells have a "location" in each direction --- either `Center`,
and therefore co-located with the primary mesh, or `Face` and located over the interfaces of the
primary mesh.
For example, the primary or `Center` cell spacings in ``z`` are

```jldoctest fields
zspacings(grid, Center())

# output
4-element view(OffsetArray(::Vector{Float64}, -2:7), 1:4) with eltype Float64:
 0.1
 0.19999999999999998
 0.3
 0.4
```

corresponding to cell interfaces located at `z = [0, 0.1, 0.3, 0.6, 1]`.
But then for the grid which is staggered in `z` relative to the primary mesh,

```jldoctest fields
zspacings(grid, Face())

# output
5-element view(OffsetArray(::Vector{Float64}, -3:7), 1:5) with eltype Float64:
 0.1
 0.15000000000000002
 0.24999999999999994
 0.3500000000000001
 0.3999999999999999
```

The cells for the vertically staggered grid have different spacings than the primary mesh.
That's because the _edges_ of the vertically-staggered mesh coincide with the _nodes_ (the cell centers)
of the primary mesh. The nodes of the primary mesh are

```jldoctest fields
znodes(grid, Center(), with_halos=true)

# output
6-element OffsetArray(::Vector{Float64}, 0:5) with eltype Float64 with indices 0:5:
 -0.05
  0.05
  0.2
  0.44999999999999996
  0.8
  1.2
```

The center of the leftmost "halo cell" is `z = -0.05`, while the center of the first cell from the left is `z = 0.05`.
This means that the width of the first cell on the vertically-staggered grid is `0.05 + 0.05 = 0.1` --- and so on.
Finally, note that the nodes of the staggered mesh coincide with the cell interfaces of the primary mesh, so:


```jldoctest fields
znodes(grid, Center())

# output
5-element view(OffsetArray(::Vector{Float64}, -2:8), 1:5) with eltype Float64:
 0.0
 0.1
 0.3
 0.6
 1.0
```

In a three-dimensional domain, there are ``2³ = 8`` meshes -- 1 primary mesh, and 7 meshes that are
staggered to varying degrees from the primary mesh.
This system of staggered grids is commonly used in fluid dynamics and was [invented specifically for
simulations of the atmosphere and ocean](https://en.wikipedia.org/wiki/Arakawa_grids).

### Constructing Fields at specified locations

Every `Field` is associated with either the primary mesh or one of the staggered meshes by
it's three-dimensional "location".
To build a fully-centered `Field`, for example, we write

```jldoctest fields
c = Field{Center, Center, Center}(grid)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 6×6×6 OffsetArray(::Array{Float64, 3}, 0:5, 0:5, 0:5) with eltype Float64 with indices 0:5×0:5×0:5
    └── max=0.0, min=0.0, mean=0.0
```

Fully-centered fields also go by the alias `CenterField`,

```jldoctest fields
c == CenterField(grid)

# output
true
```

Many fluid dynamical variables are located at cell centers --- for example, tracers like temperature and salinity.
Another common type of `Field` we encounter have cells located over the `x`-interfaces of the primary grid,

```jldoctest fields
u = Field{Face, Center, Center}(grid)

# output
4×4×4 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 6×6×6 OffsetArray(::Array{Float64, 3}, 0:5, 0:5, 0:5) with eltype Float64 with indices 0:5×0:5×0:5
    └── max=0.0, min=0.0, mean=0.0
```

which also goes by `u = XFaceField(grid)`.
The name `u` is suggestive: in the Arakawa type-C grid ('C-grid' for short) used by Oceananigans,
the `x`-component of the velocity field is stored at `Face, Center, Center`.

The centers of the `u` cells are shifted to the left relative to the `c` cells:

```jldoctest fields
@show xnodes(c)
@show xnodes(u)
nothing # hide

# output
xnodes(c) = [0.125, 0.375, 0.625, 0.875]
xnodes(u) = [0.0, 0.25, 0.5, 0.75]
```

Notice that the first `u`-node is at `x=0`, the left end of the grid, but the last `u`-node is at `x=0.75`.
Because the `x`-direction is `Periodic`, the `XFaceField` `u` has 4 cells in `x` --- the cell just right of `x=0.75`
is the same as the cell at `x=0`.

Because the vertical direction is `Bounded`, however, vertically-staggered fields have more vertical cells
than `CenterField`s:

```jldoctest fields
w = Field{Center, Center, Face}(grid)

@show znodes(c)
@show znodes(w)
nothing # hide

# output
znodes(c) = [0.05, 0.2, 0.44999999999999996, 0.8]
znodes(w) = [0.0, 0.1, 0.3, 0.6, 1.0]
```

`Field`s at `Center, Center, Face` are also called `ZFaceField`,
and the vertical velocity is a `ZFaceField` on the C-grid.
Let's visualize the situation:

```
using CairoMakie
CairoMakie.activate!(type = "svg") # hide

fig = Figure(size=(400, 120))
ax = Axis(fig[1, 1], xlabel="x")

# Visualize the domain
lines!(ax, [0, 1], [0, 0], color=:gray)

xc = xnodes(c)
xu = xnodes(u)

scatter!(ax, xc, 0 * xc, marker=:circle, markersize=10, label="Cell centers")
scatter!(ax, xu, 0 * xu, marker=:vline, markersize=20, label="Cell interfaces")

ylims!(ax, -1, 1)
xlims!(ax, -0.1, 1.1)
hideydecorations!(ax)
hidexdecorations!(ax, ticklabels=false, label=false)
hidespines!(ax)

Legend(fig[0, 1], ax, nbanks=2, framevisible=false)

current_figure() # hide
```

## Setting `Field`s

When `Field`s are created, they are full of 0's.
For example,

```jldoctest setting
using Oceananigans

grid = RectilinearGrid(size = (4, 4, 4),
                       topology = (Periodic, Periodic, Bounded),
                       x = (0, 4),
                       y = (0, 4),
                       z = (0, 4))

c = CenterField(grid)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=0.0, min=0.0, mean=0.0
```

Not very exciting. Fortunately we can improve the situation by using [`set!`](@ref) to change the values of a field.
For example,

```jldoctest setting
set!(c, 42)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=42.0, min=42.0, mean=42.0
```

Now `c` is filled with `42`s. We can confirm this by inspecting individual values of `c`,

```jldoctest setting
c[1, 1, 1]

# output
42.0
```

or a range of values

```jldoctest setting
c[1:4, 1:4, 1]

# output
4×4 Matrix{Float64}:
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
```

Note that when we index into a `Field`, we are simpliy indexing into the `Field`'s `data`: 

```jldoctest setting
c.data[1:4, 1:4, 1]
 
# output
4×4 Matrix{Float64}:
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
 42.0  42.0  42.0  42.0
```

We can also use arrays,

```@setup setting
using Random
Random.seed!(123)
```

```jldoctest setting
random_stuff = rand(size(c)...)
set!(c, random_stuff)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=0.991405, min=0.0303789, mean=0.520014
```

and even functions,

```jldoctest setting
fun_stuff(x, y, z) = 2x

set!(c, fun_stuff)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=7.0, min=1.0, mean=4.0
```

For `Field`s on three-dimensional grids, the functions must have arguments `x, y, z` for `RectilinearGrid`, or `λ, φ, z` for `LatitudeLongitudeGrid` and `OrthogonalSphericalShellGrid`.
But for `Field`s on one- and two-dimensional grids, only the non-`Flat` directions are included.
For example

```jldoctest setting
one_d_grid = RectilinearGrid(size=7, x=(0, 7), topology=(Periodic, Flat, Flat))
one_d_c = CenterField(one_d_grid)
more_fun_stuff(x) = 3x
set!(one_d_c, more_fun_stuff)

# output
7×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 7×1×1 RectilinearGrid{Float64, Periodic, Flat, Flat} on CPU with 3×0×0 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Nothing, north: Nothing, bottom: Nothing, top: Nothing, immersed: ZeroFlux
└── data: 13×1×1 OffsetArray(::Array{Float64, 3}, -2:10, 1:1, 1:1) with eltype Float64 with indices -2:10×1:1×1:1
    └── max=19.5, min=1.5, mean=10.5
```

Now, since

```jldoctest setting
xnodes(c)

# output
4-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, -2:7), 1:4) with eltype Float64:
 0.5
 1.5
 2.5
 3.5
```

we find that

```jldoctest setting
c[1:4, 1, 1]

# ouptut
4-element Vector{Float64}:
 1.0
 3.0
 5.0
 7.0
```

In other words, `fun_stuff` is evaluated using the coordinates at the center of each cell for a given `Field`.
One consequence is that the result is different for a `Field` located at `x` faces,

```jldoctest setting
u = XFaceField(grid)
set!(u, fun_stuff)
u[1:4, 1, 1]

# output
4-element Vector{Float64}:
 0.0
 2.0
 4.0
 6.0
```

## Halo regions and boundary conditions

```jldoctest setting
c[:, 1, 1]

# output
10-element OffsetArray(::Vector{Float64}, -2:7) with eltype Float64 with indices -2:7:
 0.0
 0.0
 0.0
 1.0
 3.0
 5.0
 7.0
 0.0
 0.0
 0.0
```
