# Fields basics

`Field`s and its relatives are core Oceananigans data structures.
`Field`s are more or less arrays of `data` located on a `grid`, whose entries
correspond to the average value of some quantity over some finite-sized volume.
`Field`s also may contain `boundary_conditions`, may be computed from an `operand` 
or expression involving other fields, and may cover only a portion of the total
`indices` spanned by the grid.

## "Staggered" grids and field locations

In order to grasp the syntax for constructing and manipulating `Field`s,
we first have to introduce the concept of "staggered grids".
As elaborated on extensively in [Grids and architectures](@ref), a grid represents
a domain that is divided into finite volumes, with data stored on a specified architecture 
and at a specified floating point precision.

We refer to this mesh of finite volumes as the "primary" mesh.
In addition to the primary grid, we define additional auxiliary meshe s
whose cells are "staggered", or shifted by half a cell width relative to the primitive mesh.
In a three-dimensional domain, there are ``2³ = 8`` total meshes -- 1 primary mesh, and 7 auxiliary "staggered" meshes.
This system of staggered grids is commonly used in fluid dynamics and was [invented specifically for
simulations of the atmosphere and ocean](https://en.wikipedia.org/wiki/Arakawa_grids).

`Field`s therefore all have a "location", which actually refers to the underlying mesh.
We refer to `Field`s whos finite volumes coincide with the primary mesh as "centered" or "located at cell centers".
To build such `Fields`, we write

```jldoctest fields
using Oceananigans

grid = RectilinearGrid(size = (4, 4, 4),
                       topology = (Periodic, Periodic, Bounded),
                       x = (0, 1),
                       y = (0, 1),
                       z = (0, 1))

c = Field{Center, Center, Center}(grid)

# output
4×4×4 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 10×10×10 OffsetArray(::Array{Float64, 3}, -2:7, -2:7, -2:7) with eltype Float64 with indices -2:7×-2:7×-2:7
    └── max=0.0, min=0.0, mean=0.0
```

Fully-centered fields also go by the alias `CenterField`,

```jldoctest fields
c == CenterField(grid)

# output
true
```

`CenterField`s are some of the most common `Field`s -- in fluid dynamics simulations, for example, tracer quantities like heat and salt are located at cell centers.
Another common `Field` we encounter at those with finite volumes centered over the `x`-interfaces of the primary grid,

```jldoctest fields
u = Field{Face, Center, Center}(grid)

# output
```

which can also be constructed by writing `u = XFaceField(grid)`.
The name `u` is suggestive, because within the "C-grid" system used by Oceananigans,
the `x`-component of the velocity field is stored at `Face, Center, Center`.

Staggering means that the "position" of `CenterField` differs from `XFaceField`.
For example, the center of the primary cells have `x`-coordinates


```jldoctest fields
xnodes(c)

# output
4-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, -2:7), 1:4) with eltype Float64:
 0.125
 0.375
 0.625
 0.875
```

whereas the center of the `XFaceField` cells are located at

```jldoctest fields
xnodes(u)

# output
4-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, -2:7), 1:4) with eltype Float64:
 0.0
 0.25
 0.5
 0.75
```

Notice that the first node is at `x=0`, the left end of the grid.
Because the `x`-direction is `Periodic`, however, the last node is `x=0.75` rather than the right end of the domain at `x=1`.
This reflects the fact that `x=0` and `x=1` are the same location.

In a `Bounded` direction, however, the left and right end points differ and are included among the nodes.
For example,

```jldoctest fields
w = Field{Center, Center, Face}(grid)

znodes(w)

# output
5-element view(OffsetArray(::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, -2:8), 1:5) with eltype Float64:
 0.0
 0.25
 0.5
 0.75
 1.0
```

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
```
