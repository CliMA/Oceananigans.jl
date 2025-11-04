using Adapt
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Grids: xnode, ynode, znode, λnode, φnode, rnode
using Oceananigans.Fields: AbstractField, default_indices, location
using Oceananigans.Operators: Δx, Δy, Δz, Δr, Ax, Δλ, Δφ, Ay, Az, volume
using Oceananigans.Operators: XNode, YNode, ZNode, ΛNode, ΦNode, RNode
using Oceananigans.Operators: x, y, z, λ, φ, r

import Oceananigans.Grids: xspacings, yspacings, zspacings, rspacings, λspacings, φspacings

const GridMetric = Union{XNode, YNode, ZNode, ΛNode, ΦNode, RNode,
                         typeof(Δx),
                         typeof(Δy),
                         typeof(Δz),
                         typeof(Δr),
                         typeof(Δλ),
                         typeof(Δφ),
                         typeof(Ax),
                         typeof(Ay),
                         typeof(Az),
                         typeof(volume)} # Do we want it to be `volume` or just `V` like in the Operators module?

metric_function(loc, ::XNode) = xnode
metric_function(loc, ::YNode) = ynode
metric_function(loc, ::ZNode) = znode
metric_function(loc, ::ΛNode) = λnode
metric_function(loc, ::ΦNode) = φnode
metric_function(loc, ::RNode) = rnode

"""
    metric_function(loc, metric::GridMetric)

Return the function associated with `metric::GridMetric` at `loc`ation.
"""
function metric_function(loc, metric)
    code = Tuple(interpolation_code(ℓ) for ℓ in loc)
    if metric isa typeof(volume)
        metric_function_symbol = Symbol(:V, code...)
    else
        metric_function_symbol = Symbol(metric, code...)
    end
    return getglobal(@__MODULE__, metric_function_symbol)
end

"""
    grid_metric_operation(loc, metric, grid)

Return a `KernelFunctionOperation` of `metric` that participates
in a `BinaryOperation` at `loc`ation of the `grid`.

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.AbstractOperations: Ax, grid_metric_operation

julia> Axᶠᶜᶜ = grid_metric_operation((Face, Center, Center), Ax, RectilinearGrid(size=(2, 2, 3), extent=(1, 2, 3)))
KernelFunctionOperation at (Face, Center, Center)
├── grid: 2×2×3 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×2×3 halo
├── kernel_function: Axᶠᶜᶜ (generic function with 2 methods)
└── arguments: ()
```

```jldoctest
julia> using Oceananigans

julia> c = CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 2, 3)));

julia> using Oceananigans.Operators: Δz

julia> c_dz = c * Δz; # returns BinaryOperation between Field and GridMetric

julia> set!(c, 1);

julia> c_dz[1, 1, 1]
3.0
```
"""
grid_metric_operation(loc::Tuple, metric, grid) = grid_metric_operation(instantiate(loc), metric, grid)

grid_metric_operation(loc::Tuple{LX, LY, LZ}, metric, grid) where {LX<:Location, LY<:Location, LZ<:Location} =
    KernelFunctionOperation{LX, LY, LZ}(metric_function(loc, metric), grid)

const NodeMetric = Union{XNode, YNode, ZNode, ΛNode, ΦNode, RNode}

function grid_metric_operation(loc::Tuple{LX, LY, LZ}, metric::NodeMetric, grid) where {LX<:Location, LY<:Location, LZ<:Location} 
    ℓx, ℓy, ℓz = loc
    ξnode = metric_function(loc, metric)
    return KernelFunctionOperation{LX, LY, LZ}(ξnode, grid, ℓx, ℓy, ℓz)
end

#####
##### Spacings
#####

"""
    xspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``x`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> xspacings(grid, Center(), Center(), Center())
KernelFunctionOperation at (Center, Center, Center)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δx (generic function with 29 methods)
└── arguments: ("Center", "Center", "Center")
```
"""
function xspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δx_op = KernelFunctionOperation{LX, LY, LZ}(Δx, grid, ℓx, ℓy, ℓz)
    return Δx_op
end

"""
    yspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``y`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> yspacings(grid, Center(), Face(), Center())
KernelFunctionOperation at (Center, Face, Center)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δy (generic function with 29 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function yspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δy_op = KernelFunctionOperation{LX, LY, LZ}(Δy, grid, ℓx, ℓy, ℓz)
    return Δy_op
end

"""
    zspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``z`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> zspacings(grid, Center(), Center(), Face())
KernelFunctionOperation at (Center, Center, Face)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δz (generic function with 28 methods)
└── arguments: ("Center", "Center", "Face")
```
"""
function zspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δz_op = KernelFunctionOperation{LX, LY, LZ}(Δz, grid, ℓx, ℓy, ℓz)
    return Δz_op
end

"""
    rspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``r`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(2, 4, 8), extent=(1, 1, 1));

julia> rspacings(grid, Center(), Center(), Face())
KernelFunctionOperation at (Center, Center, Face)
├── grid: 2×4×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×3×3 halo
├── kernel_function: Δr (generic function with 28 methods)
└── arguments: ("Center", "Center", "Face")
```
"""
function rspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δr_op = KernelFunctionOperation{LX, LY, LZ}(Δr, grid, ℓx, ℓy, ℓz)
    return Δr_op
end

"""
    λspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``λ`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(36, 34, 25),
                                    longitude = (-180, 180),
                                    latitude = (-85, 85),
                                    z = (-1000, 0));

julia> λspacings(grid, Center(), Face(), Center())
KernelFunctionOperation at (Center, Face, Center)
├── grid: 36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── kernel_function: Δλ (generic function with 29 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function λspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δλ_op = KernelFunctionOperation{LX, LY, LZ}(Δλ, grid, ℓx, ℓy, ℓz)
    return Δλ_op
end

"""
    φspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``φ`` direction at location `ℓx, ℓy, ℓz`.

Examples
========
```jldoctest
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(36, 34, 25),
                                    longitude = (-180, 180),
                                    latitude = (-85, 85),
                                    z = (-1000, 0));

julia> φspacings(grid, Center(), Face(), Center())
KernelFunctionOperation at (Center, Face, Center)
├── grid: 36×34×25 LatitudeLongitudeGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×3 halo and with precomputed metrics
├── kernel_function: Δφ (generic function with 29 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function φspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δφ_op = KernelFunctionOperation{LX, LY, LZ}(Δφ, grid, ℓx, ℓy, ℓz)
    return Δφ_op
end

@inline xspacings(field::AbstractField) = xspacings(field.grid, location(field)...)
@inline yspacings(field::AbstractField) = yspacings(field.grid, location(field)...)
@inline zspacings(field::AbstractField) = zspacings(field.grid, location(field)...)
@inline λspacings(field::AbstractField) = λspacings(field.grid, location(field)...)
@inline φspacings(field::AbstractField) = φspacings(field.grid, location(field)...)
@inline rspacings(field::AbstractField) = rspacings(field.grid, location(field)...)

# Some defaults for e.g. easy CFL computations.
@inline xspacings(grid::AbstractGrid) = xspacings(grid, Center(), Center(), Center())
@inline yspacings(grid::AbstractGrid) = yspacings(grid, Center(), Center(), Center())
@inline zspacings(grid::AbstractGrid) = zspacings(grid, Center(), Center(), Center())
@inline λspacings(grid::AbstractGrid) = λspacings(grid, Center(), Center(), Center())
@inline φspacings(grid::AbstractGrid) = φspacings(grid, Center(), Center(), Center())
@inline rspacings(grid::AbstractGrid) = rspacings(grid, Center(), Center(), Center())
