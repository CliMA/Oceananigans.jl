using Adapt
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Grids: xnode, ynode, znode, λnode, φnode, rnode
using Oceananigans.Fields: AbstractField, default_indices, location
using Oceananigans.Operators: Δx, Δy, Δz, Δr, Ax, Δλ, Δφ, Ay, Az, volume
using Oceananigans.Operators: Operators, XNode, YNode, ZNode, ΛNode, ΦNode, RNode

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
$(TYPEDSIGNATURES)

Return the function associated with `metric::GridMetric` at `loc`ation.
"""
function metric_function(loc, metric)
    code = Tuple(interpolation_code(ℓ) for ℓ in loc)
    if metric isa typeof(volume)
        metric_function_symbol = Symbol(:V, code...)
    else
        metric_function_symbol = Symbol(metric, code...)
    end
    return getglobal(Operators, metric_function_symbol)
end

"""
$(TYPEDSIGNATURES)

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
grid_metric_operation(loc::Tuple{LX, LY, LZ}, metric, grid) where {LX<:Location, LY<:Location, LZ<:Location} =
    KernelFunctionOperation{LX, LY, LZ}(metric_function(loc, metric), grid)

# Instantiated location if location types are passed as values
grid_metric_operation(Loc::Tuple, metric, grid) = grid_metric_operation((Loc[1](), Loc[2](), Loc[3]()), metric, grid)

const NodeMetric = Union{XNode, YNode, ZNode, ΛNode, ΦNode, RNode}

function grid_metric_operation(loc::Tuple{LX, LY, LZ}, metric::NodeMetric, grid) where {LX<:Location, LY<:Location, LZ<:Location}
    ℓx, ℓy, ℓz = loc
    ξnode = metric_function(loc, metric)
    return KernelFunctionOperation{LX, LY, LZ}(ξnode, grid, ℓx, ℓy, ℓz)
end

#####
##### Spacings
#####

# Docstrings for xspacings, yspacings, λspacings, φspacings live in
# src/Grids/nodes_and_spacings.jl and for zspacings, rspacings in
# src/Grids/vertical_discretization.jl so that @autodocs picks them up
# from the Oceananigans.Grids module where these functions are exported.

function xspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δx_op = KernelFunctionOperation{LX, LY, LZ}(Δx, grid, ℓx, ℓy, ℓz)
    return Δx_op
end

function yspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δy_op = KernelFunctionOperation{LX, LY, LZ}(Δy, grid, ℓx, ℓy, ℓz)
    return Δy_op
end

function zspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δz_op = KernelFunctionOperation{LX, LY, LZ}(Δz, grid, ℓx, ℓy, ℓz)
    return Δz_op
end

function rspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δr_op = KernelFunctionOperation{LX, LY, LZ}(Δr, grid, ℓx, ℓy, ℓz)
    return Δr_op
end

function λspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δλ_op = KernelFunctionOperation{LX, LY, LZ}(Δλ, grid, ℓx, ℓy, ℓz)
    return Δλ_op
end

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
