using Adapt
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: AbstractField, default_indices, location
using Oceananigans.Operators: Δx, Δy, Δz, Ax, Δλ, Δφ, Ay, Az, volume

import Oceananigans.Grids: xspacings, yspacings, zspacings, λspacings, φspacings

const AbstractGridMetric = Union{typeof(Δx),
                                 typeof(Δy),
                                 typeof(Δz),
                                 typeof(Δλ),
                                 typeof(Δφ),
                                 typeof(Ax),
                                 typeof(Ay),
                                 typeof(Az),
                                 typeof(volume)} # Do we want it to be `volume` or just `V` like in the Operators module?

"""
    metric_function(loc, metric::AbstractGridMetric)

Return the function associated with `metric::AbstractGridMetric` at `loc`ation.
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

struct GridMetricOperation{LX, LY, LZ, G, T, M} <: AbstractOperation{LX, LY, LZ, G, T}
          metric :: M
            grid :: G
    function GridMetricOperation{LX, LY, LZ}(metric::M, grid::G) where {LX, LY, LZ, M, G}
        T = eltype(grid)
        return new{LX, LY, LZ, G, T, M}(metric, grid)
    end
end

Adapt.adapt_structure(to, gm::GridMetricOperation{LX, LY, LZ}) where {LX, LY, LZ} =
         GridMetricOperation{LX, LY, LZ}(Adapt.adapt(to, gm.metric),
                                         Adapt.adapt(to, gm.grid))

on_architecture(to, gm::GridMetricOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    GridMetricOperation{LX, LY, LZ}(on_architecture(to, gm.metric),
                                    on_architecture(to, gm.grid))

@inline Base.getindex(gm::GridMetricOperation, i, j, k) = gm.metric(i, j, k, gm.grid)

indices(::GridMetricOperation) = default_indices(3)

"""
    GridMetricOperation(L, metric, grid)

Instance of `GridMetricOperation` that generates `BinaryOperation`s between `AbstractField`s and the metric `metric`
at the same location as the `AbstractField`.

Example
=======
```jldoctest
julia> using Oceananigans

julia> using Oceananigans.Operators: Δz

julia> c = CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 2, 3)));

julia> c_dz = c * Δz; # returns BinaryOperation between Field and GridMetricOperation

julia> c .= 1;

julia> c_dz[1, 1, 1]
3.0
```
"""
GridMetricOperation(L, metric, grid) = GridMetricOperation{L[1], L[2], L[3]}(metric_function(L, metric), grid)

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
