using Adapt
using Oceananigans.Operators
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: AbstractField, default_indices, location

import Oceananigans.Grids: xspacings, yspacings, zspacings, λspacings, φspacings

abstract type AbstractGridMetric end

struct XSpacingMetric <: AbstractGridMetric end
struct YSpacingMetric <: AbstractGridMetric end
struct ZSpacingMetric <: AbstractGridMetric end

metric_function_prefix(::XSpacingMetric) = :Δx
metric_function_prefix(::YSpacingMetric) = :Δy
metric_function_prefix(::ZSpacingMetric) = :Δz

struct XAreaMetric <: AbstractGridMetric end
struct YAreaMetric <: AbstractGridMetric end
struct ZAreaMetric <: AbstractGridMetric end

metric_function_prefix(::XAreaMetric) = :Ax
metric_function_prefix(::YAreaMetric) = :Ay
metric_function_prefix(::ZAreaMetric) = :Az

struct VolumeMetric <: AbstractGridMetric end

metric_function_prefix(::VolumeMetric) = :V

# Convenient instances for users
const Δx = XSpacingMetric()
const Δy = YSpacingMetric()

"""
    Δz = ZSpacingMetric()

Instance of `ZSpacingMetric` that generates `BinaryOperation`s
between `AbstractField`s and the vertical grid spacing evaluated
at the same location as the `AbstractField`.

`Δx` and `Δy` play a similar role for horizontal grid spacings.

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.AbstractOperations: Δz

julia> c = CenterField(RectilinearGrid(size=(1, 1, 1), extent=(1, 2, 3)));

julia> c_dz = c * Δz # returns BinaryOperation between Field and GridMetricOperation
BinaryOperation at (Center, Center, Center)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
└── tree:
    * at (Center, Center, Center)
    ├── 1×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
    └── Δzᶜᶜᶜ at (Center, Center, Center)

julia> c .= 1;

julia> c_dz[1, 1, 1]
3.0
```
"""
const Δz = ZSpacingMetric()

const Ax = XAreaMetric()
const Ay = YAreaMetric()
const Az = ZAreaMetric()

"""
    volume = VolumeMetric()

Instance of `VolumeMetric` that generates `BinaryOperation`s
between `AbstractField`s and their cell volumes. Summing
this `BinaryOperation` yields an integral of `AbstractField`
over the domain.

Example
=======

```jldoctest
julia> using Oceananigans

julia> using Oceananigans.AbstractOperations: volume

julia> c = CenterField(RectilinearGrid(size=(2, 2, 2), extent=(1, 2, 3)));

julia> c .= 1;

julia> c_dV = c * volume
BinaryOperation at (Center, Center, Center)
├── grid: 2×2×2 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 2×2×2 halo
└── tree:
    * at (Center, Center, Center)
    ├── 2×2×2 Field{Center, Center, Center} on RectilinearGrid on CPU
    └── Vᶜᶜᶜ at (Center, Center, Center)

julia> c_dV[1, 1, 1]
0.75

julia> sum(c_dV)
6.0
```
"""
const volume = VolumeMetric()

"""
    metric_function(loc, metric::AbstractGridMetric)

Return the function associated with `metric::AbstractGridMetric`
at `loc`ation.
"""
function metric_function(loc, metric::AbstractGridMetric)
    code = Tuple(interpolation_code(ℓ) for ℓ in loc)
    prefix = metric_function_prefix(metric)
    metric_function_symbol = Symbol(prefix, code...)
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

# Special constructor for BinaryOperation
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
├── kernel_function: xspacing (generic function with 27 methods)
└── arguments: ("Center", "Center", "Center")
```
"""
function xspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δx_op = KernelFunctionOperation{LX, LY, LZ}(xspacing, grid, ℓx, ℓy, ℓz)
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
├── kernel_function: yspacing (generic function with 27 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function yspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δy_op = KernelFunctionOperation{LX, LY, LZ}(yspacing, grid, ℓx, ℓy, ℓz)
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
├── kernel_function: zspacing (generic function with 27 methods)
└── arguments: ("Center", "Center", "Face")
```
"""
function zspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δz_op = KernelFunctionOperation{LX, LY, LZ}(zspacing, grid, ℓx, ℓy, ℓz)
    return Δz_op
end

"""
    λspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``z`` direction at location `ℓx, ℓy, ℓz`.

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
├── kernel_function: λspacing (generic function with 5 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function λspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δλ_op = KernelFunctionOperation{LX, LY, LZ}(λspacing, grid, ℓx, ℓy, ℓz)
    return Δλ_op
end

"""
    φspacings(grid, ℓx, ℓy, ℓz)

Return a `KernelFunctionOperation` that computes the grid spacings for `grid`
in the ``z`` direction at location `ℓx, ℓy, ℓz`.

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
├── kernel_function: φspacing (generic function with 5 methods)
└── arguments: ("Center", "Face", "Center")
```
"""
function φspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = map(typeof, (ℓx, ℓy, ℓz))
    Δφ_op = KernelFunctionOperation{LX, LY, LZ}(φspacing, grid, ℓx, ℓy, ℓz)
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
