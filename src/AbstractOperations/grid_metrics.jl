using Adapt
using Oceananigans.Operators
using Oceananigans.Fields: default_indices, location

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

function xspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = typeof.((ℓx, ℓy, ℓz))
    Δx_op = KernelFunctionOperation{LX, LY, LZ}(xspacing, grid, ℓx, ℓy, ℓz)
    Δx_field = Field(Δx_op)
    compute!(Δx_field)
    return Δx_field
end

function yspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = typeof.((ℓx, ℓy, ℓz))
    Δy_op = KernelFunctionOperation{LX, LY, LZ}(yspacing, grid, ℓx, ℓy, ℓz)
    Δy_field = Field(Δy_op)
    compute!(Δy_field)
    return Δy_field
end

function zspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = typeof.((ℓx, ℓy, ℓz))
    Δz_op = KernelFunctionOperation{LX, LY, LZ}(zspacing, grid, ℓx, ℓy, ℓz)
    Δz_field = Field(Δz_op)
    compute!(Δz_field)
    return Δz_field
end

function λspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = typeof.((ℓx, ℓy, ℓz))
    Δλ_op = KernelFunctionOperation{LX, LY, LZ}(λspacing, grid, ℓx, ℓy, ℓz)
    Δλ_field = Field(Δλ_op)
    compute!(Δλ_field)
    return Δλ_field
end

function φspacings(grid, ℓx, ℓy, ℓz)
    LX, LY, LZ = typeof.((ℓx, ℓy, ℓz))
    Δφ_op = KernelFunctionOperation{LX, LY, LZ}(φspacing, grid, ℓx, ℓy, ℓz)
    Δφ_field = Field(Δφ_op)
    compute!(Δφ_field)
    return Δφ_field
end

@inline xspacings(grid, ℓx) = xspacings(grid, ℓx, Center(), Center())
@inline yspacings(grid, ℓy) = yspacings(grid, Center(), ℓy, Center())
@inline zspacings(grid, ℓz) = zspacings(grid, Center(), Center(), ℓz)

@inline λspacings(grid, ℓx) = λspacings(grid, ℓx, Center(), Center())
@inline φspacings(grid, ℓy) = φspacings(grid, Center(), ℓy, Center())

@inline xspacings(grid, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, Center())
@inline yspacings(grid, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, Center())

@inline xspacings(field) = xspacings(field.grid, location(field)...)
@inline yspacings(field) = yspacings(field.grid, location(field)...)
@inline zspacings(field) = zspacings(field.grid, location(field)...)
