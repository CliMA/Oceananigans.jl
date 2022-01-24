using Adapt
using Oceananigans.Operators

# Define aliases for some spacings and areas at all locations.
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)

    x_spacing_alias = Symbol(:Δx, LX, LY, LZ)
    y_spacing_alias = Symbol(:Δy, LX, LY, LZ)
    z_spacing_alias = Symbol(:Δz, LX, LY, LZ)

    z_area_alias = Symbol(:Az, LX, LY, LZ)

    x_spacing_function = Symbol(:Δx, LX, LY, :ᵃ)
    y_spacing_function = Symbol(:Δy, LX, LY, :ᵃ)
    z_spacing_function = Symbol(:Δz, :ᵃᵃ, LZ)

    z_area_function = Symbol(:Az, LX, LY, :ᵃ)

    @eval begin
        const $x_spacing_alias = $x_spacing_function
        const $y_spacing_alias = $y_spacing_function
        const $z_spacing_alias = $z_spacing_function
        const $z_area_alias = $z_area_function
    end
end

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

julia> grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 2, 3)); c = CenterField(grid);

julia> c_dz = c * Δz # returns BinaryOperation between Field and GridMetricOperation
BinaryOperation at (Center, Center, Center)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
└── tree:
    * at (Center, Center, Center)
    ├── 1×1×1 Field{Center, Center, Center} on RectilinearGrid on CPU
    └── Δzᵃᵃᶜ at (Center, Center, Center)

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

julia> grid = RectilinearGrid(size=(2, 2, 2), extent=(1, 2, 3)); c = CenterField(grid);

julia> c .= 1;

julia> c_dV = c * volume
BinaryOperation at (Center, Center, Center)
├── grid: 2×2×2 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
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

Returns the function associated with `metric::AbstractGridMetric`
at `loc`ation.
"""
function metric_function(loc, metric::AbstractGridMetric)
    code = Tuple(interpolation_code(ℓ) for ℓ in loc)
    prefix = metric_function_prefix(metric)
    metric_function_symbol = Symbol(prefix, code...)
    return eval(metric_function_symbol)
end

struct GridMetricOperation{LX, LY, LZ, G, T, M} <: AbstractOperation{LX, LY, LZ, G, T}
          metric :: M
            grid :: G

    function GridMetricOperation{LX, LY, LZ}(metric::M, grid::G) where {LX, LY, LZ, M, G}
        T = eltype(grid)
        return new{LX, LY, LZ, G, T, M}(metric, grid)
    end
end

Adapt.adapt_structure(to, gm::GridMetricOperation{LX, LY, LZ}) where {LX, LY, LZ}=
         GridMetricOperation{LX, LY, LZ}(Adapt.adapt(to, gm.metric),
                                         Adapt.adapt(to, gm.grid))

@inline Base.getindex(gm::GridMetricOperation, i, j, k) = gm.metric(i, j, k, gm.grid)

# Special constructor for BinaryOperation
GridMetricOperation(L, metric, grid) = GridMetricOperation{L[1], L[2], L[3]}(metric_function(L, metric), grid)


