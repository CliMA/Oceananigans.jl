using Statistics: mean!, sum!

using Oceananigans.Utils: tupleit
using Oceananigans.Grids: regular_dimensions
using Oceananigans.Fields: Scan, condition_operand, reverse_cumsum!, AbstractAccumulating, AbstractReducing
using Oceananigans.Fields: filter_nothing_dims, instantiated_location

#####
##### Metric inference
#####

reduction_grid_metric(dims::Number) = reduction_grid_metric(tuple(dims))

reduction_grid_metric(dims) = dims === tuple(1)  ? Δx :
                              dims === tuple(2)  ? Δy :
                              dims === tuple(3)  ? Δz :
                              dims === (1, 2)    ? Az :
                              dims === (1, 3)    ? Ay :
                              dims === (2, 3)    ? Ax :
                              dims === (1, 2, 3) ? volume :
                              throw(ArgumentError("Cannot determine grid metric for reducing over dims = $dims"))

#####
##### Metric reductions
#####

struct Averaging{V} <: AbstractReducing
    volume :: V
end

const Average = Scan{<:Averaging}
const AveragedField = Field{<:Any, <:Any, <:Any, <:Average}

function average!(avg::AveragedField, operand)
    sum!(avg, operand)
    averaging = avg.operand.type

    V = if averaging.volume isa Field
        interior(averaging.volume)
    else
        averaging.volume
    end

    interior(avg) ./= V

    return avg
end

Base.summary(r::Average) = string("Average of ", summary(r.operand), " over dims ", r.dims)

"""
    Average(field::AbstractField; dims=:, condition=nothing, mask=0)

Return `Reduction` representing a spatial average of `field` over `dims`.

Over regularly-spaced dimensions this is equivalent to a numerical `mean!`.

Over dimensions of variable spacing, `field` is multiplied by the
appropriate "averaging metric" (length, area or volume for 1D, 2D, or 3D averages),
and divided by the sum of the metric over the averaging region.

See [`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
for information and examples using `condition` and `mask` kwargs.
"""
function Average(field::AbstractField; dims=:, condition=nothing, mask=0)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dims = filter_nothing_dims(dims, instantiated_location(field))

    if all(d in regular_dimensions(field.grid) for d in dims)
        # Dimensions being reduced are regular, so we don't need to involve the grid metrics
        operand = condition_operand(field, condition, mask)
        N = conditional_length(operand, dims)
        averaging = Averaging(N)
        return Scan(averaging, average!, operand, dims)
    else
        # Compute "size" (length, area, or volume) of averaging region
        dx = reduction_grid_metric(dims)
        metric = GridMetricOperation(location(field), dx, field.grid)
        volume = sum(metric; condition, mask, dims)

        # Construct summand of the Average
        # V⁻¹_field_dx = field * dx / volume
        # operand = condition_operand(V⁻¹_field_dx, condition, mask)
        # return Scan(Averaging(), sum!, operand, dims)

        field_dx = field * dx
        operand = condition_operand(field_dx, condition, mask)

        metric = GridMetricOperation(location(field), dx, field.grid)
        volume = sum(metric; condition, mask, dims)
        averaging = Averaging(volume)
        return Scan(averaging, average!, operand, dims)
    end
end

struct Integrating <: AbstractReducing end
const Integral = Scan{<:Integrating}
Base.summary(r::Integral) = string("Integral of ", summary(r.operand), " over dims ", r.dims)

"""
    Integral(field::AbstractField; dims=:, condition=nothing, mask=0)


Return a `Reduction` representing a spatial integral of `field` over `dims`.

See [`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
for information and examples using `condition` and `mask` kwargs.

Example
=======

Compute the integral of ``f(x, y, z) = x y z`` over the domain
``(x, y, z) ∈ [0, 1] × [0, 1] × [0, 1]``. The analytical answer
is ``∭ x y z \\, dx \\, dy \\, dz = 1/8``.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1));

julia> f = CenterField(grid);

julia> set!(f, (x, y, z) -> x * y * z)
8×8×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 14×14×14 OffsetArray(::Array{Float64, 3}, -2:11, -2:11, -2:11) with eltype Float64 with indices -2:11×-2:11×-2:11
    └── max=0.823975, min=0.000244141, mean=0.125

julia> ∫f = Integral(f)
Integral of BinaryOperation at (Center, Center, Center) over dims (1, 2, 3)
└── operand: BinaryOperation at (Center, Center, Center)
    └── grid: 8×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo

julia> ∫f = Field(Integral(f));

julia> compute!(∫f);

julia> ∫f[1, 1, 1]
0.125
```
"""
function Integral(field::AbstractField; dims=:, condition=nothing, mask=0)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dims = filter_nothing_dims(dims, instantiated_location(field))
    dx = reduction_grid_metric(dims)
    operand = condition_operand(field * dx, condition, mask)
    return Scan(Integrating(), sum!, operand, dims)
end

#####
##### CumulativeIntegral
#####

struct CumulativelyIntegrating <: AbstractAccumulating end
const CumulativeIntegral = Scan{<:CumulativelyIntegrating}
Base.summary(c::CumulativeIntegral) = string("CumulativeIntegral of ", summary(c.operand), " over dims ", c.dims)

"""
    CumulativeIntegral(field::AbstractField; dims, reverse=false, condition=nothing, mask=0)

Return an `Accumulation` representing the cumulative spatial integral of `field` over `dims`.

See [`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
for information and examples using `condition` and `mask` kwargs.

Example
=======

Compute the cumulative integral of ``f(z) = z`` over z ∈ [0, 1].

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=8, z=(0, 1), topology=(Flat, Flat, Bounded));

julia> c = CenterField(grid);

julia> set!(c, z -> z)
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── grid: 1×1×8 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Nothing, east: Nothing, south: Nothing, north: Nothing, bottom: ZeroFlux, top: ZeroFlux, immersed: Nothing
└── data: 1×1×14 OffsetArray(::Array{Float64, 3}, 1:1, 1:1, -2:11) with eltype Float64 with indices 1:1×1:1×-2:11
    └── max=0.9375, min=0.0625, mean=0.5

julia> C_op = CumulativeIntegral(c, dims=3)
CumulativeIntegral of BinaryOperation at (Center, Center, Center) over dims 3
└── operand: BinaryOperation at (Center, Center, Center)
    └── grid: 1×1×8 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo

julia> C = compute!(Field(C_op))
1×1×8 Field{Center, Center, Center} on RectilinearGrid on CPU
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 8)
├── grid: 1×1×8 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── operand: CumulativeIntegral of BinaryOperation at (Center, Center, Center) over dims 3
├── status: time=0.0
└── data: 1×1×14 OffsetArray(::Array{Float64, 3}, 1:1, 1:1, -2:11) with eltype Float64 with indices 1:1×1:1×-2:11
    └── max=0.5, min=0.0078125, mean=0.199219

julia> C[1, 1, 8]
0.5
```
"""
function CumulativeIntegral(field::AbstractField; dims, reverse=false, condition=nothing, mask=0)
    dims ∈ (1, 2, 3) || throw(ArgumentError("CumulativeIntegral only supports dims=1, 2, or 3."))
    maybe_reverse_cumsum = reverse ? reverse_cumsum! : cumsum!
    dx = reduction_grid_metric(dims)
    operand = condition_operand(field * dx, condition, mask)
    return Scan(CumulativelyIntegrating(), maybe_reverse_cumsum, operand, dims)
end
