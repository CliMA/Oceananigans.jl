using Statistics: mean!, sum!

using Oceananigans.Utils: tupleit
using Oceananigans.Grids: regular_dimensions
using Oceananigans.Fields: Scan, condition_operand, reverse_cumsum!, AbstractReducing, AbstractAccumulating

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

struct Averaging <: AbstractReducing end
const Average = Scan{<:Averaging}
Base.summary(r::Average) = string("Average of ", summary(r.operand), " over dims ", r.dims)

"""
    Average(field::AbstractField; dims=:, condition=nothing, mask=0)

Return `Reduction` representing a spatial average of `field` over `dims`.

Over regularly-spaced dimensions this is equivalent to a numerical `mean!`.

Over dimensions of variable spacing, `field` is multiplied by the
appropriate grid length, area or volume, and divided by the total
spatial extent of the interval.
"""
function Average(field::AbstractField; dims=:, condition=nothing, mask=0)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dx = reduction_grid_metric(dims)

    if all(d in regular_dimensions(field.grid) for d in dims)
        # Dimensions being reduced are regular; just use mean!
        operand = condition_operand(field, condition, mask)
        return Scan(Averaging(), mean!, operand, dims)
    else
        # Compute "size" (length, area, or volume) of averaging region
        metric = GridMetricOperation(location(field), dx, field.grid)
        L = sum(metric; condition, mask, dims)

        # Construct summand of the Average
        L⁻¹_field_dx = field * dx / L

        operand = condition_operand(L⁻¹_field_dx, condition, mask)

        return Scan(Averaging(), sum!, operand, dims)
    end
end

struct Integrating <: AbstractReducing end
const Integral = Scan{<:Integrating}
Base.summary(r::Integral) = string("Integral of ", summary(r.operand), " over dims ", r.dims)

"""
    Integral(field::AbstractField; dims=:, condition=nothing, mask=0)


Return a `Reduction` representing a spatial integral of `field` over `dims`.

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
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
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
│   └── west: Nothing, east: Nothing, south: Nothing, north: Nothing, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 1×1×14 OffsetArray(::Array{Float64, 3}, 1:1, 1:1, -2:11) with eltype Float64 with indices 1:1×1:1×-2:11
    └── max=0.9375, min=0.0625, mean=0.5

julia> C_op = CumulativeIntegral(c, dims=3)
CumulativeIntegral of BinaryOperation at (Center, Center, Center) over dims 3
└── operand: BinaryOperation at (Center, Center, Center)
    └── grid: 1×1×8 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo

julia> C = compute!(Field(C_op))

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

