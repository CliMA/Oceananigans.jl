using Statistics: mean!, sum!

using Oceananigans.Utils: tupleit
using Oceananigans.Grids: regular_dimensions
using Oceananigans.Fields: condition_operand
import Oceananigans.Fields: Reduction

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

struct Average end

function Reduction(avg::Average, field::AbstractField; condition = nothing, mask = 0, dims)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dx = reduction_grid_metric(dims)

    if all(d in regular_dimensions(field.grid) for d in dims)
        # Dimensions being reduced are regular; just use mean!
        return Reduction(mean!, condition_operand(field, condition, mask); dims)
    else
        # Compute "size" (length, area, or volume) of averaging region
        metric = GridMetricOperation(location(field), dx, field.grid)
        L = sum(metric; condition, mask, dims)

        # Construct summand of the Average
        L⁻¹_field_dx = field * dx / L

        return Reduction(sum!, condition_operand(L⁻¹_field_dx, condition, mask), dims)
    end
end

"""
    Average(field::AbstractField; condition = nothing, mask = 0, dims=:)

Return `Reduction` representing a spatial average of `field` over `dims`.

Over regularly-spaced dimensions this is equivalent to a numerical `mean!`.

Over dimensions of variable spacing, `field` is multiplied by the
appropriate grid length, area or volume, and divided by the total
spatial extent of the interval.
"""
Average(field::AbstractField; condition = nothing, mask = 0, dims=:) = Reduction(Average(), field; condition, mask, dims)

struct Integral end

function Reduction(int::Integral, field::AbstractField; condition = nothing, mask = 0, dims)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dx = reduction_grid_metric(dims)
    return Reduction(sum!, condition_operand(field * dx, condition, mask), dims)
end

"""
    Integral(field::AbstractField; condition = nothing, mask = 0, dims=:)


Return a `Reduction` representing a spatial integral of `field` over `dims`.

Example
=======

Compute the integral of ``f(x, y, z) = x y z`` over the domain
``(x, y, z) ∈ [0, 1] × [0, 1] × [0, 1]``.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(18, 18, 8), x=(0, 1), y=(0, 1), z=(0, 1), topology = (Periodic, Periodic, Periodic));

julia> f = CenterField(grid);

julia> set!(f, (x, y, z) -> x * y * z);

julia> ∫f = Field(Integral(f));

julia> compute!(∫f);

julia> ∫f[1, 1, 1]
0.125
```
"""
Integral(field::AbstractField; condition = nothing, mask = 0, dims=:) = Reduction(Integral(), condition_operand(field, condition, mask); dims)

#####
##### show
#####

Base.summary(r::Reduction{<:Average}) = string("Average of ", summary(r.operand), " over dims ", r.dims)
Base.summary(r::Reduction{<:Integral}) = string("Integral of ", summary(r.operand), " over dims ", r.dims)
