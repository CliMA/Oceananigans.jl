using Statistics: mean!, sum!

using Oceananigans.Utils: tupleit
using Oceananigans.Grids: regular_dimensions
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
        return Reduction(mean!, field; condition, mask, dims)
    else
        # Compute "size" (length, area, or volume) of averaging region
        metric = GridMetricOperation(location(field), dx, field.grid)
        L = sum(metric; dims)

        # Construct summand of the Average
        L⁻¹_field_dx = field * dx / L

        return Reduction(sum!, L⁻¹_field_dx; condition, mask, dims)
    end
end

"""
    Average(field; dims=:)

Return `Reduction` representing a spatial average of `field` over `dims`.

Over regularly-spaced dimensions this is equivalent to a numerical `mean!`.

Over dimensions of variable spacing, `field` is multipled by the
appropriate grid length, area or volume, and divided by the total
spatial extent of the interval.
"""
Average(field::AbstractField; condition = nothing, mask = 0, dims=:) = Reduction(Average(), field; condition, mask, dims)

struct Integral end

function Reduction(int::Integral, field::AbstractField; condition = nothing, mask = 0, dims)
    dims = dims isa Colon ? (1, 2, 3) : tupleit(dims)
    dx = reduction_grid_metric(dims)
    return Reduction(sum!, field * dx; condition, mask, dims)
end

"""
    Integral(field; dims=:)

Return a `Reduction` representing a spatial integral of `field` over `dims`.
"""
Integral(field::AbstractField; condition = nothing, mask = 0, dims=:) = Reduction(Integral(), field; condition, mask, dims)

#####
##### show
#####

Base.summary(r::Reduction{<:Average}) = string("Average of ", summary(r.operand), " over dims ", r.dims)
Base.summary(r::Reduction{<:Integral}) = string("Integral of ", summary(r.operand), " over dims ", r.dims)
                                             
