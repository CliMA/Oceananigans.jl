import Oceananigans.Fields: Reduction

##### 
##### Metric inference
##### 

reduction_grid_metric(dims::Number) reduction_grid_metric(tuple(dims))
reduction_grid_metric(dims) = dims === tuple(1)  ? Δx :
                              dims === tuple(2)  ? Δy :
                              dims === tuple(3)  ? Δz :
                              dims === (1, 2)    ? Az :
                              dims === (1, 3)    ? Ay :
                              dims === (2, 3)    ? Ax :
                              dims === (1, 2, 3) ? volume

##### 
##### Metric reductions
##### 

abstract type AbstractMetricReduction end
struct Average <: AbstractMetricReduction end
struct Integral <: AbstractMetricReduction end

reduction(::Average) = mean!
reduction(::Integral) = sum!

function Reduction(r::AbstractMetricReductions, operand; dims)
    dx = reduction_grid_metric(dims)
    field_dx = field * dx
    return Reduction(reduction(r), field_dx; dims)
end

# Convenience
Average(field::AbstractField; dims) = Reduction(Average(), field; dims)
Integral(field::AbstractField; dims) = Reduction(Integral(), field; dims)

