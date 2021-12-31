using Statistics: mean!, sum!

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

"""docstring..."""
struct Average end

function Reduction(avg::Average, field::AbstractField; dims)
    dx = reduction_grid_metric(dims)

    # Compute "size" (length, area, or volume) of averaging region
    metric = GridMetricOperation(location(field), dx, field.grid)
    L = sum(metric; dims)
    L⁻¹_field_dx = field * dx / L

    return Reduction(sum!, L⁻¹_field_dx; dims)
end

Average(field::AbstractField; dims=:) = Reduction(Average(), field; dims)

const AveragedField = Field{<:Any, <:Any, <:Any, <:Reduction{<:Average}}

"""docstring..."""
struct Integral end

function Reduction(int::Integral, field::AbstractField; dims)
    dx = reduction_grid_metric(dims)
    return Reduction(sum!, field * dx; dims)
end

# Convenience
Integral(field::AbstractField; dims=:) = Reduction(Integral(), field; dims)

const IntegratedField = Field{<:Any, <:Any, <:Any, <:Reduction{<:Integral}}
