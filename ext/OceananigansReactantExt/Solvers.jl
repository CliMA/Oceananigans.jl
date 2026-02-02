module Solvers

using Reactant
using AbstractFFTs

using Oceananigans.Grids: Bounded, Periodic, Flat

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
import ..Architectures: AnyConcreteReactantArray

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}

# Lightweight plan structs - FFT executes when applied via `plan * A`
struct ReactantForwardFFTPlan{D}
    dims::D
end

struct ReactantBackwardFFTPlan{D}
    dims::D
end

# Plan application - compiled to XLA FFT ops during Reactant tracing
Base.:*(plan::ReactantForwardFFTPlan, A::AbstractArray) = (AbstractFFTs.fft!(A, plan.dims); A)
Base.:*(plan::ReactantBackwardFFTPlan, A::AbstractArray) = (AbstractFFTs.ifft!(A, plan.dims); A)

# Periodic topology (FFT)
function plan_forward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    return ReactantForwardFFTPlan(dims)
end

function plan_backward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    return ReactantBackwardFFTPlan(dims)
end

# Bounded topology (DCT) - not yet supported
function plan_forward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic or ExplicitFreeSurface.")
end

function plan_backward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic or ExplicitFreeSurface.")
end

# Flat topology - no transform needed
plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

end # module
