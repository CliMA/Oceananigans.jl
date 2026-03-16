module Solvers

using Reactant
using Oceananigans.Grids: Bounded, Periodic, Flat

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
import ..Architectures: AnyConcreteReactantArray

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}
const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#####
##### Periodic topology (FFT) - uses ReactantAbstractFFTsExt plans
#####

function plan_forward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantFFTInPlacePlan{T}(dims)
end

function plan_backward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantIFFTInPlacePlan{T}(dims)
end

#####
##### Bounded topology (DCT) - not yet supported
#####

function plan_forward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

function plan_backward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

#####
##### Flat topology - no transform needed
#####

plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

end # module
