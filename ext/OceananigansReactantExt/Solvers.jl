module Solvers

using ..Architectures: AnyConcreteReactantArray
using Oceananigans: Oceananigans
using Oceananigans.Grids: BoundaryTopology, Flat, Periodic
using Reactant

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}
const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#####
##### Periodic topology (FFT) - uses ReactantAbstractFFTsExt plans
#####

function Oceananigans.Solvers.plan_forward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantFFTInPlacePlan{T}(dims)
end

function Oceananigans.Solvers.plan_backward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantIFFTInPlacePlan{T}(dims)
end

#####
##### Bounded topology (DCT) - not yet supported
#####

function Oceananigans.Solvers.plan_forward_transform(A::AnyReactantArray, ::BoundaryTopology, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

function Oceananigans.Solvers.plan_backward_transform(A::AnyReactantArray, ::BoundaryTopology, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

#####
##### Flat topology - no transform needed
#####

Oceananigans.Solvers.plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
Oceananigans.Solvers.plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

end # module
