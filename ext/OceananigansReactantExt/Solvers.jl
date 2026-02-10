module Solvers

using Reactant
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: Bounded, Periodic, Flat, inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
import Oceananigans.Models.NonhydrostaticModels: compute_source_term!
import ..Architectures: AnyConcreteReactantArray
import ..Grids: ReactantGrid

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

#####
##### B.6.7 workaround: avoid ComplexF64 stores in KA kernels.
##### KA kernel writes Float64 into a scratch, then broadcast copies into complex storage.
#####

@kernel function _compute_source_term_real!(scratch, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds scratch[i, j, k] = active * δ
end

function compute_source_term!(solver::FFTBasedPoissonSolver{<:ReactantGrid}, ::Nothing, Ũ, Δt)
    rhs = solver.storage
    arch = architecture(solver)
    grid = solver.grid
    scratch = similar(rhs, real(eltype(rhs)))
    launch!(architecture(solver), grid, :xyz, _compute_source_term_real!, scratch, grid, Ũ)
    rhs .= scratch
    return nothing
end

end # module
