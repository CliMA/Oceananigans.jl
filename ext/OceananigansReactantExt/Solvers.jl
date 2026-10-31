module Solvers

using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: BoundaryTopology, Periodic, Flat
using Oceananigans.Solvers: permute_index, unpermute_index

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform,
                             permute_indices!, unpermute_indices!, spectral_poisson_solve!
import ..Architectures: AnyConcreteReactantArray

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}
const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#####
##### FFTs: Periodic directions use them directly, and Bounded directions build DCTs from them,
##### as on the GPU (see `FFTBasedDCTArchitecture`)
#####

function plan_forward_transform(A::AnyReactantArray, ::Union{BoundaryTopology, Periodic}, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantFFTInPlacePlan{T}(dims)
end

function plan_backward_transform(A::AnyReactantArray, ::Union{BoundaryTopology, Periodic}, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantIFFTInPlacePlan{T}(dims)
end

plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

#####
##### Complex-valued kernels with a computed store index or a complex `ifelse` cannot be raised yet
##### (Reactant.jl#3497, #3475), and raising is required for reverse-mode differentiation,
##### so these steps are array operations under Reactant.
#####

gather(A, indices, dim) = dim == 1 ? A[indices, :, :] : dim == 2 ? A[:, indices, :] : A[:, :, indices]

function permute_indices!(dst, src, ::ReactantState, grid, dim)
    N = size(src, dim)
    dst .= gather(src, [unpermute_index(i, N) for i in 1:N], dim)
    return nothing
end

function unpermute_indices!(dst, src, ::ReactantState, grid, dim)
    N = size(src, dim)
    dst .= gather(src, [permute_index(i, N) for i in 1:N], dim)
    return nothing
end

function spectral_poisson_solve!(ϕc, b, λx, λy, λz, m, ::ReactantState, grid)
    denominator = λx .+ λy .+ λz .- m
    ϕc .= ifelse.(denominator .== 0, zero(eltype(ϕc)), .- b ./ denominator)
    return nothing
end

end # module
