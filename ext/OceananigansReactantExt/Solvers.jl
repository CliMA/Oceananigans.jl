module Solvers

using ..Architectures: AnyConcreteReactantArray
using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: BoundaryTopology, Flat, Periodic
using Oceananigans.Solvers: permute_index, unpermute_index
using Reactant

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}
const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#####
##### FFTs: Periodic directions use them directly, and Bounded directions build DCTs from them,
##### as on the GPU (see `FFTBasedDCTArchitecture`)
#####

function Oceananigans.Solvers.plan_forward_transform(A::AnyReactantArray, ::Union{BoundaryTopology, Periodic}, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantFFTInPlacePlan{T}(dims)
end

function Oceananigans.Solvers.plan_backward_transform(A::AnyReactantArray, ::Union{BoundaryTopology, Periodic}, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantIFFTInPlacePlan{T}(dims)
end

Oceananigans.Solvers.plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
Oceananigans.Solvers.plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

#####
##### Complex-valued kernels with a computed store index or a complex `ifelse` cannot be raised yet
##### (Reactant.jl#3497, #3475), and raising is required for reverse-mode differentiation,
##### so these steps are array operations under Reactant.
#####

gather(A, indices, dim) = dim == 1 ? A[indices, :, :] : dim == 2 ? A[:, indices, :] : A[:, :, indices]

function Oceananigans.Solvers.permute_indices!(dst, src, ::ReactantState, grid, dim)
    N = size(src, dim)
    dst .= gather(src, [unpermute_index(i, N) for i in 1:N], dim)
    return nothing
end

function Oceananigans.Solvers.unpermute_indices!(dst, src, ::ReactantState, grid, dim)
    N = size(src, dim)
    dst .= gather(src, [permute_index(i, N) for i in 1:N], dim)
    return nothing
end

function Oceananigans.Solvers.spectral_poisson_solve!(ϕc, b, λx, λy, λz, m, ::ReactantState, grid)
    denominator = λx .+ λy .+ λz .- m
    ϕc .= ifelse.(denominator .== 0, zero(eltype(ϕc)), .- b ./ denominator)
    return nothing
end

end # module
