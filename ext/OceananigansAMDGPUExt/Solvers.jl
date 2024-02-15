module Solvers

using AMDGPU
using Oceananigans.Grids
using ..Architectures: ROCArray, ROCmGPU

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform

function plan_forward_transform(A::ROCArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_fft!(A, dims)
end

function plan_backward_transform(A::ROCArray, ::Union{Bounded, Periodic}, dims, planner_flag)
    length(dims) == 0 && return nothing
    return AMDGPU.rocFFT.plan_bfft!(A, dims)
end

plan_backward_transform(A::Union{Array, ROCArray}, ::Flat, args...) = nothing
plan_forward_transform(A::Union{Array, ROCArray}, ::Flat, args...) = nothing

end # module