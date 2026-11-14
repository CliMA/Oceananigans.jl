include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Pkg
using CUDA

Pkg.instantiate(; verbose=true)
Pkg.precompile(; strict=true)
Pkg.status()

using Oceananigans.DistributedComputations

try
    @root MPI.versioninfo()
catch; end

try
    CUDA.precompile_runtime()
    @root CUDA.versioninfo()
catch; end
