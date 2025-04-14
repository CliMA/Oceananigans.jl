# TODO: move these imports into test-specific files?
using Reactant
using Enzyme
using Metal
using AMDGPU
using oneAPI
using ConservativeRegridding

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

