using Reactant
using Enzyme
using Metal

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

