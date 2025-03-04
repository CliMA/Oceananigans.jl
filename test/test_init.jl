using Reactant
using Enzyme
using Metal

Pkg.instantiate(; verbose=true)
Pkg.precompile(; strict=true)
Pkg.status()

try
    MPI.versioninfo()
catch; end

try
    CUDA.precompile_runtime()
    CUDA.versioninfo()
catch; end

