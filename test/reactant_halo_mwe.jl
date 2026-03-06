# MWE: Reactant MLIR pass failure on periodic halo-filling kernel
#
# Standalone reproducer (no Oceananigans dependency) for the MLIR
# optimization pass bug on Linux x64 with halo size H=3.
# Passes with H=1, fails with H=3.

using KernelAbstractions
using Reactant
using CUDA
using OffsetArrays

const RBackend = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt).ReactantBackend

@kernel function fill_periodic_halo!(c, N, H)
    j, k = @index(Global, NTuple)
    @inbounds for i = 1:H
        parent(c)[i, j, k]     = parent(c)[N+i, j, k]
        parent(c)[N+H+i, j, k] = parent(c)[H+i, j, k]
    end
end

N, H = 4, 3
total = N + 2H
raw = Reactant.to_rarray(randn(total, total, total))
c = OffsetArray(raw, -H:N+H-1, -H:N+H-1, -H:N+H-1)

kernel = fill_periodic_halo!(RBackend(), (16, 16), (total, total))

println("Compiling with raise=true (H=$H)...")
compiled! = @compile raise=true raise_first=true kernel(c, N, H)
println("Running...")
compiled!(c, N, H)
println("Success!")
