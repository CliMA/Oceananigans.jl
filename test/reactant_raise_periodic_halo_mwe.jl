# MWE: Reactant "cannot raise if yet" error on periodic halo-filling kernel
#
# Reproduces MLIR pass failure when compiling a periodic halo-copy kernel
# with `raise=true` and halo size H ≥ 2. The `for i = 1:H` loop pattern
# cannot be raised to pure functional (StableHLO) form by Reactant's MLIR passes.
#
# This is the same error that causes compute_simple_Gu! tests to fail on CI
# (Linux x64) for any topology with a Periodic direction and halo=(3,3,3).
#
# Usage:
#   julia --project test/reactant_raise_periodic_halo_mwe.jl

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
println("Expected error: 'cannot raise if yet (non-pure or yielded values)'")
compiled = @compile raise=true raise_first=true kernel(c, N, H)
println("Running...")
compiled(c, N, H)
println("Success!")
