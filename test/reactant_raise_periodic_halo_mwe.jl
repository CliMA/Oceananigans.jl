# MWE: compute_simple_Gu! fails with raise=true on periodic topologies
#
# Reactant's MLIR raise pass fails on the periodic halo-filling kernel
# (called by fill_halo_regions!) with halo ≥ 2:
#   "cannot raise if yet (non-pure or yielded values)"
#
# This reproduces the 3 CI failures in compute_simple_Gu! for topologies
# (Periodic, Periodic, Bounded), (Periodic, Bounded, Bounded),
# and (Bounded, Periodic, Bounded).
#
# Usage:
#   julia --project test/reactant_raise_periodic_halo_mwe.jl

using Oceananigans
using Reactant
using CUDA

arch = ReactantState()
topo = (Periodic, Periodic, Bounded)
grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1), topology=topo)

u = XFaceField(grid)
set!(u, randn(size(u)...))

println("Compiling fill_halo_regions! with raise=true on $topo grid...")
rfill! = @compile raise=true raise_first=true fill_halo_regions!(u)
println("Running...")
rfill!(u)
println("Success!")
