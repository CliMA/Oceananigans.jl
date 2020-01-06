using Oceananigans:
    CPU, GPU, AbstractGrid, AbstractPoissonSolver,
    BC, Periodic, ModelBoundaryConditions

using Oceananigans.Grids: RegularCartesianGrid, unpack_grid

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)
