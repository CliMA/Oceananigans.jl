using Oceananigans:
    CPU, GPU, AbstractGrid, AbstractPoissonSolver,
    BC, Periodic, ModelBoundaryConditions

using Oceananigans.Grids: RegularCartesianGrid, unpack_grid

poisson_bc_symbol(::BC) = :N
poisson_bc_symbol(::BC{<:Periodic}) = :P

function PressureSolver(arch, grid, pressure_bcs, planner_flag=FFTW.PATIENT)
    x = poisson_bc_symbol(pressure_bcs.x.left)
    y = poisson_bc_symbol(pressure_bcs.y.left)
    z = poisson_bc_symbol(pressure_bcs.z.left)
    bc_symbol = Symbol(x, y, z)

    if bc_symbol == :PPN
        return HorizontallyPeriodicPressureSolver(arch, grid, pressure_bcs, planner_flag)
    end
end

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)
