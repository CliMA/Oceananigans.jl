using Oceananigans:
    CPU, GPU, AbstractGrid, AbstractPoissonSolver,
    RegularCartesianGrid, BC, Periodic, ModelBoundaryConditions

# PoissonBCs are named XYZ, where each of X, Y, and Z is either
# 'P' (for Periodic) or 'N' (for Neumann).
struct PPN <: PoissonBCs end
struct PNN <: PoissonBCs end

# Not yet supported:
#struct PPP <: PoissonBCs end
#struct NNN <: PoissonBCs end
#struct NPP <: PoissonBCs end
#struct PNP <: PoissonBCs end
#struct NPN <: PoissonBCs end
#struct NNP <: PoissonBCs end

poisson_bc_symbol(::BC) = :N
poisson_bc_symbol(::BC{<:Periodic}) = :P

"""
    PoissonBCs(bcs)

Returns the boundary conditions for the Poisson solver corresponding
to the model boundary conditions `bcs`.
"""
function PoissonBCs(bcs)
    # We assume that bounary conditions on all fields are
    # consistent with the boundary conditions on one side of ``u``.
    x = poisson_bc_symbol(bcs.u.x.left)
    y = poisson_bc_symbol(bcs.u.y.left)
    z = poisson_bc_symbol(bcs.u.z.left)

    return eval(Expr(:call, Symbol(x, y, z)))
end

PoissonBCs(model_bcs::ModelBoundaryConditions) = PoissonBCs(model_bcs.solution)

PoissonSolver(::CPU, pbcs::PoissonBCs, grid::AbstractGrid) = PoissonSolverCPU(pbcs, grid)
PoissonSolver(::GPU, pbcs::PoissonBCs, grid::AbstractGrid) = PoissonSolverGPU(pbcs, grid)

unpack_grid(grid::AbstractGrid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz

"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

"""
    λi(grid::AbstractGrid, ::PoissonBCs)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λi(grid::AbstractGrid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    @. (2sin((is-1)*π/Nx) / (Lx/Nx))^2
end

"""
    λj(grid::AbstractGrid, ::PPN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::AbstractGrid, ::PPN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/Ny) / (Ly/Ny))^2
end

"""
    λj(grid::AbstractGrid, ::PNN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::AbstractGrid, ::PNN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/(2Ny)) / (Ly/Ny))^2
end

"""
    λk(grid::AbstractGrid, ::PoissonBCs)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λk(grid::AbstractGrid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    @. (2sin((ks-1)*π/(2Nz)) / (Lz/Nz))^2
end
