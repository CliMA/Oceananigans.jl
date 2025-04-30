using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Statistics: mean
using Oceananigans.Solvers: compute_laplacian!, DefaultPreconditioner

using KernelAbstractions: @kernel, @index

import Oceananigans.Architectures: architecture

struct KrylovPoissonSolver{G, R, S}
    grid :: G
    right_hand_side :: R
    krylov_solver :: S
end

architecture(kps::KrylovPoissonSolver) = architecture(kps.grid)
iteration(kps::KrylovPoissonSolver) = iteration(kps.krylov_solver)

Base.summary(kps::KrylovPoissonSolver) =
    "KrylovPoissonSolver with $(kps.krylov_solver) method, $(summary(kps.krylov_solver.preconditioner)) preconditioner on $(summary(kps.grid))"

function Base.show(io::IO, kps::KrylovPoissonSolver)
    A = architecture(kps.grid)
    print(io, "KrylovPoissonSolver:", '\n',
              "├── grid: ", summary(kps.grid), '\n',
              "└── krylov_solver: ", summary(kps.krylov_solver), '\n',
              "    ├── maxiter: ", prettysummary(kps.krylov_solver.maxiter), '\n',
              "    ├── reltol: ", prettysummary(kps.krylov_solver.reltol), '\n',
              "    ├── abstol: ", prettysummary(kps.krylov_solver.abstol), '\n',
              "    ├── preconditioner: ", prettysummary(kps.krylov_solver.preconditioner), '\n',
              "    └── iteration: ", prettysummary(kps.krylov_solver.workspace.stats.niter))
end

"""
    KrylovPoissonSolver(grid;
                        method = :cg,
                        preconditioner = DefaultPreconditioner(),
                        reltol = sqrt(eps(grid)),
                        abstol = sqrt(eps(grid)),
                        kw...)

Creates a `KrylovPoissonSolver` with `method` on `grid` using a `preconditioner`.
`KrylovPoissonSolver` is iterative, and will stop when both the relative error in the
pressure solution is smaller than `reltol` and the absolute error is smaller than `abstol`. Other
keyword arguments are passed to `KrylovSolver`.
"""
function KrylovPoissonSolver(grid;
                             method = :cg,
                             preconditioner = DefaultPreconditioner(),
                             reltol = sqrt(eps(grid)),
                             abstol = sqrt(eps(grid)),
                             kw...)

    # if method ∉ [:cg, :bicgstab]
    #     @warn "Currently, KrylovPoissonSolver only supports :cg and :bicgstab methods. Support for other methods will be added soon!"
    # end    

    if preconditioner isa DefaultPreconditioner # try to make a useful default
        if grid isa ImmersedBoundaryGrid && grid.underlying_grid isa GridWithFFTSolver
            preconditioner = fft_poisson_solver(grid.underlying_grid)
        else
            preconditioner = DiagonallyDominantPreconditioner()
        end
    end

    rhs = CenterField(grid)

    krylov_solver = KrylovSolver(compute_laplacian!;
                                 method,
                                 reltol,
                                 abstol,
                                 preconditioner,
                                 template_field = rhs,
                                 kw...)
        
    return KrylovPoissonSolver(grid, rhs, krylov_solver)
end