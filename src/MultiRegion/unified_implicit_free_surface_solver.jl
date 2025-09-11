using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Architectures
using Oceananigans.Fields: Field

using Oceananigans.Models.HydrostaticFreeSurfaceModels: PCGImplicitFreeSurfaceSolver

build_implicit_step_solver(::Val{:PreconditionedConjugateGradient}, grid::MultiRegionGrids, settings, gravitational_acceleration) =
    throw(ArgumentError("Cannot use PCG solver with Multi-region grids!! Select :Default or :HeptadiagonalIterativeSolver as solver_method"))
build_implicit_step_solver(::Val{:Default}, grid::ConformalCubedSphereGridOfSomeKind, settings, gravitational_acceleration) =
    PCGImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)
build_implicit_step_solver(::Val{:PreconditionedConjugateGradient}, grid::ConformalCubedSphereGridOfSomeKind, settings, gravitational_acceleration) =
    PCGImplicitFreeSurfaceSolver(grid, settings, gravitational_acceleration)
build_implicit_step_solver(::Val{:HeptadiagonalIterativeSolver}, grid::ConformalCubedSphereGridOfSomeKind, settings, gravitational_acceleration) =
    throw(ArgumentError("Cannot use Matrix solvers with ConformalCubedSphereGrid!! Select :Default or :PreconditionedConjugateGradient as solver_method"))
