using Oceananigans.Architectures: architecture
using Oceananigans.Grids: interior_parent_indices
using Statistics: norm, dot
using LinearAlgebra
# using AlgebraicMultigrid
using AlgebraicMultigrid: _solve!, init, RugeStubenAMG
using Oceananigans.Operators: volume, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶜᶜᵃ, Δyᵃᶜᵃ, Δxᶜᵃᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, ∇²ᶜᶜᶜ

import Oceananigans.Architectures: architecture

mutable struct MultigridSolver{A, G, L, T, M}
               architecture :: A
                       grid :: G
            linear_operator :: L
                  tolerance :: T
         maximum_iterations :: Int
              amg_algorithm :: M
end

architecture(solver::MultigridSolver) = solver.architecture
    
"""
    MultigridSolver(linear_operation;
                    template_field,
                    maximum_iterations = size(template_field.grid),
                    tolerance = 1e-13)

Returns a MultigridSolver that solves the linear equation
``A x = b`` using a multigrid method.
The solver is used by calling

```
solve!(x, solver::MultigridSolver, b, args...)
```

for `solver`, right-hand side `b`, solution `x`, and optional arguments `args...`.

Arguments
=========

* `template_field`: Dummy field that is the same type and size as `x` and `b`, which
                    is used to infer the `architecture`, `grid`, and to create work arrays
                    that are used internally by the solver.

* `linear_operation`: Function with signature `linear_operation!(p, y, args...)` that calculates
                     `A*y` and stores the result in `p` for a "candidate solution `y`. `args...`
                     are optional positional arguments passed from `solve!(x, solver, b, args...)`.

* `maximum_iterations`: Maximum number of iterations the solver may perform before exiting.

* `tolerance`: Tolerance for convergence of the algorithm. The algorithm quits when
               `norm(A * x - b) < tolerance`.

* `precondition`: Function with signature `preconditioner!(z, y, args...)` that calculates
                  `P * y` and stores the result in `z` for linear operator `P`.
                  Note that some precondition algorithms describe the step
                  "solve `M * x = b`" for precondition `M`"; in this context,
                  `P = M⁻¹`.

See [`solve!`](@ref) for more information about the preconditioned conjugate-gradient algorithm.
"""
function MultigridSolver(grid,
                        linear_operation!, args...;
                        maximum_iterations = 100, #prod(size(template_field)),
                        tolerance = 1e-13, #sqrt(eps(eltype(template_field.grid))),
                        amg_algorithm = RugeStubenAMG(),
                        )

    arch = architecture(grid)

    A = create_matrix(grid, linear_operation!, args...)

    return MultigridSolver(arch,
                           grid,
                           A,
                           tolerance,
                           maximum_iterations,
                           amg_algorithm
                           )
end

function create_matrix(grid, linear_operator!, args...)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    # Currently assuming Nz = 1
    A = spzeros(eltype(grid), Nx*Ny, Nx*Ny)

    make_column(f) = reshape(interior(f), (Nx*Ny*Nz, 1))

    eᵢⱼₖ = CenterField(grid)
    ∇²eᵢⱼₖ = CenterField(grid)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        eᵢⱼₖ .= 0
        ∇²eᵢⱼₖ .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)

        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Nx*(j-1)+i] = make_column(∇²eᵢⱼₖ)
    end
    
    return A
end

"""
    solve!(x, solver::MultigridSolver, b, args...)

Solve `A * x = b` using an iterative conjugate-gradient method, where `A * x` is
determined by `solver.linear_operation`
    
See figure 2.5 in

> The Preconditioned Conjugate Gradient Method in "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods" Barrett et. al, 2nd Edition.
    
Given:
  * Linear Preconditioner operator `M!(solution, x, other_args...)` that computes `M * x = solution`
  * A matrix operator `A` as a function `A()`;
  * A dot product function `norm()`;
  * A right-hand side `b`;
  * An initial guess `x`; and
  * Local vectors: `z`, `r`, `p`, `q`
"""

function solve!(x, solver::MultigridSolver, b, args...; kwargs...)
    grid = b.grid
    Nx, Ny, Nz = size(grid)
    b_array = collect(reshape(interior(b), Nx * Ny * Nz))
    x_array = collect(reshape(interior(x), Nx * Ny * Nz)) 

    solt = init(solver.amg_algorithm, solver.linear_operator, b_array)

    _solve!(x_array, solt.ml, solt.b, maxiter=solver.maximum_iterations, abstol = solver.tolerance, kwargs...)
    
    interior(x) .= reshape(x_array, Nx, Ny, Nz)
    fill_halo_regions!(x)
end

function Base.show(io::IO, solver::MultigridSolver)
    print(io, "Multigrid solver.\n")
    print(io, " Problem size = "  , size(solver.grid), '\n')
    print(io, " Grid = "  , solver.grid)
    return nothing
end
