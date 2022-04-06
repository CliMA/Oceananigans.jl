using Oceananigans.Architectures: architecture
using Oceananigans.Grids: interior_parent_indices
using Oceananigans.Utils: prettysummary
using Statistics: norm, dot
using LinearAlgebra

import Oceananigans.Architectures: architecture

mutable struct PreconditionedConjugateGradientSolver{A, G, L, T, F, M, P}
               architecture :: A
                       grid :: G
          linear_operation! :: L
                     reltol :: T
                     abstol :: T
                    maxiter :: Int
                  iteration :: Int
                       ρⁱ⁻¹ :: T
    linear_operator_product :: F
           search_direction :: F
                   residual :: F
             preconditioner :: M
     preconditioner_product :: P
end

architecture(solver::PreconditionedConjugateGradientSolver) = solver.architecture

initialize_precondition_product(preconditioner, template_field) = similar(template_field)
initialize_precondition_product(::Nothing, template_field) = nothing

# "Nothing" preconditioner
@inline precondition!(z, ::Nothing, r, args...) = r

"""
    PreconditionedConjugateGradientSolver(linear_operation;
                                          template_field,
                                          maxiter = size(template_field.grid),
                                          reltol = sqrt(eps(eltype(template_field.grid))),
                                          abstol = 0,
                                          preconditioner = nothing)

Returns a PreconditionedConjugateGradientSolver that solves the linear equation
``A x = b`` using a iterative conjugate gradient method with optional preconditioning.
The solver is used by calling

```
solve!(x, solver::PreconditionedConjugateGradientOperator, b, args...)
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

* `maxiter`: Maximum number of iterations the solver may perform before exiting.

* `reltol, abstol`: Relative and absolute tolerance for convergence of the algorithm.
                    The iteration stops when `norm(A * x - b) < tolerance`.

* `preconditioner`: Object for which `precondition!(z, preconditioner, r, args...)`
                    computes `z = P * r`, where `r` is the residual. Typically `P`
                    is approximately `A⁻¹`.

See [`solve!`](@ref) for more information about the preconditioned conjugate-gradient algorithm.
"""
function PreconditionedConjugateGradientSolver(linear_operation;
                                               template_field::AbstractField,
                                               maxiter = prod(size(template_field)),
                                               reltol = sqrt(eps(eltype(template_field.grid))),
                                               abstol = 0,
                                               preconditioner = nothing)

    arch = architecture(template_field)
    grid = template_field.grid

    # Create work arrays for solver
    linear_operator_product = similar(template_field) # A*xᵢ = qᵢ
    search_direction = similar(template_field) # pᵢ
            residual = similar(template_field) # rᵢ

    # Either nothing (no preconditioner) or P*xᵢ = zᵢ
    precondition_product = initialize_precondition_product(preconditioner, template_field)

    FT = eltype(grid)

    return PreconditionedConjugateGradientSolver(arch,
                                                 grid,
                                                 linear_operation,
                                                 FT(reltol),
                                                 FT(abstol),
                                                 maxiter,
                                                 0,
                                                 zero(FT),
                                                 linear_operator_product,
                                                 search_direction,
                                                 residual,
                                                 preconditioner,
                                                 precondition_product)
end

"""
    solve!(x, solver::PreconditionedConjugateGradientSolver, b, args...)

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

This function executes the psuedocode algorithm
    
```
β  = 0
r = b - A(x)
iteration  = 0

Loop:
     if iteration > maxiter
        break
     end

     ρ = r ⋅ z

     z = M(r)
     β = ρⁱ⁻¹ / ρ
     p = z + β * p
     q = A(p)

     α = ρ / (p ⋅ q)
     x = x + α * p
     r = r - α * q

     if |r| < tolerance
        break
     end

     iteration += 1
     ρⁱ⁻¹ = ρ
```
"""
function solve!(x, solver::PreconditionedConjugateGradientSolver, b, args...)

    # Initialize
    solver.iteration = 0

    # q = A*x
    q = solver.linear_operator_product
    solver.linear_operation!(q, x, args...)

    # r = b - A*x
    parent(solver.residual) .= parent(b) .- parent(q)

    residual_norm = norm(solver.residual)
    tolerance = max(solver.reltol * residual_norm, solver.abstol)

    @debug "PreconditionedConjugateGradientSolver, |b|: $(norm(b))"
    @debug "PreconditionedConjugateGradientSolver, |A(x)|: $(norm(q))"

    while iterating(solver, tolerance)
        iterate!(x, solver, b, args...)
    end

    return x
end

function iterate!(x, solver, b, args...)
    r = solver.residual
    p = solver.search_direction
    q = solver.linear_operator_product

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |r|: $(norm(r))"

    # Preconditioned:   z = P * r
    # Unpreconditioned: z = r
    z = precondition!(solver.preconditioner_product, solver.preconditioner, r, x, args...) 
    ρ = dot(z, r)

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), ρ: $ρ"
    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |z|: $(norm(z))"

    pp = parent(p)
    zp = parent(z)

    if solver.iteration == 0
        pp .= zp
    else
        β = ρ / solver.ρⁱ⁻¹
        pp .= zp .+ β .* pp

        @debug "PreconditionedConjugateGradientSolver $(solver.iteration), β: $β"
    end

    # q = A * p
    solver.linear_operation!(q, p, args...)
    α = ρ / dot(p, q)

    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), |q|: $(norm(q))"
    @debug "PreconditionedConjugateGradientSolver $(solver.iteration), α: $α"
        
    xp = parent(x)
    rp = parent(r)
    qp = parent(q)

    xp .+= α .* pp
    rp .-= α .* qp

    solver.iteration += 1
    solver.ρⁱ⁻¹ = ρ

    return nothing
end

function iterating(solver, tolerance)
    # End conditions
    solver.iteration >= solver.maxiter && return false
    norm(solver.residual) <= tolerance && return false
    return true
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "PreconditionedConjugateGradientSolver on ", summary(solver.architecture), '\n',
              "├── template field: ", summary(solver.residual), '\n',
              "├── grid: ", summary(solver.grid),
              "├── linear_operation!: ", prettysummary(solver.linear_operation!), '\n',
              "├── preconditioner: ", prettysummary(solver.preconditioner), '\n',
              "├── reltol: ", prettysummary(solver.reltol), '\n',
              "├── abstol: ", prettysummary(solver.abstol), '\n',
              "└── maxiter: ", solver.maxiter)
end

