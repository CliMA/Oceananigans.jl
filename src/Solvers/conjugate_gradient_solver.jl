using Oceananigans.Utils: prettysummary, @apply_regionally
using LinearAlgebra: norm, dot

mutable struct ConjugateGradientSolver{A, G, L, T, F, M, P, E, N}
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
    enforce_gauge_condition! :: E
               residual_norm :: N
end

Architectures.architecture(solver::ConjugateGradientSolver) = solver.architecture
iteration(cgs::ConjugateGradientSolver) = cgs.iteration

initialize_precondition_product(preconditioner, template_field) = similar(template_field)
initialize_precondition_product(::Nothing, template_field) = nothing

Base.summary(::ConjugateGradientSolver) = "ConjugateGradientSolver"

# "Nothing" preconditioner
@inline precondition!(z, ::Nothing, r, args...) = r

# Default no gauge condition enforcement
@inline no_gauge_enforcement!(x, r) = nothing

"""
    ConjugateGradientSolver(linear_operation;
                            template_field,
                            maxiter = size(template_field.grid),
                            reltol = sqrt(eps(template_field.grid)),
                            abstol = 0,
                            preconditioner = nothing,
                            enforce_gauge_condition! = no_gauge_enforcement!)

Return a `ConjugateGradientSolver` that solves the linear equation ``A x = b``
using a iterative conjugate gradient method with optional preconditioning.

The solver is used by calling

```julia
solve!(x, solver::PreconditionedConjugateGradientOperator, b, args...)
```

for `solver`, right-hand side `b`, solution `x`, and optional arguments `args...`.

Arguments
=========

* `linear_operation`: Function with signature `linear_operation!(p, y, args...)` that calculates
                      `A * y` and stores the result in `p` for a "candidate solution" `y`. `args...`
                      are optional positional arguments passed from `solve!(x, solver, b, args...)`.

* `template_field`: Dummy field that is the same type and size as `x` and `b`, which
                    is used to infer the `architecture`, `grid`, and to create work arrays
                    that are used internally by the solver.

* `maxiter`: Maximum number of iterations the solver may perform before exiting.

* `reltol, abstol`: Relative and absolute tolerance for convergence of the algorithm.
                    The iteration stops when `norm(A * x - b) < tolerance`.

* `preconditioner`: Object for which `precondition!(z, preconditioner, r, args...)` computes `z = P * r`,
                    where `r` is the residual. Typically `P` is approximately `A⁻¹`.

* `enforce_gauge_condition!`: Function with signature `enforce_gauge_condition!(x, r)` that
                              enforces a gauge condition on the solution `x` and residual `r`.
                              This is useful for problems where the solution is not unique, such as
                              the Poisson equation with purely Neumann boundary conditions.
                              The function is called at the end of each iteration of a conjugate
                              gradient iteration to ensure that the solution remains consistent
                              with the gauge condition.
                              The default is `no_gauge_enforcement!`, which does not enforce a gauge condition.

See [`solve!`](@ref) for more information about the preconditioned conjugate-gradient algorithm.
"""
function ConjugateGradientSolver(linear_operation;
                                 template_field::AbstractField,
                                 maxiter = prod(size(template_field)),
                                 reltol = sqrt(eps(eltype(template_field.grid))),
                                 abstol = 0,
                                 preconditioner = nothing,
                                 enforce_gauge_condition! = no_gauge_enforcement!,
                                 residual_norm = norm)

    arch = architecture(template_field)
    grid = template_field.grid

    # Create work arrays for solver
    linear_operator_product = similar(template_field) # A*xᵢ = qᵢ
    search_direction = similar(template_field) # pᵢ
    residual = similar(template_field) # rᵢ

    # Either nothing (no preconditioner) or P*xᵢ = zᵢ
    precondition_product = initialize_precondition_product(preconditioner, template_field)

    FT = eltype(grid)

    return ConjugateGradientSolver(arch,
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
                                   precondition_product,
                                   enforce_gauge_condition!,
                                   residual_norm)
end

"""
    solve!(x, solver::ConjugateGradientSolver, b, args...)

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
function solve!(x, solver::ConjugateGradientSolver, b, args...)
    # Initialize
    solver.iteration = 0

    # q = A * x
    q = solver.linear_operator_product

    @apply_regionally initialize_solution!(q, x, b, solver, args...)

    residual_norm = solver.residual_norm(solver.residual)
    tolerance = max(solver.reltol * residual_norm, solver.abstol)

    @debug "ConjugateGradientSolver, |b|: $(norm(b))"
    @debug "ConjugateGradientSolver, |A * x|: $(norm(q))"

    while iterating(solver, tolerance)
        iterate!(x, solver, b, args...)
    end

    return x
end

@inline function perform_linear_operation!(linear_operation!, q, p, args...)
    @apply_regionally linear_operation!(q, p, args...)
end

function iterate!(x, solver, b, args...)
    r = solver.residual
    p = solver.search_direction
    q = solver.linear_operator_product

    @debug "ConjugateGradientSolver $(solver.iteration), |r|: $(norm(r))"

    # Preconditioned:   z = P * r
    # Unpreconditioned: z = r
    @apply_regionally z = precondition!(solver.preconditioner_product, solver.preconditioner, r, args...)

    ρ = dot(z, r)

    @debug "ConjugateGradientSolver $(solver.iteration), ρ: $ρ"
    @debug "ConjugateGradientSolver $(solver.iteration), |z|: $(norm(z))"

    @apply_regionally perform_iteration!(q, p, ρ, z, solver, args...)

    perform_linear_operation!(solver.linear_operation!, q, p, args...)

    α = ρ / dot(p, q)

    @debug "ConjugateGradientSolver $(solver.iteration), |q|: $(norm(q))"
    @debug "ConjugateGradientSolver $(solver.iteration), α: $α"

    @apply_regionally update_solution_and_residuals!(x, r, q, p, α, solver.enforce_gauge_condition!)

    solver.iteration += 1
    solver.ρⁱ⁻¹ = ρ

    return nothing
end

""" first iteration of the PCG """
function initialize_solution!(q, x, b, solver, args...)
    solver.linear_operation!(q, x, args...)
    # r = b - A * x
    parent(solver.residual) .= parent(b) .- parent(q)

    return nothing
end

""" one conjugate gradient iteration """
function perform_iteration!(q, p, ρ, z, solver, args...)
    pp = parent(p)
    zp = parent(z)

    if solver.iteration == 0
        pp .= zp
    else
        β = ρ / solver.ρⁱ⁻¹
        pp .= zp .+ β .* pp

        @debug "ConjugateGradientSolver $(solver.iteration), β: $β"
    end

    # q = A * p
    solver.linear_operation!(q, p, args...)

    return nothing
end

function update_solution_and_residuals!(x, r, q, p, α, enforce_gauge_condition!)
    xp = parent(x)
    rp = parent(r)
    qp = parent(q)
    pp = parent(p)

    xp .+= α .* pp
    rp .-= α .* qp

    enforce_gauge_condition!(x, r)

    return nothing
end

function iterating(solver, tolerance)
    # End conditions
    solver.iteration >= solver.maxiter && return false
    solver.residual_norm(solver.residual) <= tolerance && return false
    return true
end

function Base.show(io::IO, solver::ConjugateGradientSolver)
    print(io, "ConjugateGradientSolver on ", summary(solver.architecture), "\n",
              "├── template_field: ", summary(solver.residual), "\n",
              "├── grid: ", summary(solver.grid), "\n",
              "├── linear_operation!: ", prettysummary(solver.linear_operation!), "\n",
              "├── preconditioner: ", prettysummary(solver.preconditioner), "\n",
              "├── reltol: ", prettysummary(solver.reltol), "\n",
              "├── abstol: ", prettysummary(solver.abstol), "\n",
              "├── residual_norm: ", prettysummary(solver.residual_norm), "\n",
              "└── maxiter: ", solver.maxiter)
end
