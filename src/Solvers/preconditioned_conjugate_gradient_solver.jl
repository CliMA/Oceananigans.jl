using Oceananigans.Grids: interior_parent_indices
using Statistics
using LinearAlgebra

struct PreconditionedConjugateGradientSolver{A, G, L, M, T, F}
              architecture :: A
                      grid :: G
          linear_operator! :: L
           preconditioner! :: M
                 tolerance :: T
        maximum_iterations :: Int
    linear_operator_output :: F
     preconditioner_output :: F
          search_direction :: F
                  residual :: F
end

no_preconditioner!(args...) = nothing

function PreconditionedConjugateGradientSolver(; architecture,
                                               grid,
                                               linear_operator,
                                               template_solution,
                                               max_iterations = size(grid),
                                               tolerance = 1e-13,
                                               preconditioner = no_preconditioner)

    # Create work arrays for solver
    linear_operator_output = similar(example_solution) # q
     preconditioner_output = similar(example_solution) # z
          search_direction = similar(example_solution) # p
                  residual = similar(example_solution) # r

   return PreconditionedConjugateGradientSolver(architecture,
                                                grid,
                                                linear_operator,
                                                preconditioner,
                                                tolerance,
                                                maximum_iterations,
                                                linear_operator_output,
                                                preconditioner_output,
                                                search_direction,
                                                residual)

end

function Statistics.norm(a::AbstractField)
    ii = interior_parent_indices(location(a, 1), topology(a.grid, 1), a.grid.Nx, a.grid.Hx)
    ji = interior_parent_indices(location(a, 2), topology(a.grid, 2), a.grid.Ny, a.grid.Hy)
    ki = interior_parent_indices(location(a, 3), topology(a.grid, 3), a.grid.Nz, a.grid.Hz)
    return sqrt(mapreduce(x -> x * x, +, view(a, ii, ji, ki)))
end

function Statistics.dot(a::AbstractField, b::AbstractField)
    ii = interior_parent_indices(location(a, 1), topology(a.grid, 1), a.grid.Nx, a.grid.Hx)
    ji = interior_parent_indices(location(a, 2), topology(a.grid, 2), a.grid.Ny, a.grid.Hy)
    ki = interior_parent_indices(location(a, 3), topology(a.grid, 3), a.grid.Nz, a.grid.Hz)
    return sqrt(mapreduce((x, y) -> x * y, +, view(a, ii, ji, ki), view(b, ii, ji, ki)))
end

"""
    solve!(x, solver::PreconditionedConjugateGradientSolver, b, args...)

Solves A*x = b using an iterative conjugate-gradient method,
where A*x is determined by solver.linear_operator
    
See fig 2.5 in

> The Preconditioned Conjugate Gradient Method in
  "Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods"
  Barrett et. al, 2nd Edition.
    
Given:
    * Linear Preconditioner operator M!(solution, x, other_args...) that computes `M*x=solution`
    * Linear A matrix operator A as a function A();
    * A dot product function norm();
    * A right-hand side b;
    * An initial guess x; and
    * Local vectors: z, r, p, q

This function executes the algorithm
    
```
    β  = 0
    r .= b - A(x)
    iteration  = 0
    
    Loop:
         if iteration > MAXIT
            break
         end

         z  = M(r)
         ρ .= dotprod(r, z)
         p  = z + β*p
         q  = A(p)

         α = ρ / dotprod(p, q)
         x = x .+ α * p
         r = r .- α * q

         if norm2(r) < tol
            break
         end

         iteration = iteration + 1
         ρⁱᵐ1 .= ρ
         β    .= ρⁱ⁻¹/ρ
```
"""
function solve!(initial_guess, solver::PreconditionedConjugateGradientSolver, b, args...)

    # Unpack some solver properties
    arch = solver.architecture
    grid = solver.grid
    
    # Initialize
    iteration = 0
    β = 0.0
    ρ = 0.0
    ρⁱ⁻¹ = 0.0

    x = initial_guess
    r = parent(solver.residual)
    p = parent(solver.search_direction)
    z = parent(solver.preconditioner_output)
    q = parent(solver.linear_operator_output)

    solver.linear_operator!(solver.linear_operator_output, x, args...)

    # r = b - A*x
    @. r = b - q

    @debug "PreconditionedConjugateGradientSolver $iteration, |b|: $(norm(b))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |A(x)|: $(norm(A(initial_guess, args...)))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |r|: $(norm(r))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |z|: $(norm(z))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |q|: $(norm(q))"

    while true
        # End conditions
        iteration >= solver.maximum_iterations && break
        norm(solver.residual) <= solver.tolerance && break

        # z = M(r)
        solver.preconditioner!(solver.preconditioner_output, solver.residual, args...)
        ρ = dot(solver.preconditioner_output, solver.residual)

        @debug "PreconditionedConjugateGradientSolver $iteration, ρ: $ρ"

        if iteration == 0
            p .= z
        else
            β = ρ / ρⁱ⁻¹
            @. p = z + β * p
        end

        # q = A * p
        solver.linear_operator!(solver.linear_operator_output, solver.search_direction, args...)

        # α = ρ / (p ⋅ q) 
        α = ρ / dot(solver.search_direction, solver.linear_operator_output)

        @debug "PreconditionedConjugateGradientSolver $iteration, α: $α"
        
        @. x += α * p
        @. r -= α * q

        iteration += 1
        ρⁱ⁻¹ = ρ
    end

    solution = x

    fill_halo_regions!(solution, arch)

    return nothing
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "Oceanigans compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , size(solver.settings.q) )
    print(io, "\n Grid = "  , solver.settings.grid  )
    return nothing
end
