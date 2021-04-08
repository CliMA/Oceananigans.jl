using Oceananigans.Grids: interior_parent_indices
using Statistics

struct PreconditionedConjugateGradientSolver{A, G, L, M, T}
       architecture :: A
               grid :: G
    linear_operator :: L
     preconditioner :: M
          tolerance :: T
    maximum_iterations :: Int
           residual :: R
           settings :: S
end

identity_preconditioner(x, args...) = x

function PreconditionedConjugateGradientSolver(;
                                               architecture,
                                               grid,
                                               solution,
                                               max_iterations = size(grid),
                                               tolerance = 1e-13,
                                               preconditioner = identity_preconditioner,
                                               parameters = parameters)

    a_residual = similar(solution)
    residual = similar(solution)
    search_direction = similar(solution)

    q     = similar(solution) #.data)
    p     = similar(solution) #.data)
    z     = similar(solution) #.data)
    r     = similar(solution) #.data)
    RHS   = similar(solution) #.data)
    x     = similar(solution) #.data)

    parent(residual) .= 0

    ii = interior_parent_indices(location(tf, 1), topology(grid, 1), grid.Nx, grid.Hx)
    ji = interior_parent_indices(location(tf, 2), topology(grid, 2), grid.Ny, grid.Hy)
    ki = interior_parent_indices(location(tf, 3), topology(grid, 3), grid.Nz, grid.Hz)

    dotproduct(x, y) = mapreduce((x, y) -> x * y, +, view(x, ii, ji, ki), view(y, ii, ji, ki))
    norm(x) = sqrt(mapreduce(x -> x * x, +, view(x, ii, ji, ki)))

    Amatrix_function = parameters.Amatrix_function
    A(x, args...) = (Amatrix_function(a_residual, x, arch, args...); return a_residual)

    reference_pressure_solver = nothing
    if haskey(parameters, :reference_pressure_solver )
        reference_pressure_solver = parameters.reference_pressure_solver
    end

    settings = (
                                q = q,
                                p = p,
                                z = z,
                                r = r,
                                x = x,
                              RHS = RHS,
                              bcs = bcs,
                             grid = grid,
                                A = A,
                                M = M,
                            maxit = maxit,
                              tol = tol,
                          dotprod = dotproduct,
                             norm = norm,
                             arch = arch,
        reference_pressure_solver = reference_pressure_solver,
    )

   return PreconditionedConjugateGradientSolver(arch, settings)
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
    * Linear Preconditioner operator M as a function M();
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
    settings   = solver.settings
    z, r, p, q = settings.z, settings.r, settings.p, settings.q
    A          = settings.A
    M          = settings.M
    maxit      = settings.maxit
    tol        = settings.tol
    dotprod    = settings.dotprod
    norm       = settings.norm

    # Initialize
    β    = 0.0
    iteration = 0
    ρ    = 0.0
    ρⁱ⁻¹ = 0.0

    x = initial_guess
    r = parent(solver.residual)
    p = parent(solver.search_direction)
    z = parent(solver.preconditioned_residual)

    r .= b .- A(x, args...)

    @debug "PreconditionedConjugateGradientSolver $iteration, |b|: $(norm(b))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |A(x)|: $(norm(A(initial_guess, args...)))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |r|: $(norm(r))"
    @debug "PreconditionedConjugateGradientSolver $iteration, |z|: $(norm(z))"
    @debug "PreconditionedConjugateGradientSolver $iteration, ρ: $ρ"
    @debug "PreconditionedConjugateGradientSolver $iteration, |q|: $(norm(q))"
    @debug "PreconditionedConjugateGradientSolver $iteration, α: $α"

    while true
        # End conditions
        iteration >= solver.maximum_iterations && break
        norm(r) <= solver.tolerance && break

        # z = M(r)
        z = M(solver.residual, args...)
        ρ = dotprod(z, r)

        if iteration == 0
            p .= z
        else
            β = ρ / ρⁱ⁻¹
            @. p = z + β * p
        end

        # q = A(p)
        q .= parent(A(solver.search_direction, args...))
        α = ρ / dotprod(p, q)
        
        @. x += α * p
        @. r += α * q

        iteration += 1
        ρⁱ⁻¹  = ρ
    end

    solution = initial_guess

    fill_halo_regions!(solution, arch)

    return nothing
end

function Base.show(io::IO, solver::PreconditionedConjugateGradientSolver)
    print(io, "Oceanigans compatible preconditioned conjugate gradient solver.\n")
    print(io, " Problem size = "  , size(solver.settings.q) )
    print(io, "\n Grid = "  , solver.settings.grid  )
    return nothing
end
