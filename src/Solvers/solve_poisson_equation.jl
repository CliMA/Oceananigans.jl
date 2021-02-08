"""
    solve_poisson_equation!(solver)

Solves Poisson's equation ∇²ϕ = RHS where `RHS = solver.storage` using periodic or staggered
Neumann boundary conditions as determined by the `solver.grid`. `solver.storage` will be mutated
to contain the solution.
"""
function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS = ϕ = solver.storage

    # Apply forward transforms in order
    [transform!(RHS, solver.buffer) for transform! in solver.transforms.forward]

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Set the volume mean of the solution to be zero.
    # λx[1, 1, 1] + λy[1, 1, 1] + λz[1, 1, 1] = 0 so if ϕ[1, 1, 1] = 0 we get NaNs
    # everywhere after the inverse transform. In eigenspace, ϕ[1, 1, 1] is the
    # "zeroth mode" corresponding to the volume mean of the transform of ϕ, or of ϕ
    # in physical space.
    # Another way of thinking about this: Solutions to Poisson's equation are only
    # unique up to a constant (the global mean of the solution), so we need to pick
    # a constant. ϕ[1, 1, 1] = 0 chooses the constant to be zero so that the solution
    # has zero-mean.
    CUDA.@allowscalar ϕ[1, 1, 1] = 0

    # Apply backward transforms in order
    [transform!(RHS, solver.buffer) for transform! in solver.transforms.backward]

    return nothing
end
