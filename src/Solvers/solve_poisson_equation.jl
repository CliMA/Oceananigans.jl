function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms in order
    [transform!(RHS, solver.buffer) for transform! in solver.transforms.forward]

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    CUDA.@allowscalar ϕ[1, 1, 1] = 0

    # Apply backward transforms in order
    [transform!(RHS, solver.buffer) for transform! in solver.transforms.backward]

    return nothing
end
