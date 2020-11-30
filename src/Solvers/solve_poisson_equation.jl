function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms
    solver.transforms.forward.bounded(RHS)
    solver.transforms.forward.periodic(RHS)

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    CUDA.@allowscalar ϕ[1, 1, 1] = 0

    # Apply backward transforms
    solver.transforms.backward.periodic(ϕ)
    solver.transforms.backward.bounded(ϕ)

    return nothing
end