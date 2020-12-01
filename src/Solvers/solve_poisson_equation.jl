function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms
    if :bounded in keys(solver.transforms.forward)
        solver.transforms.forward.bounded(RHS, solver.buffer)
        solver.transforms.forward.periodic(RHS, solver.buffer)
    elseif :x in keys(solver.transforms.forward)
        # Do DCT first
        solver.transforms.forward.z(RHS, solver.buffer)
        solver.transforms.forward.y(RHS, solver.buffer)
        solver.transforms.forward.x(RHS, solver.buffer)
    end

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    CUDA.@allowscalar ϕ[1, 1, 1] = 0

    # Apply backward transforms
    if :bounded in keys(solver.transforms.backward)
        solver.transforms.backward.periodic(ϕ, solver.buffer)
        solver.transforms.backward.bounded(ϕ, solver.buffer)
    elseif :x in keys(solver.transforms.backward)
        # Do DCT last
        solver.transforms.backward.x(ϕ, solver.buffer)
        solver.transforms.backward.y(ϕ, solver.buffer)
        solver.transforms.backward.z(ϕ, solver.buffer)
    end

    return nothing
end