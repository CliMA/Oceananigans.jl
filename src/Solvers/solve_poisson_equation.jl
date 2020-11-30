normalization_factor(::CPU, ::Periodic, N) = 1
normalization_factor(::CPU, ::Bounded, N) = 1/(2N)

function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms
    if !isnothing(solver.transforms.forward.periodic)
        solver.transforms.forward.periodic * RHS
    end

    if !isnothing(solver.transforms.forward.bounded)
        solver.transforms.forward.bounded * RHS
    end

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    # Apply backward transforms
    if !isnothing(solver.transforms.backward.bounded)
        solver.transforms.backward.bounded * ϕ
    end

    if !isnothing(solver.transforms.backward.periodic)
        solver.transforms.backward.periodic * ϕ
    end

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    factor = prod(normalization_factor(solver.architecture, T(), N) for (T, N) in zip(topo, size(solver.grid)))
    @. ϕ = factor * ϕ

    return nothing
end