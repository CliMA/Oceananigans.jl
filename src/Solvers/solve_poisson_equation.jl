# TODO: Move to transforms.jl
normalization_factor(arch, topo, N) = 1
normalization_factor(::CPU, ::Bounded, N) = 1/(2N)

function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    λx, λy, λz = solver.eigenvalues

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms
    solver.transforms.forward.periodic(RHS)
    solver.transforms.forward.bounded(RHS)

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (λx + λy + λz)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    # Apply backward transforms
    solver.transforms.backward.bounded(ϕ)
    solver.transforms.backward.periodic(ϕ)

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    factor = prod(normalization_factor(solver.architecture, T(), N) for (T, N) in zip(topo, size(solver.grid)))
    @. ϕ = factor * ϕ

    return nothing
end