normalization_factor(::CPU, ::Periodic, N) = 1
normalization_factor(::CPU, ::Bounded, N) = 1/(2N)

function solve_poisson_equation!(solver)
    topo = TX, TY, TZ = topology(solver.grid)
    kx², ky², kz² = solver.wavenumbers

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Apply forward transforms
    solver.transforms.forward.periodic * RHS
    solver.transforms.forward.bounded * RHS

    # Solve the discrete Poisson equation.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    # Apply backward transforms
    solver.transforms.backward.periodic * RHS
    solver.transforms.backward.bounded * RHS

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    factor = prod(normalization_factor(arch, T(), N) for (T, N) in zip(topo, size(grid)))
    @. ϕ = ϕ / factor

    return nothing
end