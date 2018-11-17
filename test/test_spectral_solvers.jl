function test_solve_poisson_1d_pbc_cosine_source()
    H = 12           # Length of domain.
    Nz = 100         # Number of grid points.
    Δz = H / Nz      # Grid spacing.
    z = Δz * (0:(Nz-1))  # Grid point locations.

    f = cos.(2*π*z ./ H)  # Source term.
    ϕa = @. -(H / (2π))^2 * cos((2π / H) * z)  # Analytic solution.

    ϕs = solve_poisson_1d_pbc(f, H)

    ϕs ≈ ϕa  # Should have machine precision due to spectral convergence.
end
end
