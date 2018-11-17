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

function test_solve_poisson_1d_pbc_cosine_source_multiple_resolutions()
    Ns = [4, 8, 10, 50, 100, 500, 1000, 2000, 5000, 10000]

    for N in Ns
        H = 12      # Length of domain.
        Δz = H / N  # Grid spacing.
        z = Δz * (0:(N-1))  # Grid point locations.

        f = cos.(2*π*z ./ H)  # Source term.
        ϕa = @. -(H / (2π))^2 * cos((2π / H) * z)  # Analytic solution.

        ϕs = solve_poisson_1d_pbc(f, H)

        # max_error should be ≈ machine epsilon but we're giving it a bit of
        # leeway here.
        max_error = maximum(abs.(ϕs - ϕa))
        if max_error > 1e-14
            return false
        end
    end
    true
end
