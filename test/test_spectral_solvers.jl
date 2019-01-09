using Statistics: mean

using FFTW

# Testing with the cos(2πz/H) source term should give a spectral solution that
# is numerically accurate within ≈ machine epsilon, so in these tests we can
# use ≈ to check the spectral solution with the analytic solution.
#
# With the source term f(z) = cos(2πz/H), the analytic solution is given by
# ϕ(z) = -(H/2π)² cos(2πz/H) and is periodic as ϕ(0) = ϕ(H).

function test_solve_poisson_1d_pbc_cosine_source(N)
    H = 12       # Length of domain.
    Nz = N       # Number of grid points.
    Δz = H / Nz  # Grid spacing.
    z = Δz * (0:(Nz-1))  # Grid point locations.

    f = cos.(2*π*z ./ H)  # Source term.
    ϕa = @. -(H / (2π))^2 * cos((2π / H) * z)  # Analytic solution.

    ϕs = solve_poisson_1d_pbc(f, H, :analytic)

    ϕs ≈ ϕa
end

function test_solve_poisson_1d_pbc_divergence_free(N)
    laplacian1d(f) = circshift(f, 1) - 2 .* f + circshift(f, -1)

    A = rand(N)
    A .= A .- mean(A)
    B = solve_poisson_1d_pbc(A, N, :second_order)
    A′ = laplacian1d(B)
    A ≈ A′
end

# Testing with the exp(-(x²+y²)) Gaussian source term won't we do see spectral
# convergence, however, I'm not sure how to best normalize the solution to see
# this. So for now I'm just testing to make sure that the maximum error is 1e-5.
#
# With the source term f(x,y) = 4(x²+y²-1)exp(-(x²+y²)), the analytic solution
# is given by ϕ(x,y) = exp(-(x²+y²)) which I suppose is "numerically periodic"
# if it decays to zero pretty quickly at the boundaries.

function test_solve_poisson_2d_pbc_gaussian_source(Nx, Ny)
    Lx, Ly = 8, 8;  # Domain size.
    Δx, Δy = Lx/Nx, Ly/Ny;  # Grid spacing.

    # Grid point locations.
    x = Δx * (0:(Nx-1));
    y = Δy * (0:(Ny-1));

    # Primed coordinates to easily calculate a Gaussian centered at
    # (Lx/2, Ly/2).
    x′ = @. x - Lx/2;
    y′ = @. y - Ly/2;

    f = @. 4 * (x′^2 + y′'^2 - 1) * exp(-(x′^2 + y′'^2))  # Source term
    f .= f .- mean(f)  # Ensure that source term integrates to zero.

    ϕa = @. exp(-(x′^2 + y′'^2))  # Analytic solution

    ϕs = solve_poisson_2d_pbc(f, Lx, Ly, :analytic)

    # Choosing the solution that integrates out to zero.
    ϕs = ϕs .- minimum(ϕs)

    maximum(abs.(ϕs - ϕa)) < 1e-5
end

function test_solve_poisson_2d_pbc_divergence_free(N)
    laplacian2d(f) = circshift(f, (1, 0)) - 2 .* f + circshift(f, (-1, 0)) + circshift(f, (0, 1)) - 2 .* f + circshift(f, (0, -1))

    A = rand(N, N)
    A .= A .- mean(A)
    B = solve_poisson_2d_pbc(A, N, N, :second_order)
    A′ = laplacian2d(B)
    A ≈ A′
end

function test_mixed_fft_commutativity(N)
    A = rand(N, N, N)
    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])
    Ã1 ≈ Ã2
end

function test_mixed_ifft_commutativity(N)
    A = rand(N, N, N)

    Ã1 = FFTW.dct(FFTW.rfft(A, [1, 2]), 3)
    Ã2 = FFTW.rfft(FFTW.dct(A, 3), [1, 2])

    A11 = FFTW.irfft(FFTW.idct(Ã1, 3), N, [1, 2])
    A12 = FFTW.idct(FFTW.irfft(Ã1, N, [1, 2]), 3)
    A21 = FFTW.irfft(FFTW.idct(Ã2, 3), N, [1, 2])
    A22 = FFTW.idct(FFTW.irfft(Ã2, N, [1, 2]), 3)
    A ≈ A11 && A ≈ A12 && A ≈ A21 && A ≈ A22
end

# Testing with the exp(-(x²+y²)) Gaussian source term won't we do see spectral
# convergence, however, I'm not sure how to best normalize the solution to see
# this. So for now I'm just testing to make sure that the maximum error is 1e-5.
#
# With the source term f(x,y,z) = 4(x²+y²-1-π²/Lz²)cos(2πz/Lz)exp(-(x²+y²)), the
# analytic solution is given by ϕ(x,y,z) = cos(2πz/Lz)exp(-(x²+y²)) which I
# suppose is "numerically periodic" if it decays to zero pretty quickly at the
# boundaries.

function test_solve_poisson_3d_mbc_gaussian_cosine_source()
end


function test_3d_poisson_solver_ppn_div_free(Nx, Ny, Nz)
    f = rand(Nx, Ny, Nz)
    f .= f .- mean(f)
    ϕ = solve_poisson_3d_ppn(f, Nx, Ny, Nz, 1, 1, 1)
    laplacian3d_ppn(ϕ) ≈ f
end

function test_3d_poisson_solver_ppn!_div_free(Nx, Ny, Nz)
    if Nx == 1 || Ny == 1
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=2)
    else
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=3)
    end

    RHS = CellField(g, Complex{eltype(g)})
    RHS_orig = CellField(g, Complex{eltype(g)})
    ϕ = CellField(g, Complex{eltype(g)})
    ∇²ϕ = CellField(g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    solve_poisson_3d_ppn!(g, RHS, ϕ)

    ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

function test_fftw_planner(Nx, Ny, Nz, planner_flag)
    if Nx == 1 || Ny == 1
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=2)
    else
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=3)
    end

    RHS = CellField(g, Complex{eltype(g)})
    ssp = SpectralSolverParameters(g, RHS, FFTW.PATIENT)
    true  # Just making sure our SpectralSolverParameters does not spit an error.
end

function test_3d_poisson_ppn_planned!_div_free(Nx, Ny, Nz, planner_flag)
    if Nx == 1 || Ny == 1
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=2)
    else
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=3)
    end

    RHS = CellField(g, Complex{eltype(g)})
    RHS_orig = CellField(g, Complex{eltype(g)})
    ϕ = CellField(g, Complex{eltype(g)})
    ∇²ϕ = CellField(g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    ssp = SpectralSolverParameters(g, RHS, planner_flag)

    solve_poisson_3d_ppn_planned!(ssp, g, RHS, ϕ)
    ∇²_ppn!(g, ϕ, ∇²ϕ)

    ∇²ϕ.data ≈ RHS_orig.data
end

function test_3d_poisson_solver_ppn_all_equal(Nx, Ny, Nz, planner_flag)
    if Nx == 1 || Ny == 1
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=2)
    else
        g = RegularCartesianGrid((Nx, Ny, Nz), (100, 100, 100); dim=3)
    end

    RHS = CellField(g, Complex{eltype(g)})
    RHS_orig = CellField(g, Complex{eltype(g)})
    ϕ = CellField(g, Complex{eltype(g)})
    ∇²ϕ = CellField(g, Complex{eltype(g)})

    RHS.data .= rand(Nx, Ny, Nz)
    RHS.data .= RHS.data .- mean(RHS.data)

    RHS_orig.data .= copy(RHS.data)

    ssp = SpectralSolverParameters(g, RHS, planner_flag)
end
