import FFTW

# Solve a 1D Poisson equation ∇²ϕ = d²ϕ/dx² = f(x) with periodic boundary
# conditions and grid spacing Δx using a second-order Fourier-spectral method.
# Right now the solver expects an odd number of points in f where the last point
# is the periodic point and equal to the first
function solve_poisson_1d_pbc(f, Δx)
    N = length(f) - 1  # Number of grid points (excluding the periodic end point).
    n = 0:N            # Wavenumber indices.
    k² = @. (4 / Δx^2) * sin(π*n / N)^2  # Wavenumber squared.

    # Forward transform the real-valued source term.
    fh = FFTW.rfft(f)

    # Calculate the Fourier coefficients of the source term.
    # We only need to compute the first (N-1)/2 + 1 Fourier coefficients
    # as ϕh(N-i) = ϕh(i) for a real-valued f.
    ϕh = - fh ./ k²[1:Int(N/2 + 1)]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1] = 0

    # Take the inverse transform of the . We need to specify that the input f
    # had a length of N+1 as
    ϕ = FFTW.irfft(ϕh, N+1)
end
