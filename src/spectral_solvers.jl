import FFTW

# Solve a 1D Poisson equation ∇²ϕ = d²ϕ/dx² = f(x) with periodic boundary
# conditions and domain length L using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_1d_pbc(f, L, wavenumbers=:second_order)
    N = length(f)  # Number of grid points (excluding the periodic end point).
    n = 0:N        # Wavenumber indices.

    if wavenumbers == :second_order
        # Wavenumbers if Laplacian is discretized using second-order
        # centered-difference scheme. Gives second-order convergence and ensures
        # that ∇²ϕ == f(x) to machine precision.
        Δx = L / N
        k² = @. (4 / Δx^2) * sin(π*n / N)^2
    elseif wavenumbers == :analytic
        # Wavenumbers if the derivatives are not discretized, should give
        # spectral convergence so that ϕ is accurate but ∇²ϕ ≈ f(x) as the ∇²
        # operator must be discretized.
        k² = @. ((2*π / L) * n)^2
    end

    # Forward transform the real-valued source term.
    fh = FFTW.rfft(f)

    # Calculate the Fourier coefficients of the source term.
    # We only need to compute the first (N-1)/2 + 1 Fourier coefficients
    # as ϕh(N-i) = ϕh(i) for a real-valued f.
    ϕh = - fh ./ k²[1:Int(N/2 + 1)]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1] = 0

    # Take the inverse transform of the . We need to specify that the input f
    # had a length of N as an rrft of length N/2 may have come from an array
    # of length N (if N is even) or N-1 (if N is odd).
    ϕ = FFTW.irfft(ϕh, N)
end

# Solve a 2D Poisson equation ∇²ϕ = f(x,y) with periodic boundary conditions and
# domain lengths Lx, Ly using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_2d_pbc(f, Lx, Ly)
    Nx, Ny = size(f)  # Number of grid points (excluding the periodic end point).

    # Forward transform the real-valued source term.
    fh = FFTW.rfft(f)

    # Wavenumbers. Can't figure out why this didn't work...
    # l, m = 0:Nx, 0:Ny  # Wavenumber indices.
    # kx² = @. ((2*π / Lx) * l)^2
    # ky² = @. ((2*π / Ly) * m)^2
    # ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :]; ϕh[1, 1] = 0; ϕh[1, end] = 0; ϕh

    # Wavenumbers that work.
    l1 = 0:Int(Nx/2)
    l2 = Int(-Nx/2 + 1):-1
    m1 = 0:Int(Ny/2)
    m2 = Int(-Ny/2 + 1):-1
    kx = reshape((2π/Lx) * cat(l1, l2, dims=1), (Nx, 1))
    ky = reshape((2π/Ly) * cat(m1, m2, dims=1), (1, Ny))
    k² = @. kx^2 + ky^2

    ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(ϕh, Nx)
end

# Solve a 3D Poisson equation ∇²ϕ = f(x,y,z) with periodic boundary conditions
# and domain lengths Lx, Ly, Lz using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_3d_pbc(f, Lx, Ly, Lz)
    Nx, Ny, Nz = size(f)  # Number of grid points (excluding the periodic end point).

    # Forward transform the real-valued source term.
    fh = FFTW.rfft(f)

    # Wavenumber indices.
    l1 = 0:Int(Nx/2)
    l2 = Int(-Nx/2 + 1):-1
    m1 = 0:Int(Ny/2)
    m2 = Int(-Ny/2 + 1):-1
    n1 = 0:Int(Nz/2)
    n2 = Int(-Nz/2 + 1):-1

    kx = reshape((2π/Lx) * cat(l1, l2, dims=1), (Nx, 1, 1))
    ky = reshape((2π/Ly) * cat(m1, m2, dims=1), (1, Ny, 1))
    kz = reshape((2π/Ly) * cat(n1, n2, dims=1), (1, 1, Nz))

    k² = @. kx^2 + ky^2 + kz^2

    ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :, :]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(ϕh, Nx)
end

# Solve a 3D Poisson equation ∇²ϕ = f(x,y,z) with periodic boundary conditions
# in the horizontal directions (x,y) and Neumann boundary conditions in the
# z-direction using the Fourier-spectral method.
#
# This corresponds to representing the solution with complex exponentials in the
# horizontal and cosines in the vertical so that a discrete Fourier transform
# (DFT) is employed in forward transforming the source term and inverse
# transforming the solution in the horizontal, while a discrete cosine transform
# (DCT) is emplored in the vertical.
#
# The domain lengths Lx, Ly, Lz are required to calculate the wavenumbers.
# Solutions to Poisson's equation with periodic boundary conditions are unique
# up to a constant so you may need to appropriately normalize the solution if
# you careabout the numerical value of the solution itself and not just
# derivatives of the solution.
function solve_poisson_3d_mbc(f, Lx, Ly, Lz)
    Nx, Ny, Nz = size(f)  # Number of grid points (excluding the periodic end point).

    # Forward transform the real-valued source term.
    fh = FFTW.dct(FFTW.rfft(f, [1, 2]), 3)

    # Wavenumber indices.
    l1 = 0:Int(Nx/2)
    l2 = Int(-Nx/2 + 1):-1
    m1 = 0:Int(Ny/2)
    m2 = Int(-Ny/2 + 1):-1
    n1 = 0:Int(Nz/2)
    n2 = Int(-Nz/2 + 1):-1

    kx = reshape((2π/Lx) * cat(l1, l2, dims=1), (Nx, 1, 1))
    ky = reshape((2π/Ly) * cat(m1, m2, dims=1), (1, Ny, 1))
    kz = reshape((1π/Ly) * cat(n1, n2, dims=1), (1, 1, Nz))

    k² = @. kx^2 + ky^2 + kz^2

    ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :, :]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(FFTW.idct(ϕh, 3), Nx, [1, 2])
end
