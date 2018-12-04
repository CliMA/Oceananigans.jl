import FFTW

# Solve a 1D Poisson equation ∇²ϕ = d²ϕ/dx² = f(x) with periodic boundary
# conditions and domain length L using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_1d_pbc(f, L, wavenumbers)
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

# Solve a 1D Poisson equation ∇²ϕ = d²ϕ/dx² = f(x) with Neumann boundary
# conditions and domain length L using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_1d_nbc(f, L, wavenumbers)
    @show N = length(f)  # Number of grid points (excluding the periodic end point).
    n = 0:(N-1)        # Wavenumber indices.

    if wavenumbers == :second_order
        # Wavenumbers for second-order convergence.
        Δx = L / N
        k² = @. (2 / Δx^2) * (cos(π*n / N) - 1)
    elseif wavenumbers == :analytic
        # Wavenumbers for spectral convergence.
        k² = @. ((1*π / L) * n)^2
    end

    # Forward transform the real-valued source term.
    fh = FFTW.dct(f)

    # Calculate the Fourier coefficients of the source term.
    # We only need to compute the first (N-1)/2 + 1 Fourier coefficients
    # as ϕh(N-i) = ϕh(i) for a real-valued f.
    ϕh = fh ./ k²

    # Setting the DC/zero Fourier component to zero.
    ϕh[1] = 0

    # Take the inverse transform of the . We need to specify that the input f
    # had a length of N+1 as
    ϕ = FFTW.idct(ϕh)
end

# Solve a 2D Poisson equation ∇²ϕ = f(x,y) with periodic boundary conditions and
# domain lengths Lx, Ly using the Fourier-spectral method. Solutions to
# Poisson's equation with periodic boundary conditions are unique up to a
# constant so you may need to appropriately normalize the solution if you care
# about the numerical value of the solution itself and not just derivatives of
# the solution.
function solve_poisson_2d_pbc(f, Lx, Ly, wavenumbers)
    Nx, Ny = size(f)  # Number of grid points (excluding the periodic end point).

    # Forward transform the real-valued source term.
    fh = FFTW.rfft(f)

    # Wavenumber indices.
    l1 = 0:Int(Nx/2)
    l2 = Int(-Nx/2+1):-1
    m1 = 0:Int(Ny/2)
    m2 = Int(-Ny/2+1):-1

    if wavenumbers == :second_order
        Δx = Lx/Nx
        Δy = Ly/Ny
        kx² = reshape((4/Δx^2) .* sin.( (π/Nx) .* cat(l1, l2, dims=1)).^2, (Nx, 1))
        ky² = reshape((4/Δy^2) .* sin.( (π/Ny) .* cat(m1, m2, dims=1)).^2, (1, Ny))
        k² = @. kx² + ky²
    elseif wavenumbers == :analytic
        kx = reshape((2π/Lx) * cat(l1, l2, dims=1), (Nx, 1))
        ky = reshape((2π/Ly) * cat(m1, m2, dims=1), (1, Ny))
        k² = @. kx^2 + ky^2
    end

    ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(ϕh, Nx)
end

function solve_poisson_2d_mbc(f, Lx, Ly, wavenumbers=:second_order)
    Nx, Ny = size(f)  # Number of grid points (excluding the periodic end point).

    # Forward transform the real-valued source term.
    fh = FFTW.dct(FFTW.rfft(f, 1), 2)

    # Wavenumber indices.
    i1 = 0:Int(Nx/2)
    i2 = Int(-Nx/2+1):-1
    j1 = 0:Int(Ny/2)
    j2 = Int(-Ny/2+1):-1

    if wavenumbers == :second_order
        Δx = Lx/Nx
        Δy = Ly/Ny
        kx² = reshape((4/Δx^2) .* sin.( (π/Nx) .* cat(i1, i2, dims=1)).^2, (Nx, 1))
        ky² = reshape((2/Δz^2) .* (cos.( (π/Nz) .* (1:(Nz-1))) .- 1), (1, Ny))
        k² = @. kx² + ky²
        ϕh = fh ./ k²[1:Int(Nx/2 + 1), :]
    elseif wavenumbers == :analytic
        kx = reshape((2π/Lx) * cat(i1, i2, dims=1), (Nx, 1))
        ky = reshape((1π/Ly) * cat(j1, j2, dims=1), (1, Ny))
        k² = @. kx^2 + ky^2
        ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :]
    end

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(FFTW.idct(ϕh, 2), Nx, 1)
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

function solve_poisson_3d_mbc(f, Lx, Ly, Lz, wavenumbers)
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

    if wavenumbers == :second_order
        kx² = reshape((4/Δx^2) .* sin.( (π/Nx) .* cat(l1, l2, dims=1)).^2, (Nx, 1, 1))
        ky² = reshape((4/Δy^2) .* sin.( (π/Ny) .* cat(m1, m2, dims=1)).^2, (1, Ny, 1))
        kz² = reshape((4/Δz^2) .* sin.( (π/(2*Nz) .* 0:(Nz-1))).^2, (1, 1, Nz))

        k² = @. kx² + ky² + kz²
    elseif wavenumbers == :analytic
        kx = reshape((2π/Lx) * cat(l1, l2, dims=1), (Nx, 1, 1))
        ky = reshape((2π/Ly) * cat(m1, m2, dims=1), (1, Ny, 1))
        kz = reshape((1π/Ly) * cat(n1, n2, dims=1), (1, 1, Nz))
        k² = @. kx^2 + ky^2 + kz^2
    end

    ϕh = - fh ./ k²[1:Int(Nx/2 + 1), :, :]

    # Setting the DC/zero Fourier component to zero.
    ϕh[1, 1, 1] = 0

    # Take the inverse transform of the solution's Fourier coefficients.
    ϕ = FFTW.irfft(FFTW.idct(ϕh, 3), Nx, [1, 2])
end

function solve_poisson_3d_ppn(f, Lx, Ly, Lz, wavenumbers)
    # Solve FFT style
    function mkwaves(N,L)
        k²_cyc = zeros(N,1)
        k²_neu = zeros(N,1)
        for i in 1:N
            k²_cyc[i] = (2sin((i-1) * π/N)   / (L/N))^2
            k²_neu[i] = (2sin((i-1) * π/(2N))/ (L/N))^2
        end
        return k²_cyc, k²_neu
    end

    # fh′ = FFTW.r2r(reshape(f,Nx,Ny,Nz),FFTW.REDFT10,3)
    fh′ = FFTW.r2r(f, FFTW.REDFT10, 3)
    fh  = FFTW.fft(fh′, [1, 2])

    kx²_cyc, kx²_neu = mkwaves(Nx, Lx)
    ky²_cyc, ky²_neu = mkwaves(Nx, Lx)
    kz²_cyc, kz²_neu = mkwaves(Nx, Lx)
    kx² = kx²_cyc
    ky² = ky²_cyc
    kz² = kz²_neu

    ϕh = -fh

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ϕh[i, j, k] = fh[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕh[1, 1, 1] = 0

    ϕ′ = FFTW.ifft(ϕh, [1, 2])
    ϕ  = FFTW.r2r(real.(ϕ′), FFTW.REDFT01, 3) / (2Nz)
end
