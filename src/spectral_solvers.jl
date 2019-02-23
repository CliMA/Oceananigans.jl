import FFTW
using GPUifyLoops

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

function solve_poisson_3d_ppn(f, Nx, Ny, Nz, Δx, Δy, Δz)
    Lx, Ly, Lz = Nx*Δx, Ny*Δy, Nz*Δz

    function mkwaves(N,L)
        k²_cyc = zeros(N, 1)
        k²_neu = zeros(N, 1)

        for i in 1:N
            k²_cyc[i] = (2sin((i-1)*π/N)   /(L/N))^2
            k²_neu[i] = (2sin((i-1)*π/(2N))/(L/N))^2
        end

        return k²_cyc, k²_neu
    end

    fh = FFTW.fft(FFTW.r2r(f, FFTW.REDFT10, 3), [1, 2])

    kx²_cyc, kx²_neu = mkwaves(Nx, Lx)
    ky²_cyc, ky²_neu = mkwaves(Ny, Ly)
    kz²_cyc, kz²_neu = mkwaves(Nz, Lz)

    kx² = kx²_cyc
    ky² = ky²_cyc
    kz² = kz²_neu

    ϕh = zeros(Complex{Float64}, Nx, Ny, Nz)

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds ϕh[i, j, k] = -fh[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕh[1, 1, 1] = 0

    FFTW.r2r(real.(FFTW.ifft(ϕh, [1, 2])), FFTW.REDFT01, 3) / (2Nz)
end

function solve_poisson_3d_ppn!(g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    kx² = zeros(g.Nx, 1)
    ky² = zeros(g.Ny, 1)
    kz² = zeros(g.Nz, 1)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    FFTW.r2r!(f.data, FFTW.REDFT10, 3)
    FFTW.fft!(f.data, [1, 2])

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
    end
    ϕ.data[1, 1, 1] = 0

    FFTW.ifft!(ϕ.data, [1, 2])

    @. ϕ.data = real(ϕ.data) / (2g.Nz)
    # for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
    #     ϕ[i, j, k] = real(ϕ[i, j, k])
    # end

    FFTW.r2r!(ϕ.data, FFTW.REDFT01, 3)

    nothing
end

struct SpectralSolverParameters{T<:AbstractArray}
    kx²::T
    ky²::T
    kz²::T
    FFT!
    DCT!
    IFFT!
    IDCT!
end

let pf2s = Dict(FFTW.ESTIMATE   => "FFTW.ESTIMATE",
                FFTW.MEASURE    => "FFTW.MEASURE",
                FFTW.PATIENT    => "FFTW.PATIENT",
                FFTW.EXHAUSTIVE => "FFTW.EXHAUSTIVE")
    global plannerflag2string
    plannerflag2string(k::Integer) = pf2s[Int(k)]
end

function SpectralSolverParameters(g::Grid, exfield::CellField, planner_flag=FFTW.PATIENT; verbose=false)
    kx² = zeros(eltype(g), g.Nx)
    ky² = zeros(eltype(g), g.Ny)
    kz² = zeros(eltype(g), g.Nz)

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    if verbose
        print("Planning Fourier transforms... (planner_flag=$(plannerflag2string(planner_flag)))\n")
        print("FFT!:  "); @time FFT!  = FFTW.plan_fft!(exfield.data, [1, 2]; flags=planner_flag)
        print("IFFT!: "); @time IFFT! = FFTW.plan_ifft!(exfield.data, [1, 2]; flags=planner_flag)
        print("DCT!:  "); @time DCT!  = FFTW.plan_r2r!(exfield.data, FFTW.REDFT10, 3; flags=planner_flag)
        print("IDCT!: "); @time IDCT! = FFTW.plan_r2r!(exfield.data, FFTW.REDFT01, 3; flags=planner_flag)
    else
        FFT!  = FFTW.plan_fft!(exfield.data, [1, 2]; flags=planner_flag)
        IFFT! = FFTW.plan_ifft!(exfield.data, [1, 2]; flags=planner_flag)
        DCT!  = FFTW.plan_r2r!(exfield.data, FFTW.REDFT10, 3; flags=planner_flag)
        IDCT! = FFTW.plan_r2r!(exfield.data, FFTW.REDFT01, 3; flags=planner_flag)
    end

    SpectralSolverParameters{Array{eltype(g),1}}(kx², ky², kz², FFT!, DCT!, IFFT!, IDCT!)
end

function solve_poisson_3d_ppn_planned!(ssp::SpectralSolverParameters, g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    ssp.DCT!*f.data  # Calculate DCTᶻ(f) in place.
    ssp.FFT!*f.data  # Calculate FFTˣʸ(f) in place.

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (ssp.kx²[i] + ssp.ky²[j] + ssp.kz²[k])
    end
    ϕ.data[1, 1, 1] = 0

    ssp.IFFT!*ϕ.data  # Calculate IFFTˣʸ(ϕ) in place.
    ssp.IDCT!*ϕ.data  # Calculate IDCTᶻ(ϕ) in place.
    @. ϕ.data = ϕ.data / (2*g.Nz)
    nothing
end

function dct_dim3_gpu!(g, f, dct_factors)
    # Nx, Ny, Nz = size(f)
    # This is now done in time_step_kernel_part3!.
    # f .= cat(f[:, :, 1:2:g.Nz], f[:, :, g.Nz:-2:2]; dims=3)
    fft!(f, 3)

    # factors = 2 * exp.(collect(-1im*π*(0:Nz-1) / (2*Nz)))
    # f .*= cu(repeat(reshape(factors, 1, 1, Nz), Nx, Ny, 1))
    f .*= dct_factors

    nothing
end

function idct_dim3_gpu!(g, f, idct_bfactors)
    # Nx, Ny, Nz = size(f)

    # bfactors = exp.(collect(1im*π*(0:Nz-1) / (2*Nz)))
    # bfactors[1] *= 0.5

    # f .*= cu(repeat(reshape(bfactors, 1, 1, Nz), Nx, Ny, 1))

    f .*= idct_bfactors
    ifft!(f, 3)

    # Both these steps have been merged into idct_permute! in the time-stepping loop.
    # f .= CuArray{eltype(f)}(reshape(permutedims(cat(f[:, :, 1:Int(g.Nz/2)], f[:, :, end:-1:Int(g.Nz/2)+1]; dims=4), (1, 2, 4, 3)), g.Nx, g.Ny, g.Nz))
    # @. f = real(f)  # Don't do it here. We'll do it when assigning real(ϕ) to pNHS to save some measly FLOPS.

    nothing
end

function solve_poisson_3d_ppn_gpu!(g::RegularCartesianGrid, f::CellField, ϕ::CellField)
    kx² = cu(zeros(g.Nx, 1))
    ky² = cu(zeros(g.Ny, 1))
    kz² = cu(zeros(g.Nz, 1))

    for i in 1:g.Nx; kx²[i] = (2sin((i-1)*π/g.Nx)    / (g.Lx/g.Nx))^2; end
    for j in 1:g.Ny; ky²[j] = (2sin((j-1)*π/g.Ny)    / (g.Ly/g.Ny))^2; end
    for k in 1:g.Nz; kz²[k] = (2sin((k-1)*π/(2g.Nz)) / (g.Lz/g.Nz))^2; end

    print("FFT!  "); @time fft!(f.data, [1, 2])
    print("DCT!  "); @time dct_dim3_gpu!(f.data)

    print("ϕCALC ");
    @time begin
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            @inbounds ϕ.data[i, j, k] = -f.data[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
        end
        ϕ.data[1, 1, 1] = 0
    end

    print("IFFT! "); @time ifft!(ϕ.data, [1, 2])

    @. ϕ.data = real(ϕ.data) / (2g.Nz)
    # for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
    #     ϕ[i, j, k] = real(ϕ[i, j, k])
    # end

    print("IDCT! "); @time idct_dim3_gpu!(f.data)

    nothing
end

function solve_poisson_3d_ppn_gpu!(Tx, Ty, Bx, By, Bz, g::RegularCartesianGrid, f::CellField, ϕ::CellField, kx², ky², kz², dct_factors, idct_bfactors)
    dct_dim3_gpu!(g, f.data, dct_factors)
    @. f.data = real(f.data)

    fft!(f.data, [1, 2])

    @hascuda @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(Val(:GPU), g.Nx, g.Ny, g.Nz, f.data, ϕ.data, kx², ky², kz²)
    ϕ.data[1, 1, 1] = 0

    ifft!(ϕ.data, [1, 2])
    idct_dim3_gpu!(g, ϕ.data, idct_bfactors)

    nothing
end

function f2ϕ!(::Val{Dev}, Nx, Ny, Nz, f, ϕ, kx², ky², kz²) where Dev
    @setup Dev

    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ϕ[i, j, k] = -f[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
            end
        end
    end

    @synchronize
end
