import FFTW
import GPUifyLoops: @launch, @loop, @synchronize

abstract type PoissonBCs end
struct PPN <: PoissonBCs end  # Periodic BCs in x,y. Neumann BC in z.
struct PNN <: PoissonBCs end  # Periodic BCs in x. Neumann BC in y,z.

PoissonSolver(::CPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverCPU(pbcs, grid)
PoissonSolver(::GPU, pbcs::PoissonBCs, grid::Grid) = PoissonSolverGPU(pbcs, grid)

unpack_grid(grid::Grid) = grid.Nx, grid.Ny, grid.Nz, grid.Lx, grid.Ly, grid.Lz


"""
    ω(M, k)

Return the `M`th root of unity raised to the `k`th power.
"""
@inline ω(M, k) = exp(-2im*π*k/M)

"""
    λi(grid::Grid, ::PoissonBCs)

Return an Nx×1×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the x-dimension on `grid`.
"""
function λi(grid::Grid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    is = reshape(1:Nx, Nx, 1, 1)
    @. (2sin((is-1)*π/Nx) / (Lx/Nx))^2
end

"""
    λj(grid::Grid, ::PPN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with periodic boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::Grid, ::PPN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/Ny) / (Ly/Ny))^2
end

"""
    λj(grid::Grid, ::PNN)

Return an 1×Ny×1 array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λj(grid::Grid, ::PNN)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    js = reshape(1:Ny, 1, Ny, 1)
    @. (2sin((js-1)*π/(2Ny)) / (Ly/Ny))^2
end

"""
    λk(grid::Grid, ::PoissonBCs)

Return an 1×1×Nz array of eigenvalues satisfying the discrete form of Poisson's
equation with staggered Neumann boundary conditions in the y-dimension on `grid`.
"""
function λk(grid::Grid, ::PoissonBCs)
    Nx, Ny, Nz, Lx, Ly, Lz = unpack_grid(grid)
    ks = reshape(1:Nz, 1, 1, Nz)
    @. (2sin((ks-1)*π/(2Nz)) / (Lz/Nz))^2
end

"""
    plan_transforms(::PPN, A::Array; planner_flag=FFTW.PATIENT)

Return the transforms required to solve Poisson's equation with periodic
boundary conditions in the x,y dimensions and staggered Neumann boundary
conditions in the z-dimension.

Fast Fourier transforms (FFTs) are used in the x,y dimensions and real-to-real
discrete cosine transforms are used in the z dimension. Note that the DCT-II is
used for the DCT and the DCT-III for the IDCT which correspond to REDFT10 and
REDFT01 in FFTW.

They operatore on an array with the shape of `A`, which is needed to plan
efficient transforms. `A` will be mutated.

Note that the transforms returns operate on Arrays and so only work on CPUs.
"""
function plan_transforms(::PPN, A::Array, planner_flag=FFTW.PATIENT)
    FFT!  = FFTW.plan_fft!(A, [1, 2]; flags=planner_flag)
    IFFT! = FFTW.plan_ifft!(A, [1, 2]; flags=planner_flag)
    DCT!  = FFTW.plan_r2r!(A, FFTW.REDFT10, 3; flags=planner_flag)
    IDCT! = FFTW.plan_r2r!(A, FFTW.REDFT01, 3; flags=planner_flag)
    return FFT!, IFFT!, DCT!, IDCT!
end

"""
    plan_transforms(::PPN, A::Array; planer_flag=FFTW.PATIENT)

Similar to `plan_transforms(::PPN, ...)` but return the transforms required to
solve Poisson's equation with periodic boundary conditions in the x dimension
and staggered Neumann boundary conditions in the y,z dimensions.
"""
function plan_transforms(::PNN, A::Array, planner_flag=FFTW.PATIENT)
    FFT!  = FFTW.plan_fft!(A, 1; flags=planner_flag)
    IFFT! = FFTW.plan_ifft!(A, 1; flags=planner_flag)
    DCT!  = FFTW.plan_r2r!(A, FFTW.REDFT10, [2, 3]; flags=planner_flag)
    IDCT! = FFTW.plan_r2r!(A, FFTW.REDFT01, [2, 3]; flags=planner_flag)
    return FFT!, IFFT!, DCT!, IDCT!
end

struct PoissonSolverCPU{BC, AAR, AAC, FFTT, DCTT, IFFTT, IDCTT} <: PoissonSolver
    bcs::BC
    kx²::AAR
    ky²::AAR
    kz²::AAR
    storage::AAC
    FFT!::FFTT
    DCT!::DCTT
    IFFT!::IFFTT
    IDCT!::IDCTT
end

function PoissonSolverCPU(pbcs::PoissonBCs, grid::Grid, planner_flag=FFTW.PATIENT)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    # The eigenvalues of the discrete form of Poisson's equation correspond
    # to discrete wavenumbers so we call them k² to make the analogy with
    # solving Poisson's equation in spectral space.
    kx² = λi(grid, pbcs)
    ky² = λj(grid, pbcs)
    kz² = λk(grid, pbcs)

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz)

    FFT!, IFFT!, DCT!, IDCT! = plan_transforms(pbcs, storage, planner_flag)

    PoissonSolverCPU(pbcs, kx², ky², kz², storage, FFT!, DCT!, IFFT!, IDCT!)
end

normalize_idct_output(::PPN, grid::Grid, ϕ::Array) = (@. ϕ = ϕ / (2grid.Nz))
normalize_idct_output(::PNN, grid::Grid, ϕ::Array) = (@. ϕ = ϕ / (4grid.Ny*grid.Nz))

"""
    solve_poisson_3d!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)

Solve Poisson equation on a uniform staggered grid (Arakawa C-grid) with
appropriate boundary conditions as specified by `solver.bcs` using planned FFTs
and DCTs. The right-hand-side RHS is stored in solver.storage which the solver
mutates to produce the solution, so it will also be stored in solver.storage.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_3d!(solver::PoissonSolverCPU, grid::RegularCartesianGrid)
    Nx, Ny, Nz, _ = unpack_grid(grid)
    kx², ky², kz² = solver.kx², solver.ky², solver.kz²

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    solver.DCT! * RHS  # Calculate DCTᶻ(f) in place.
    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are essentially
    # computing the Fourier coefficients of the solution from the Fourier
    # coefficients of the RHS.
    @. ϕ = -RHS / (kx² + ky² + kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ) in place.
    solver.IDCT! * ϕ  # Calculate IDCTᶻ(ϕ) in place.

    # Must normalize by 2N for each dimension transformed via FFTW.REDFT.
    normalize_idct_output(solver.bcs, grid, ϕ)

    nothing
end

struct PoissonSolverGPU{BC, AAR, AAC, FFT, FFTD, IFFT, IFFTD} <: PoissonSolver
    bcs::BC
    kx²::AAR
    ky²::AAR
    kz²::AAR
    ω_4Ny⁺::AAC
    ω_4Ny⁻::AAC
    ω_4Nz⁺::AAC
    ω_4Nz⁻::AAC
    storage::AAC
    FFT!::FFT
    FFT_DCT!::FFTD
    IFFT!::IFFT
    IFFT_DCT!::IFFTD
end

function PoissonSolverGPU(pbcs::PoissonBCs, grid::Grid)
    Nx, Ny, Nz, _ = unpack_grid(grid)

    # The eigenvalues of the discrete form of Poisson's equation correspond
    # to discrete wavenumbers so we call them k² to make the analogy with
    # solving Poisson's equation in spectral space.
    kx² = λi(grid, pbcs) |> CuArray
    ky² = λj(grid, pbcs) |> CuArray
    kz² = λk(grid, pbcs) |> CuArray

    # Storage for RHS and Fourier coefficients is hard-coded to be Float64
    # because of precision issues with Float32.
    # See https://github.com/climate-machine/Oceananigans.jl/issues/55
    storage = zeros(Complex{Float64}, grid.Nx, grid.Ny, grid.Nz) |> CuArray

    ky⁺ = reshape(0:Ny-1,       1, Ny, 1)
    kz⁺ = reshape(0:Nz-1,       1, 1, Nz)
    ky⁻ = reshape(0:-1:-(Ny-1), 1, Ny, 1)
    kz⁻ = reshape(0:-1:-(Nz-1), 1, 1, Nz)

    ω_4Ny⁺ = ω.(4Ny, ky⁺) |> CuArray
    ω_4Nz⁺ = ω.(4Nz, kz⁺) |> CuArray
    ω_4Ny⁻ = ω.(4Ny, ky⁻) |> CuArray
    ω_4Nz⁻ = ω.(4Nz, kz⁻) |> CuArray

    ω_4Nz⁻[1] *= 1/2

    if pbcs == PPN()
        FFT!      = plan_fft!(storage, [1, 2])
        FFT_DCT!  = plan_fft!(storage, 3)
        IFFT!     = plan_ifft!(storage, [1, 2])
        IFFT_DCT! = plan_ifft!(storage, 3)
    elseif pbcs == PNN()
        FFT!      = plan_fft!(storage, 1)
        FFT_DCT!  = plan_fft!(storage, [2, 3])
        IFFT!     = plan_ifft!(storage, 1)
        IFFT_DCT! = plan_ifft!(storage, [2, 3])
    end

    PoissonSolverGPU(pbcs, kx², ky², kz², ω_4Ny⁺, ω_4Ny⁻, ω_4Nz⁺, ω_4Nz⁻,
                     storage, FFT!, FFT_DCT!, IFFT!, IFFT_DCT!)
end

"""
    solve_poisson_3d!(solver::PoissonSolverGPU, grid::RegularCartesianGrid)

Similar to solve_poisson_3d!(solver::PoissonSolverCPU, ...) except that since
the discrete cosine transform is not available through cuFFT, we perform our
own fast cosine transform (FCT) via an algorithm that utilizes the FFT.

Note that for the FCT algorithm to work, the input must have been permuted along
the dimension the FCT is to be calculated by ordering the odd elements first
followed by the even elements. For example,

    [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]

The output will be permuted in this way and so the permutation must be undone.

We should describe the algorithm in detail in the documentation.
"""
function solve_poisson_3d!(Tx, Ty, Bx, By, Bz, solver::PoissonSolverGPU{<:PPN}, grid::RegularCartesianGrid)
    ω_4Ny⁺, ω_4Ny⁻, ω_4Nz⁺, ω_4Nz⁻ = solver.ω_4Ny⁺, solver.ω_4Ny⁻, solver.ω_4Nz⁺, solver.ω_4Nz⁻

    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS
    @. RHS = 2 * real(ω_4Nz⁺ * RHS)

    solver.FFT! * RHS  # Calculate FFTˣʸ(f) in place.

    # Solve the discrete Poisson equation in spectral space. We are essentially
    # computing the Fourier coefficients of the solution from the Fourier
    # coefficients of the RHS.
    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(grid, RHS, ϕ, solver.kx², solver.ky², solver.kz²)

    # Setting DC component of the solution (the mean) to be zero. This is also
    # necessary because the source term to the Poisson equation has zero mean
    # and so the DC component comes out to be ∞.
    ϕ[1, 1, 1] = 0

    solver.IFFT! * ϕ  # Calculate IFFTˣʸ(ϕ̂) in place.

    # Calculate IDCTᶻ(ϕ̂) in place using the FFT.
    ϕ .*= ω_4Nz⁻
    solver.IFFT_DCT! * ϕ

    nothing
end

function solve_poisson_3d!(Tx, Ty, Bx, By, Bz, solver::PoissonSolverGPU{<:PNN}, grid::RegularCartesianGrid)
    # We can use the same storage for the RHS and the solution ϕ.
    RHS, ϕ = solver.storage, solver.storage

    # Calculate DCTʸᶻ(f) in place using the FFT.
    solver.FFT_DCT! * RHS
    RHS .*= solver.dct_factors_z
    RHS .*= solver.dct_factors_y
    @. RHS = real(RHS)

    solver.FFT! * RHS  # Calculate FFTˣ(f) in place.

    @launch device(GPU()) threads=(Tx, Ty) blocks=(Bx, By, Bz) f2ϕ!(grid, RHS, ϕ, solver.kx², solver.ky², solver.kz²)

    ϕ[1, 1, 1] = 0  # Setting DC component of the solution (the mean) to be zero.

    solver.IFFT! * ϕ  # Calculate IFFTˣ(ϕ̂) in place.

    # Calculate IDCTʸᶻ(ϕ̂) in place using the FFT.
    ϕ .*= solver.idct_bfactors_z
    ϕ .*= solver.idct_bfactors_y
    solver.IFFT_DCT! * ϕ

    nothing
end

"Kernel for computing the solution `ϕ` to Poisson equation for source term `f` on a GPU."
function f2ϕ!(grid::Grid, f, ϕ, kx², ky², kz²)
    @loop for k in (1:grid.Nz; blockIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ϕ[i, j, k] = -f[i, j, k] / (kx²[i] + ky²[j] + kz²[k])
            end
        end
    end
    @synchronize
end
